"""
LLM Provider Abstraction Layer.
Defines interfaces and implementations for text LLM providers with configuration-based switching.
"""

import os
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Literal, Union
from enum import Enum

import httpx
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import settings
from .logger import app_logger, log_processing_step

class ProviderType(str, Enum):
    """Supported provider types."""
    OPENAI = "openai"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    
    provider: ProviderType = Field(..., description="Provider type")
    api_key: str = Field(..., description="API key", min_length=10)
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL (for custom/compatible endpoints)"
    )
    model: str = Field(..., description="Model name to use")
    temperature: float = Field(
        default=0.3,
        description="Temperature for text generation",
        ge=0.0,
        le=1.0
    )
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds",
        ge=1.0,
        le=300.0
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum tokens to generate",
        ge=1,
        le=4096
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to use for all requests"
    )
    
    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate base URL if provided."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")
        return v

class ProviderResponse(BaseModel):
    """Standardized response from LLM providers."""
    
    content: str = Field(..., description="Generated text content")
    provider: str = Field(..., description="Provider name")
    model: str = Field(..., description="Model used")
    usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage and other metrics"
    )
    processing_time: float = Field(
        ...,
        description="Time taken for the request in seconds"
    )
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw response from the provider (for debugging)"
    )

class BaseTextProvider(ABC):
    """
    Abstract base class for text LLM providers.
    All providers must implement the generate_text method.
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize provider with configuration."""
        self.config = config
        self.logger = app_logger.bind(provider=config.provider.value)
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            follow_redirects=True
        )
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> ProviderResponse:
        """
        Generate text using the provider.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            max_tokens: Optional max tokens override
            temperature: Optional temperature override
            
        Returns:
            ProviderResponse with generated content and metadata
        """
        pass
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def _get_temperature(self, override: Optional[float] = None) -> float:
        """Get temperature with override capability."""
        return override if override is not None else self.config.temperature
    
    def _get_max_tokens(self, override: Optional[int] = None) -> int:
        """Get max tokens with override capability."""
        return override if override is not None else self.config.max_tokens
    
    def _get_system_prompt(self, override: Optional[str] = None) -> str:
        """Get system prompt with override capability."""
        return override if override is not None else (self.config.system_prompt or "")

class OpenAIProvider(BaseTextProvider):
    """
    OpenAI-compatible provider implementation.
    Works with OpenAI API and any OpenAI-compatible endpoint.
    """
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> ProviderResponse:
        """Generate text using OpenAI-compatible API."""
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt(system_prompt)},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self._get_temperature(temperature),
                "max_tokens": self._get_max_tokens(max_tokens),
                "response_format": {"type": "text"}
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            if self.config.provider == ProviderType.OPENAI:
                headers["OpenAI-Organization"] = os.getenv("OPENAI_ORGANIZATION", "")
            
            # Make request
            base_url = self.config.base_url or "https://api.openai.com/v1"
            url = f"{base_url}/chat/completions"
            
            self.logger.debug(f"Making request to {url}")
            
            response = await self.client.post(
                url,
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract response content
            content = response_data["choices"][0]["message"]["content"].strip()
            usage = response_data.get("usage", {})
            
            processing_time = time.time() - start_time
            
            log_processing_step(
                "openai_generate",
                processing_time,
                success=True,
                model=self.config.model,
                tokens_used=usage.get("total_tokens", 0),
                prompt_length=len(prompt)
            )
            
            return ProviderResponse(
                content=content,
                provider=self.config.provider.value,
                model=self.config.model,
                usage=usage,
                processing_time=processing_time,
                raw_response=response_data if settings.debug else None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            log_processing_step(
                "openai_generate",
                processing_time,
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
            
            self.logger.error(f"OpenAI request failed: {str(e)}")
            raise

class TextProviderFactory:
    """
    Factory class for creating text LLM providers based on configuration.
    """
    
    @staticmethod
    def create_provider(config: ProviderConfig) -> BaseTextProvider:
        """
        Create a text provider based on the configuration.
        
        Args:
            config: Provider configuration
            
        Returns:
            Initialized text provider
            
        Raises:
            ValueError: If provider type is not supported
        """
        provider_map = {
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.QWEN: OpenAIProvider,  # Qwen supports OpenAI API format
            ProviderType.DEEPSEEK: OpenAIProvider,  # DeepSeek supports OpenAI API format
            ProviderType.CUSTOM: OpenAIProvider  # Custom endpoints should be OpenAI-compatible
        }
        
        provider_class = provider_map.get(config.provider)
        
        if not provider_class:
            raise ValueError(f"Unsupported provider type: {config.provider}")
        
        return provider_class(config)

async def get_text_provider(
    provider_type: Optional[ProviderType] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> BaseTextProvider:
    """
    Get a configured text provider instance.
    
    Args:
        provider_type: Optional override for provider type
        custom_config: Optional custom configuration override
        
    Returns:
        Configured text provider instance
    """
    # Get base configuration from settings
    provider_config = settings.get_provider_config("text")
    
    # Apply overrides if provided
    if provider_type:
        provider_config["provider"] = provider_type.value
    
    if custom_config:
        provider_config.update(custom_config)
    
    # Create provider config object
    config = ProviderConfig(**provider_config)
    
    # Create and return provider
    return TextProviderFactory.create_provider(config)

async def close_all_providers():
    """Close all provider connections."""
    # In a real system, we'd track active providers and close them
    pass