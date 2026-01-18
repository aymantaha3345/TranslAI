"""
Image Generation Gateway.
Abstraction layer for image generation providers with configuration-based switching.
"""

import time
import base64
from typing import Dict, Any, Optional, Literal
from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel, Field, field_validator

from .config import settings
from .logger import app_logger, log_processing_step
from .providers import ProviderType

class ImageProviderConfig(BaseModel):
    """Configuration for image generation providers."""
    
    provider: ProviderType = Field(..., description="Provider type")
    api_key: str = Field(..., description="API key", min_length=10)
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL (for custom/compatible endpoints)"
    )
    model: str = Field(..., description="Model name to use")
    size: Literal["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"] = Field(
        default="1024x1024",
        description="Image size"
    )
    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds",
        ge=1.0,
        le=300.0
    )
    quality: Literal["standard", "hd"] = Field(
        default="standard",
        description="Image quality (OpenAI specific)"
    )
    style: Optional[Literal["vivid", "natural"]] = Field(
        default=None,
        description="Image style (OpenAI specific)"
    )
    
    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate base URL if provided."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")
        return v

class ImageGenerationResponse(BaseModel):
    """Standardized response from image generation providers."""
    
    image_url: str = Field(..., description="URL to the generated image")
    image_data: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (if returned instead of URL)"
    )
    provider: str = Field(..., description="Provider name")
    model: str = Field(..., description="Model used")
    size: str = Field(..., description="Image size generated")
    usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Usage metrics and cost information"
    )
    processing_time: float = Field(
        ...,
        description="Time taken for image generation in seconds"
    )
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw response from the provider (for debugging)"
    )

class BaseImageProvider(ABC):
    """
    Abstract base class for image generation providers.
    All providers must implement the generate_image method.
    """
    
    def __init__(self, config: ImageProviderConfig):
        """Initialize provider with configuration."""
        self.config = config
        self.logger = app_logger.bind(provider=config.provider.value)
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            follow_redirects=True
        )
    
    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Optional[str] = None,
        **kwargs
    ) -> ImageGenerationResponse:
        """
        Generate an image using the provider.
        
        Args:
            prompt: English prompt for image generation
            model: Optional model override
            size: Optional size override
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ImageGenerationResponse with image URL and metadata
        """
        pass
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

class OpenAIImageProvider(BaseImageProvider):
    """
    OpenAI DALL-E image generation provider.
    """
    
    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Optional[str] = None,
        **kwargs
    ) -> ImageGenerationResponse:
        """Generate image using OpenAI DALL-E API."""
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": model or self.config.model,
                "prompt": prompt,
                "n": 1,
                "size": size or self.config.size,
                "response_format": "url",  # Can be "url" or "b64_json"
                "quality": kwargs.get("quality", self.config.quality),
            }
            
            # Add style if specified and model supports it
            if self.config.style and (model or self.config.model) == "dall-e-3":
                payload["style"] = self.config.style
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            if settings.app_env == "production":
                headers["OpenAI-Organization"] = os.getenv("OPENAI_ORGANIZATION", "")
            
            # Make request
            base_url = self.config.base_url or "https://api.openai.com/v1"
            url = f"{base_url}/images/generations"
            
            self.logger.debug(f"Generating image with prompt: {prompt[:100]}...")
            
            response = await self.client.post(
                url,
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract response content
            image_data = response_data["data"][0]
            image_url = image_data.get("url")
            b64_json = image_data.get("b64_json")
            
            usage = {
                "prompt_tokens": len(prompt.split()),
                "total_tokens": len(prompt.split()),
                "cost": self._calculate_cost(model or self.config.model, size or self.config.size)
            }
            
            processing_time = time.time() - start_time
            
            log_processing_step(
                "openai_image_generate",
                processing_time,
                success=True,
                model=model or self.config.model,
                size=size or self.config.size,
                prompt_length=len(prompt)
            )
            
            return ImageGenerationResponse(
                image_url=image_url,
                image_data=b64_json,
                provider=self.config.provider.value,
                model=model or self.config.model,
                size=size or self.config.size,
                usage=usage,
                processing_time=processing_time,
                raw_response=response_data if settings.debug else None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            log_processing_step(
                "openai_image_generate",
                processing_time,
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
            
            self.logger.error(f"OpenAI image generation failed: {str(e)}")
            raise
    
    def _calculate_cost(self, model: str, size: str) -> float:
        """Calculate estimated cost for image generation (simplified)."""
        # This is a simplified cost calculation - real implementation would use actual pricing
        base_costs = {
            "dall-e-2": {"256x256": 0.016, "512x512": 0.018, "1024x1024": 0.020},
            "dall-e-3": {"1024x1024": 0.040, "1024x1792": 0.080, "1792x1024": 0.080}
        }
        
        model_base = model.split("-")[0] + "-e-" + model.split("-")[2] if "dall-e-" in model else model
        
        return base_costs.get(model_base, {}).get(size, 0.04)

class ImageProviderFactory:
    """
    Factory class for creating image generation providers based on configuration.
    """
    
    @staticmethod
    def create_provider(config: ImageProviderConfig) -> BaseImageProvider:
        """
        Create an image provider based on the configuration.
        
        Args:
            config: Provider configuration
            
        Returns:
            Initialized image provider
            
        Raises:
            ValueError: If provider type is not supported
        """
        provider_map = {
            ProviderType.OPENAI: OpenAIImageProvider,
            # Add other providers here as needed
        }
        
        provider_class = provider_map.get(config.provider)
        
        if not provider_class:
            raise ValueError(f"Unsupported image provider type: {config.provider}")
        
        return provider_class(config)

async def get_image_provider(
    provider_type: Optional[ProviderType] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> BaseImageProvider:
    """
    Get a configured image generation provider instance.
    
    Args:
        provider_type: Optional override for provider type
        custom_config: Optional custom configuration override
        
    Returns:
        Configured image provider instance
    """
    # Get base configuration from settings
    provider_config = settings.get_provider_config("image")
    
    # Apply overrides if provided
    if provider_type:
        provider_config["provider"] = provider_type.value
    
    if custom_config:
        provider_config.update(custom_config)
    
    # Create provider config object
    config = ImageProviderConfig(**provider_config)
    
    # Create and return provider
    return ImageProviderFactory.create_provider(config)