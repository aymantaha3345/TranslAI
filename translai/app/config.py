"""
Configuration management for TRANSLai system.
Uses Pydantic Settings for type-safe, validated configuration loading from environment variables.
"""

import os
from enum import Enum
from typing import Dict, Optional, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, ValidationError
from loguru import logger

class EnhancementLevel(str, Enum):
    """Available enhancement levels for prompt enhancement."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

class TextProviderType(str, Enum):
    """Supported text LLM provider types."""
    OPENAI = "openai"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"

class ImageProviderType(str, Enum):
    """Supported image generation provider types."""
    OPENAI = "openai"
    STABILITY = "stability"
    CUSTOM = "custom"

class Settings(BaseSettings):
    """Main application settings with validation and defaults."""
    
    # Application settings
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    request_timeout: float = Field(
        default=30.0,
        description="Default timeout for external API requests in seconds",
        ge=1.0,
        le=300.0
    )
    max_prompt_length: int = Field(
        default=2000,
        description="Maximum allowed prompt length in characters",
        ge=100,
        le=10000
    )
    
    # Text LLM Provider Configuration
    text_provider: TextProviderType = Field(
        default=TextProviderType.OPENAI,
        description="Default text LLM provider for translation and enhancement"
    )
    text_provider_api_key: str = Field(
        default="",
        description="API key for text LLM provider",
        min_length=10
    )
    text_provider_base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for text LLM provider (for custom/compatible endpoints)"
    )
    text_provider_model: str = Field(
        default="gpt-4o-mini",
        description="Model name to use for text LLM provider"
    )
    text_provider_temperature: float = Field(
        default=0.3,
        description="Temperature for text generation (lower = more deterministic)",
        ge=0.0,
        le=1.0
    )
    
    # Image Generation Provider Configuration
    image_provider: ImageProviderType = Field(
        default=ImageProviderType.OPENAI,
        description="Default image generation provider"
    )
    image_provider_api_key: str = Field(
        default="",
        description="API key for image generation provider",
        min_length=10
    )
    image_provider_base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for image generation provider"
    )
    image_provider_model: str = Field(
        default="dall-e-3",
        description="Default model for image generation"
    )
    image_size: Literal["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"] = Field(
        default="1024x1024",
        description="Default image size for generation"
    )
    
    # Enhancement Configuration
    default_enhancement_level: EnhancementLevel = Field(
        default=EnhancementLevel.MEDIUM,
        description="Default enhancement level when enhancement is enabled"
    )
    enhancement_system_prompt: str = Field(
        default=(
            "You are a professional prompt engineer for image generation. "
            "Your task is to enhance the visual quality of prompts while STRICTLY PRESERVING USER INTENT. "
            "You MAY improve: lighting, composition, visual clarity, realism, and technical quality. "
            "You MUST NOT: add new objects, change the main subject, alter artistic intent, or change style unless explicitly requested. "
            "Always maintain the core meaning and creative vision of the original prompt."
        ),
        description="System prompt for prompt enhancement"
    )
    
    # Language Detection Configuration
    language_detection_threshold: float = Field(
        default=0.8,
        description="Confidence threshold for language detection",
        ge=0.0,
        le=1.0
    )
    
    # Security Configuration
    rate_limit_requests: int = Field(
        default=100,
        description="Maximum requests per minute per client",
        ge=1,
        le=1000
    )
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds",
        ge=10,
        le=300
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("debug")
    @classmethod
    def set_debug_based_on_env(cls, v: bool, info):
        """Automatically set debug mode based on environment."""
        env = info.data.get("app_env", "development")
        if env == "development":
            return True
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str):
        """Validate log level is one of the standard levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {', '.join(valid_levels)}")
        return v.upper()
    
    def get_provider_config(self, provider_type: str) -> Dict:
        """Get configuration dictionary for a specific provider type."""
        if provider_type == "text":
            return {
                "provider": self.text_provider.value,
                "api_key": self.text_provider_api_key,
                "base_url": self.text_provider_base_url,
                "model": self.text_provider_model,
                "temperature": self.text_provider_temperature,
                "timeout": self.request_timeout
            }
        elif provider_type == "image":
            return {
                "provider": self.image_provider.value,
                "api_key": self.image_provider_api_key,
                "base_url": self.image_provider_base_url,
                "model": self.image_provider_model,
                "size": self.image_size,
                "timeout": self.request_timeout
            }
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

try:
    settings = Settings()
    logger.info("Configuration loaded successfully")
    logger.debug(f"Active environment: {settings.app_env}")
except ValidationError as e:
    logger.error(f"Configuration validation failed: {e}")
    raise
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

# Environment-specific adjustments
if settings.app_env == "production":
    # Production-specific security and performance settings
    settings.debug = False
    settings.log_level = "INFO"
elif settings.app_env == "development":
    # Development-friendly settings
    settings.debug = True
    settings.log_level = "DEBUG"