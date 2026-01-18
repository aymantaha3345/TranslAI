"""
Pydantic schemas for request/response validation.
Ensures type safety and data validation at API boundaries.
"""

from datetime import datetime
from typing import Optional, Literal, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator

class EnhancementLevel(str, Enum):
    """Available enhancement levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ImageProviderType(str, Enum):
    """Supported image generation providers."""
    OPENAI = "openai"
    STABILITY = "stability"
    CUSTOM = "custom"

class GenerateImageRequest(BaseModel):
    """Request schema for image generation endpoint."""
    
    prompt: str = Field(
        ...,
        description="The text prompt in any human language",
        min_length=1,
        max_length=2000
    )
    
    enhance: bool = Field(
        default=False,
        description="Whether to enhance the translated prompt for better visual quality"
    )
    
    enhancement_level: Optional[EnhancementLevel] = Field(
        default=None,
        description="Level of enhancement to apply (only used if enhance=true)"
    )
    
    image_model: Optional[str] = Field(
        default=None,
        description="Override the default image generation model"
    )
    
    image_provider: Optional[ImageProviderType] = Field(
        default=None,
        description="Override the default image generation provider"
    )
    
    image_size: Optional[Literal["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]] = Field(
        default=None,
        description="Override the default image size"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to include in the response"
    )
    
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt content."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()

class LanguageDetectionResult(BaseModel):
    """Result of language detection."""
    
    language: str = Field(
        ...,
        description="Detected language code (ISO 639-1 format)",
        pattern="^[a-z]{2}$"
    )
    
    confidence: float = Field(
        ...,
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    
    language_name: str = Field(
        ...,
        description="Full name of the detected language"
    )

class PromptProcessingResult(BaseModel):
    """Result of prompt processing pipeline."""
    
    original_prompt: str = Field(..., description="Original prompt in source language")
    detected_language: LanguageDetectionResult = Field(..., description="Language detection result")
    translated_prompt: str = Field(..., description="Translated prompt in English")
    enhanced_prompt: Optional[str] = Field(
        default=None,
        description="Enhanced version of the translated prompt (if enhancement was applied)"
    )
    enhancement_applied: bool = Field(
        default=False,
        description="Whether enhancement was actually applied"
    )
    enhancement_level: Optional[EnhancementLevel] = Field(
        default=None,
        description="Level of enhancement applied"
    )

class ImageGenerationResult(BaseModel):
    """Result of image generation."""
    
    image_url: str = Field(
        ...,
        description="URL to the generated image"
    )
    
    image_data: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (if returned instead of URL)"
    )
    
    model_used: str = Field(
        ...,
        description="Image generation model that was used"
    )
    
    provider_used: str = Field(
        ...,
        description="Image generation provider that was used"
    )
    
    generation_time: float = Field(
        ...,
        description="Time taken for image generation in seconds"
    )

class GenerateImageResponse(BaseModel):
    """Complete response for image generation endpoint."""
    
    request_id: str = Field(
        ...,
        description="Unique identifier for this request"
    )
    
    original_prompt: str = Field(..., description="Original prompt in source language")
    detected_language: LanguageDetectionResult = Field(..., description="Language detection result")
    translated_prompt: str = Field(..., description="Translated prompt in English")
    enhanced_prompt: Optional[str] = Field(
        default=None,
        description="Enhanced version of the translated prompt (if enhancement was applied)"
    )
    
    enhancement_applied: bool = Field(
        default=False,
        description="Whether enhancement was actually applied"
    )
    
    enhancement_level: Optional[EnhancementLevel] = Field(
        default=None,
        description="Level of enhancement applied"
    )
    
    image_result: ImageGenerationResult = Field(..., description="Image generation result")
    
    providers_used: Dict[str, str] = Field(
        default_factory=dict,
        description="Providers used during processing (text, image)"
    )
    
    processing_time: float = Field(
        ...,
        description="Total processing time in seconds"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the request completion"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata from the request"
    )

class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the error"
    )