"""
Prompt Processing Pipeline.
Coordinates language detection, translation, enhancement, and image generation.
Implements strict intent preservation rules for prompt enhancement.
"""

import time
import re
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum

from langdetect import detect_langs  # ✅ Correct import
from pydantic import ValidationError

from .config import settings, EnhancementLevel
from .schemas import (
    LanguageDetectionResult,
    PromptProcessingResult,
    GenerateImageRequest,
    GenerateImageResponse,
    ImageGenerationResult
)
from .providers import (
    get_text_provider,
    ProviderResponse,
    ProviderConfig,
    ProviderType
)
from .image_gateway import (
    get_image_provider,
    ImageGenerationResponse
)
from .logger import app_logger, log_processing_step, get_request_logger

class ProcessingStage(str, Enum):
    """Stages in the processing pipeline."""
    LANGUAGE_DETECTION = "language_detection"
    TRANSLATION = "translation"
    ENHANCEMENT = "enhancement"
    IMAGE_GENERATION = "image_generation"
    COMPLETION = "completion"

class PromptEnhancer:
    """
    Handles prompt enhancement with strict intent preservation rules.
    Uses LLM to improve visual quality without changing user intent.
    """
    
    def __init__(self):
        """Initialize prompt enhancer."""
        self.logger = app_logger.bind(component="prompt_enhancer")
        self.enhancement_prompts = {
            EnhancementLevel.LOW: (
                "Enhance this image generation prompt by slightly improving lighting, composition, "
                "and visual clarity. Keep the core subject and artistic intent completely unchanged. "
                "Only make minimal improvements to visual quality."
            ),
            EnhancementLevel.MEDIUM: (
                "Enhance this image generation prompt by improving lighting, composition, "
                "and visual clarity while maintaining the exact same subject and artistic intent. "
                "Focus on making the image more visually appealing without adding new elements "
                "or changing the core meaning."
            ),
            EnhancementLevel.HIGH: (
                "Enhance this image generation prompt by significantly improving lighting, composition, "
                "realism, and visual clarity. Maintain the exact same subject, style, and artistic intent. "
                "Make the image more professional and visually compelling while preserving all original elements "
                "and creative vision."
            )
        }
    
    async def enhance_prompt(
        self,
        prompt: str,
        level: EnhancementLevel = EnhancementLevel.MEDIUM,
        detected_language: Optional[str] = None
    ) -> str:
        """
        Enhance a prompt while strictly preserving user intent.
        
        Args:
            prompt: English prompt to enhance
            level: Enhancement level (low/medium/high)
            detected_language: Original language for context
            
        Returns:
            Enhanced prompt
        """
        start_time = time.time()
        request_logger = get_request_logger()
        
        try:
            # Get enhancement instruction
            enhancement_instruction = self.enhancement_prompts.get(level, self.enhancement_prompts[EnhancementLevel.MEDIUM])
            
            # Add language context if available
            if detected_language:
                enhancement_instruction += f"\n\nNote: Original prompt was in {detected_language} language."
            
            # Get text provider
            provider = await get_text_provider()
            
            # Generate enhanced prompt
            response: ProviderResponse = await provider.generate_text(
                prompt=prompt,
                system_prompt=settings.enhancement_system_prompt + "\n\n" + enhancement_instruction,
                max_tokens=500,
                temperature=0.4  # Slightly higher for creativity
            )
            
            enhanced_prompt = response.content.strip()
            
            # Validate enhancement against intent preservation rules
            validation_result = self._validate_enhancement(prompt, enhanced_prompt)
            
            if not validation_result["valid"]:
                request_logger.warning(
                    "Enhancement failed validation, using original prompt",
                    extra={
                        "original_prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "validation_errors": validation_result["errors"]
                    }
                )
                enhanced_prompt = prompt
            
            processing_time = time.time() - start_time
            
            log_processing_step(
                "prompt_enhancement",
                processing_time,
                success=True,
                enhancement_level=level.value,
                original_length=len(prompt),
                enhanced_length=len(enhanced_prompt),
                validation_passed=validation_result["valid"]
            )
            
            return enhanced_prompt
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            log_processing_step(
                "prompt_enhancement",
                processing_time,
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
            
            request_logger.error(f"Prompt enhancement failed: {str(e)}")
            # Fall back to original prompt on failure
            return prompt
    
    def _validate_enhancement(self, original_prompt: str, enhanced_prompt: str) -> Dict[str, Any]:
        """
        Validate that enhancement preserves user intent.
        
        Checks for:
        - No new objects added
        - No subject changes
        - No artistic intent changes
        - No style changes (unless explicitly requested)
        
        Returns:
            Dict with validation results
        """
        result = {"valid": True, "errors": []}
        
        # Simple validation rules - in production, this would be more sophisticated
        original_words = set(re.findall(r'\b\w+\b', original_prompt.lower()))
        enhanced_words = set(re.findall(r'\b\w+\b', enhanced_prompt.lower()))
        
        # Check for completely new words that might indicate new objects
        new_words = enhanced_words - original_words
        suspicious_words = [
            "person", "man", "woman", "child", "animal", "car", "building", "tree", "flower",
            "water", "sky", "mountain", "ocean", "beach", "city", "house", "dog", "cat"
        ]
        
        for word in new_words:
            if word in suspicious_words and len(word) > 3:
                result["valid"] = False
                result["errors"].append(f"Potentially added new object: {word}")
        
        # Check for significant length changes that might indicate added content
        original_length = len(original_prompt)
        enhanced_length = len(enhanced_prompt)
        
        if enhanced_length > original_length * 2:  # More than double the length
            result["valid"] = False
            result["errors"].append("Enhanced prompt is significantly longer than original")
        
        # Check for style changes (simplified)
        style_indicators = ["painting", "drawing", "photograph", "digital art", "illustration", "sketch"]
        original_styles = [word for word in original_words if word in style_indicators]
        enhanced_styles = [word for word in enhanced_words if word in style_indicators]
        
        if original_styles and enhanced_styles and set(original_styles) != set(enhanced_styles):
            result["valid"] = False
            result["errors"].append("Style appears to have changed")
        
        return result

class TranslationPipeline:
    """
    Main pipeline for processing prompts through translation, enhancement, and image generation.
    Coordinates all components and ensures proper error handling and logging.
    """
    
    def __init__(self):
        """Initialize the translation pipeline."""
        self.logger = app_logger.bind(component="translation_pipeline")
        self.prompt_enhancer = PromptEnhancer()
    
    async def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of the input text using langdetect.
        
        Args:
            text: Text to detect language for
            
        Returns:
            LanguageDetectionResult with detected language and confidence
        """
        start_time = time.time()
        request_logger = get_request_logger()
        
        try:
            # Use langdetect
            detections = detect_langs(text)
            
            if detections:
                # Take the first result (highest confidence)
                language_code = detections[0].lang
                confidence = detections[0].prob
            else:
                # Default to English if detection fails
                language_code = "en"
                confidence = 0.0
            
            # Map language codes to full names
            language_names = {
                "en": "English",
                "es": "Spanish", 
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "zh": "Chinese",
                "ja": "Japanese",
                "ko": "Korean",
                "ar": "Arabic"
            }
            
            language_name = language_names.get(language_code, language_code.upper())
            
            result = LanguageDetectionResult(
                language=language_code,
                confidence=confidence,
                language_name=language_name
            )
            
            processing_time = time.time() - start_time
            
            log_processing_step(
                "language_detection",
                processing_time,
                success=True,
                detected_language=language_code,
                confidence=confidence,
                text_length=len(text)
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            log_processing_step(
                "language_detection",
                processing_time,
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
            
            request_logger.error(f"Language detection failed: {str(e)}")
            # Fall back to English on failure
            return LanguageDetectionResult(
                language="en",
                confidence=0.0,
                language_name="English"
            )
    
    async def translate_prompt(
        self,
        prompt: str,
        source_language: str,
        target_language: str = "en"
    ) -> str:
        """
        Translate prompt from source language to target language.
        
        Args:
            prompt: Prompt to translate
            source_language: Source language code
            target_language: Target language code (default: English)
            
        Returns:
            Translated prompt
        """
        start_time = time.time()
        request_logger = get_request_logger()
        
        try:
            if source_language.lower() == target_language.lower():
                request_logger.debug("Source and target languages are the same, skipping translation")
                return prompt
            
            # Get text provider
            provider = await get_text_provider()
            
            # Prepare translation prompt
            system_prompt = (
                f"You are a professional translator. Translate the following text from {source_language} to {target_language}. "
                f"Maintain the exact same meaning, tone, and style. Do not add or remove any content. "
                f"Only return the translated text, nothing else."
            )
            
            response: ProviderResponse = await provider.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.1  # Low temperature for accuracy
            )
            
            translated_prompt = response.content.strip()
            
            processing_time = time.time() - start_time
            
            log_processing_step(
                "translation",
                processing_time,
                success=True,
                source_language=source_language,
                target_language=target_language,
                original_length=len(prompt),
                translated_length=len(translated_prompt)
            )
            
            return translated_prompt
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            log_processing_step(
                "translation",
                processing_time,
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
            
            request_logger.error(f"Translation failed: {str(e)}")
            # Fall back to original prompt on failure
            return prompt
    
    async def process_prompt(
        self,
        request: GenerateImageRequest,
        request_id: str
    ) -> PromptProcessingResult:
        """
        Process a prompt through the complete pipeline.
        
        Args:
            request: Image generation request
            request_id: Unique request ID for correlation
            
        Returns:
            PromptProcessingResult with all processing stages
        """
        start_time = time.time()
        request_logger = get_request_logger(request_id)
        
        try:
            # Stage 1: Language Detection
            language_detection = await self.detect_language(request.prompt)
            
            # Stage 2: Translation to English
            translated_prompt = await self.translate_prompt(
                request.prompt,
                language_detection.language,
                "en"
            )
            
            # Stage 3: Optional Enhancement
            enhanced_prompt = None
            enhancement_applied = False
            enhancement_level = None
            
            if request.enhance:
                enhancement_level = request.enhancement_level or settings.default_enhancement_level
                enhanced_prompt = await self.prompt_enhancer.enhance_prompt(
                    translated_prompt,
                    level=enhancement_level,
                    detected_language=language_detection.language_name
                )
                enhancement_applied = True
            
            # Create result object
            result = PromptProcessingResult(
                original_prompt=request.prompt,
                detected_language=language_detection,
                translated_prompt=translated_prompt,
                enhanced_prompt=enhanced_prompt if enhancement_applied else None,
                enhancement_applied=enhancement_applied,
                enhancement_level=enhancement_level if enhancement_applied else None
            )
            
            total_time = time.time() - start_time
            
            request_logger.info(
                "Prompt processing completed successfully",
                extra={
                    "stage": ProcessingStage.COMPLETION.value,
                    "total_time": total_time,
                    "detected_language": language_detection.language,
                    "enhancement_applied": enhancement_applied,
                    "enhancement_level": enhancement_level.value if enhancement_level else None
                }
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            
            request_logger.error(
                "Prompt processing failed",
                extra={
                    "stage": "unknown",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "total_time": total_time
                }
            )
            
            # Return partial result on failure
            return PromptProcessingResult(
                original_prompt=request.prompt,
                detected_language=LanguageDetectionResult(
                    language="en",
                    confidence=0.0,
                    language_name="English"
                ),
                translated_prompt=request.prompt,  # Fall back to original
                enhanced_prompt=None,
                enhancement_applied=False,
                enhancement_level=None
            )
    
    async def generate_image(
        self,
        prompt: str,
        request: GenerateImageRequest,
        request_id: str
    ) -> ImageGenerationResult:
        """
        Generate image using the processed prompt.
        
        Args:
            prompt: Final prompt to use for image generation
            request: Original request with configuration
            request_id: Unique request ID for correlation
            
        Returns:
            ImageGenerationResult with image URL and metadata
        """
        start_time = time.time()
        request_logger = get_request_logger(request_id)
        
        try:
            # Get image provider with optional overrides
            provider_config = {}
            
            if request.image_provider:
                provider_config["provider"] = request.image_provider.value
            
            if request.image_model:
                provider_config["model"] = request.image_model
            
            if request.image_size:
                provider_config["size"] = request.image_size
            
            provider = await get_image_provider(custom_config=provider_config)
            
            # Generate image
            response: ImageGenerationResponse = await provider.generate_image(
                prompt=prompt,
                model=request.image_model,
                size=request.image_size
            )
            
            # Create result object
            result = ImageGenerationResult(
                image_url=response.image_url,
                image_data=response.image_data,
                model_used=response.model,
                provider_used=response.provider,
                generation_time=response.processing_time
            )
            
            processing_time = time.time() - start_time
            
            log_processing_step(
                "image_generation",
                processing_time,
                success=True,
                provider=response.provider,
                model=response.model,
                size=response.size,
                prompt_length=len(prompt)
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            log_processing_step(
                "image_generation",
                processing_time,
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
            
            request_logger.error(f"Image generation failed: {str(e)}")
            raise
    
    async def process_request(
        self,
        request: GenerateImageRequest,
        request_id: str
    ) -> GenerateImageResponse:
        """
        Process a complete image generation request.
        
        Args:
            request: Image generation request
            request_id: Unique request ID for correlation
            
        Returns:
            GenerateImageResponse with complete results
        """
        total_start_time = time.time()
        request_logger = get_request_logger(request_id)
        
        try:
            # Step 1: Process prompt (translation + enhancement)
            prompt_result = await self.process_prompt(request, request_id)
            
            # Step 2: Determine which prompt to use for image generation
            final_prompt = (
                prompt_result.enhanced_prompt
                if prompt_result.enhancement_applied and prompt_result.enhanced_prompt
                else prompt_result.translated_prompt
            )
            
            request_logger.info(
                "Using prompt for image generation",
                extra={
                    "prompt": final_prompt[:100] + "..." if len(final_prompt) > 100 else final_prompt,
                    "enhanced": prompt_result.enhancement_applied
                }
            )
            
            # Step 3: Generate image
            image_result = await self.generate_image(final_prompt, request, request_id)
            
            # Step 4: Create final response
            total_time = time.time() - total_start_time
            
            response = GenerateImageResponse(
                request_id=request_id,
                original_prompt=prompt_result.original_prompt,
                detected_language=prompt_result.detected_language,
                translated_prompt=prompt_result.translated_prompt,
                enhanced_prompt=prompt_result.enhanced_prompt,
                enhancement_applied=prompt_result.enhancement_applied,
                enhancement_level=prompt_result.enhancement_level,
                image_result=image_result,
                providers_used={
                    "text": settings.text_provider.value,
                    "image": settings.image_provider.value
                },
                processing_time=total_time,
                metadata=request.metadata
            )
            
            request_logger.info(
                "Request completed successfully",
                extra={
                    "total_time": total_time,
                    "providers_used": response.providers_used,
                    "image_url": image_result.image_url[:100] + "..." if image_result.image_url else None
                }
            )
            
            return response
            
        except Exception as e:
            total_time = time.time() - total_start_time
            
            request_logger.error(
                "Request processing failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "total_time": total_time,
                    "stage": "unknown"
                }
            )
            
            # Return error response
            raise

# Initialize pipeline instance
translation_pipeline = TranslationPipeline()