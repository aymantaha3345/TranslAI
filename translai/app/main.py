"""
TRANSLai - Multilingual Prompt Translation & Enhancement Middleware
FastAPI application entry point with comprehensive error handling and observability.
"""

import time
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .config import settings
from .logger import (
    app_logger,
    RequestIDMiddleware,
    get_request_logger,
    setup_exception_logging
)
from .schemas import (
    GenerateImageRequest,
    GenerateImageResponse,
    ErrorResponse
)
from .pipeline import translation_pipeline
from .providers import close_all_providers

# Setup global exception logging
setup_exception_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    # Startup events
    app_logger.info("🚀 TRANSLai application starting up...")
    app_logger.info(f"Environment: {settings.app_env}")
    app_logger.info(f"Debug mode: {'enabled' if settings.debug else 'disabled'}")
    app_logger.info(f"Text provider: {settings.text_provider}")
    app_logger.info(f"Image provider: {settings.image_provider}")
    
    # Validate critical configuration
    if not settings.text_provider_api_key:
        app_logger.warning("⚠️  Text provider API key is not configured")
    
    if not settings.image_provider_api_key:
        app_logger.warning("⚠️  Image provider API key is not configured")
    
    yield
    
    # Shutdown events
    app_logger.info("🛑 TRANSLai application shutting down...")
    await close_all_providers()
    app_logger.info("🔌 All provider connections closed")

# Initialize FastAPI app
app = FastAPI(
    title="TRANSLai",
    description="Multilingual Prompt Translation & Enhancement Middleware for Image Generation Models",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
    openapi_url="/api/openapi.json" if settings.debug else None
)

# Add middleware
app.add_middleware(RequestIDMiddleware)

# Configure CORS
if settings.app_env == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing and outcome."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_logger = get_request_logger(request_id)
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        request_logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "processing_time": processing_time,
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        request_logger.error(
            "Request failed with exception",
            extra={
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": processing_time,
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        # Return standardized error response
        error_response = ErrorResponse(
            error="Internal server error",
            code="INTERNAL_ERROR",
            details={"message": str(e)}
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump()
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "version": "1.0.0",
        "timestamp": time.time()
    }

@app.get("/api/config")
async def get_config():
    """Get application configuration (debug only)."""
    if not settings.debug:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Configuration endpoint only available in debug mode"
        )
    
    return {
        "text_provider": settings.text_provider,
        "image_provider": settings.image_provider,
        "enhancement_enabled": True,
        "max_prompt_length": settings.max_prompt_length,
        "environment": settings.app_env
    }

@app.post("/api/v1/generate", response_model=GenerateImageResponse)
async def generate_image(request: GenerateImageRequest, req: Request):
    """
    Generate image from multilingual prompt.
    
    This endpoint:
    1. Detects the language of the input prompt
    2. Translates it to English
    3. Optionally enhances the prompt for better visual quality
    4. Generates an image using the configured image provider
    5. Returns the result with metadata
    
    Args:
        request: GenerateImageRequest containing prompt and options
        req: FastAPI Request object for context
        
    Returns:
        GenerateImageResponse with image URL and processing metadata
    """
    request_id = req.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_logger = get_request_logger(request_id)
    
    # Validate prompt length
    if len(request.prompt) > settings.max_prompt_length:
        error_response = ErrorResponse(
            error=f"Prompt exceeds maximum length of {settings.max_prompt_length} characters",
            code="INVALID_PROMPT_LENGTH"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response.model_dump()
        )
    
    request_logger.info(
        "Processing image generation request",
        extra={
            "prompt_preview": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt,
            "enhance": request.enhance,
            "enhancement_level": request.enhancement_level.value if request.enhancement_level else None,
            "image_model": request.image_model,
            "image_provider": request.image_provider.value if request.image_provider else None
        }
    )
    
    try:
        # Process the complete request
        response = await translation_pipeline.process_request(request, request_id)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        request_logger.error(
            "Unexpected error during image generation",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
            }
        )
        
        error_response = ErrorResponse(
            error="Failed to generate image",
            code="IMAGE_GENERATION_FAILED",
            details={"message": str(e)}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standardized response."""
    request_logger = get_request_logger()
    
    error_response = ErrorResponse(
        error=exc.detail.get("error", "Unknown error") if isinstance(exc.detail, dict) else str(exc.detail),
        code=exc.detail.get("code", "UNKNOWN_ERROR") if isinstance(exc.detail, dict) else "HTTP_ERROR",
        details=exc.detail.get("details", {}) if isinstance(exc.detail, dict) else {"status_code": exc.status_code}
    )
    
    request_logger.warning(
        "HTTP exception occurred",
        extra={
            "status_code": exc.status_code,
            "error": error_response.error,
            "code": error_response.code
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions globally."""
    request_logger = get_request_logger()
    
    error_response = ErrorResponse(
        error="Internal server error",
        code="INTERNAL_ERROR",
        details={"message": str(exc), "type": type(exc).__name__}
    )
    
    request_logger.critical(
        "Unhandled exception occurred",
        extra={
            "error": str(exc),
            "error_type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )

@app.get("/api/v1/providers/text")
async def get_text_providers():
    """Get available text LLM providers."""
    return {
        "providers": [
            {"name": "OpenAI", "value": "openai"},
            {"name": "Qwen", "value": "qwen"},
            {"name": "DeepSeek", "value": "deepseek"},
            {"name": "Custom", "value": "custom"}
        ]
    }

@app.get("/api/v1/providers/image")
async def get_image_providers():
    """Get available image generation providers."""
    return {
        "providers": [
            {"name": "OpenAI DALL-E", "value": "openai"},
            {"name": "Stability AI", "value": "stability"},
            {"name": "Custom", "value": "custom"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    app_logger.info("Starting TRANSLai server...")
    uvicorn.run(
        "translai.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=4 if not settings.debug else 1,
        log_level=settings.log_level.lower(),
        access_log=False  # Disable access log to avoid duplication with our middleware
    )