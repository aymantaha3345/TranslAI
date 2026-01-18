"""
Structured logging configuration for TRANSLai.
Provides request ID correlation, structured JSON output, and environment-aware logging.
"""

import json
import logging
import sys
import time
import uuid
from typing import Any, Dict, Optional
from datetime import datetime
import contextvars

from loguru import logger
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

# Context variable for request ID correlation
request_id_ctx = contextvars.ContextVar("request_id", default="")

class StructuredLogger:
    """
    Production-ready structured logger with JSON formatting and correlation IDs.
    """
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging based on environment settings."""
        logger.remove()  # Remove default handler
        
        # Configure based on environment
        if settings.app_env == "production":
            self._setup_production_logging()
        else:
            self._setup_development_logging()
    
    def _setup_production_logging(self):
        """Production logging with JSON formatting."""
        def json_formatter(record: dict) -> str:
            """Custom JSON formatter with correlation IDs."""
            # Get request_id safely - use default if not present
            request_id = record["extra"].get("request_id", request_id_ctx.get() or "")
            
            log_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record["level"].name,
                "message": record["message"],
                "logger": record["name"],
                "request_id": request_id,  # Use the safely retrieved value
                "environment": settings.app_env,
                "service": "TRANSLai"
            }
            
            # Add exception info if present
            if record["exception"]:
                log_data["exception"] = {
                    "type": type(record["exception"]).__name__,
                    "message": str(record["exception"]),
                    "traceback": str(getattr(record["exception"], "__traceback__", ""))
                }
            
            # Add extra context if available
            if record["extra"]:
                # Merge extra fields but avoid duplicates
                for key, value in record["extra"].items():
                    if key not in ["request_id"]:  # Skip request_id as we already added it
                        log_data[key] = value
            
            return json.dumps(log_data, ensure_ascii=False) + "\n"
        
        logger.add(
            sys.stdout,
            format=json_formatter,
            level=settings.log_level,
            serialize=False,
            backtrace=False,
            diagnose=False
        )
        
        # Add file logging for production
        logger.add(
            "logs/translai.log",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            format=json_formatter,
            level="INFO",
            serialize=False,
            enqueue=True  # Async logging for better performance
        )
    
    def _setup_development_logging(self):
        """Development-friendly logging with color and detailed formatting."""
        def dev_formatter(record: dict) -> str:
            """Development formatter that safely handles request_id."""
            request_id = record["extra"].get("request_id", request_id_ctx.get() or "")
            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<yellow>request_id={request_id}</yellow> | "
                "<level>{message}</level>"
            ).format(
                time=record["time"],
                level=record["level"].name,
                name=record["name"],
                function=record["function"],
                line=record["line"],
                request_id=request_id,
                message=record["message"]
            )
        
        logger.add(
            sys.stdout,
            format=dev_formatter,
            level=settings.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    def get_logger(self, name: str = "translai"):
        """Get a logger instance with the given name."""
        return logger.bind(name=name)

# Initialize global logger
structured_logger = StructuredLogger()
app_logger = structured_logger.get_logger("main")

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request ID to all log entries for correlation.
    """
    
    async def dispatch(self, request: Request, call_next):
        """Add request ID to context and process request."""
        # Generate or get request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_ctx.set(request_id)
        
        # Add request ID to logger context
        logger.configure(extra={"request_id": request_id})
        
        # Log request start
        app_logger.info(
            "Request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Log request completion
            app_logger.info(
                "Request completed",
                extra={
                    "status_code": response.status_code,
                    "processing_time": processing_time,
                    "method": request.method,
                    "path": request.url.path
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Log error with full context
            app_logger.error(
                "Request failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "status_code": 500,
                    "processing_time": processing_time,
                    "method": request.method,
                    "path": request.url.path,
                    "stack_trace": str(getattr(e, "__traceback__", ""))
                }
            )
            
            raise

def get_request_logger(request_id: Optional[str] = None):
    """
    Get a logger instance bound to the current request context.
    """
    if request_id is None:
        request_id = request_id_ctx.get()
    
    return logger.bind(request_id=request_id)

def log_processing_step(step_name: str, duration: float, success: bool = True, **kwargs):
    """
    Log a processing step with timing and context.
    """
    request_logger = get_request_logger()
    
    log_data = {
        "step": step_name,
        "duration": duration,
        "success": success,
        **kwargs
    }
    
    if success:
        request_logger.debug("Processing step completed", extra=log_data)
    else:
        request_logger.error("Processing step failed", extra=log_data)

def setup_exception_logging():
    """
    Set up global exception logging for unhandled exceptions.
    """
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        request_logger = get_request_logger()
        request_logger.critical(
            "Uncaught exception",
            extra={
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value),
                "traceback": str(exc_traceback)
            }
        )
    
    sys.excepthook = handle_exception

# Setup global exception handling
setup_exception_logging()

# Create logs directory if it doesn't exist (for production file logging)
import os
if not os.path.exists("logs"):
    os.makedirs("logs")