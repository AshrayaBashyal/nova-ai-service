import time
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

# Initialize logger specifically for this layer
logger = logging.getLogger("app.middleware")

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Performance Monitoring Middleware.
    Separated from config to keep the codebase modular.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Pass the request to the next handler (the API route)
        response = await call_next(request)
        
        # Measure duration
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Method: {request.method} | Path: {request.url.path} | "
            f"Status: {response.status_code} | Latency: {duration_ms:.2f}ms"
        )
        
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response