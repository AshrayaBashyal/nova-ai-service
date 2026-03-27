import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log request path, status, and latency (ms)."""
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000
        
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.1f}ms)"
        )
        return response