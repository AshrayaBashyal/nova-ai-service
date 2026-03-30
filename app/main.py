import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.core.config import settings
from app.core.middleware import LoggingMiddleware

# 1. Initialize the FastAPI App
# We pull the project name directly from our validated Settings.
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Optimized Gemini 2.0 Flash Production Backend",
    version="1.0.0"
)

# 2. Configure CORS (Cross-Origin Resource Sharing)
# This is MANDATORY for frontend-backend communication.
# In production, replace ["*"] with your actual frontend URL (e.g., ["https://myapp.com"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

# 3. Include Routers
# We prefix all chat routes with /api so the URL becomes:
# http://localhost:8000/api/v1/chat/stream
app.include_router(chat_router, prefix="/api", tags=["Generative AI"])

# 4. Health Check Endpoint
# Vital for Docker, Kubernetes, or monitoring tools to see if the server is 'alive'.
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "model": settings.GEMINI_MODEL,
        "api_version": "v1"
    }

# 5. Root Redirect
@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs"  # Points the user to the interactive Swagger UI
    }

# 6. Development Server Launcher
if __name__ == "__main__":
    # We use uvicorn to run the ASGI application.
    # reload=True is great for dev, but should be False in production.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)