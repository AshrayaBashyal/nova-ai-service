from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.core.config import LoggingMiddleware

app = FastAPI(
    title="Gemini Backend",
    description="Production-ready async FastAPI + Gemini.",
    version="0.1.0"
)

# Logging middleware
app.add_middleware(LoggingMiddleware)

# CORS for localhost dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat_router, prefix="/api", tags=["chat"])

@app.get("/")
async def root():
    return {
        "message": "Gemini Backend running",
        "docs_url": "http://localhost:8000/docs",
        "model": "gemini-2.0-flash"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)