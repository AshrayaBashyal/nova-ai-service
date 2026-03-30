import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Determine the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    Application settings using Pydantic for automatic validation.
    If a required variable is missing from .env, the app will fail 
    immediately on startup with a clear error.
    """
    # API Keys
    GEMINI_API_KEY: str
    
    # Model Settings
    GEMINI_MODEL: str = "gemini-2.0-flash"
    TEMPERATURE: float = 0.7
    MAX_OUTPUT_TOKENS: int = 2048
    
    # Fast API Settings
    PROJECT_NAME: str = "Gemini-FastAPI-Production"
    
    # Tell Pydantic to read from the .env file
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra env vars not defined here
    )

# Create a single instance to be imported elsewhere (Singleton pattern)
settings = Settings()