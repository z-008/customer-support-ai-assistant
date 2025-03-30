"""
Configuration settings for the Customer Support AI Assistant
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Debug log to check if API key is loaded
api_key = os.getenv("GROQ_API_KEY")
logger.info(
    f"GROQ API Key loaded: {bool(api_key)}"
)  # Don't log the actual key for security


class Settings(BaseSettings):
    """Application settings"""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL_NAME: str = "llama-3.1-8b-instant"  # Using llama model from Groq
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    VECTOR_DIMENSION: int = 768
    MAX_HISTORY_LENGTH: int = 5
    MAX_RETRIEVED_DOCS: int = 3
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000

    # Database settings
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "sqlite")  # "sqlite" or "csv"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "customer_support.db")
    CSV_STORAGE_PATH: str = os.getenv("CSV_STORAGE_PATH", "interactions.csv")

    # Add these to your existing settings
    ENABLE_RESPONSE_CACHE: bool = bool(os.getenv("ENABLE_RESPONSE_CACHE", "false"))
    CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "1"))
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))

    class Config:
        env_file = ".env"


settings = Settings()
