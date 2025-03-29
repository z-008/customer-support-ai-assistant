"""
Main FastAPI application for the Customer Support AI Assistant
"""

from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="Customer Support AI Assistant",
    description="An AI-powered customer support assistant using RAG and Groq LLM",
    version="1.0.0",
)

# Include API routes
app.include_router(router, prefix="/api/v1")
