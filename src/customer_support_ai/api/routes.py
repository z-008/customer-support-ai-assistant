"""
API routes for the Customer Support AI Assistant
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from ..core.rag_engine import RAGEngine
from ..database import SQLiteStorage, CSVStorage
from ..database.models import Interaction, Conversation
from ..core.config import settings

router = APIRouter()

# Initialize components
rag_engine = RAGEngine()

# Initialize storage based on configuration
if settings.STORAGE_TYPE == "sqlite":
    db = SQLiteStorage(settings.DATABASE_URL)
else:
    db = CSVStorage(settings.CSV_STORAGE_PATH)


class QueryRequest(BaseModel):
    """Request model for generating responses"""

    query: str
    conversation_id: Optional[str] = None
    max_history_length: Optional[int] = settings.MAX_HISTORY_LENGTH


class Response(BaseModel):
    """Response model for generated responses"""

    expanded_query: str
    response: str
    retrieved_documents: List[str]
    evaluation_metrics: Dict[str, float]
    conversation_id: str
    model: str


@router.post("/generate_response", response_model=Response)
async def generate_response(request: QueryRequest):
    """Generate a response for a customer query."""
    try:
        # Get or create conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Get conversation history if available
        history = []
        if conversation_id:
            history = db.get_conversation_history(
                conversation_id, limit=request.max_history_length
            )

        # Generate response using RAG
        formatted_history = ""
        if history:
            for h in history:
                formatted_history += f"User Query: {h.query}\n"
                formatted_history += f"AI Response: {h.response}\n\n"

        result = rag_engine.generate_response(
            query=request.query,
            context=formatted_history if history else None,
        )

        # Evaluate response
        evaluation_metrics = rag_engine.evaluate_response(
            query=result["expanded_query"],
            response=result["response"],
            retrieved_docs=result["retrieved_documents"],
        )

        # Create and store interaction
        interaction = Interaction(
            query=result["expanded_query"],
            response=result["response"],
            retrieved_documents=result["retrieved_documents"],
            evaluation_metrics=evaluation_metrics,
            conversation_id=conversation_id,
        )
        db.store_interaction(interaction)

        return {
            "expanded_query": result["expanded_query"],
            "response": result["response"],
            "retrieved_documents": result["retrieved_documents"],
            "evaluation_metrics": evaluation_metrics,
            "conversation_id": conversation_id,
            "model": result["model"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str, limit: int = 10):
    """Retrieve conversation history."""
    try:
        interactions = db.get_conversation_history(conversation_id, limit=limit)
        if not interactions:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return Conversation(
            conversation_id=conversation_id,
            interactions=interactions,
            created_at=interactions[-1].timestamp,
            updated_at=interactions[0].timestamp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "storage_type": settings.STORAGE_TYPE,
    }
