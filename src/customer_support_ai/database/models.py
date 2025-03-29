"""
Database models for the Customer Support AI Assistant
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class Interaction(BaseModel):
    """Model for a customer support interaction"""

    query: str
    response: str
    retrieved_documents: List[str]
    evaluation_metrics: Dict[str, float]
    conversation_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class Conversation(BaseModel):
    """Model for a customer support conversation"""

    conversation_id: str
    interactions: List[Interaction]
    created_at: datetime
    updated_at: datetime
