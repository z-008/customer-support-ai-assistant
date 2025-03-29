"""
Base database interface for the Customer Support AI Assistant
"""

from abc import ABC, abstractmethod
from typing import List
from .models import Interaction, Conversation


class DatabaseInterface(ABC):
    """Abstract base class for database implementations"""

    @abstractmethod
    def store_interaction(self, interaction: Interaction) -> None:
        """Store a new interaction"""
        pass

    @abstractmethod
    def get_conversation_history(
        self, conversation_id: str, limit: int = 5
    ) -> List[Interaction]:
        """Retrieve conversation history"""
        pass

    @abstractmethod
    def get_all_interactions(self, limit: int = 100) -> List[Interaction]:
        """Retrieve all interactions"""
        pass
