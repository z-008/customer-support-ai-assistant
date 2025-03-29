"""
SQLite-based storage implementation for the Customer Support AI Assistant
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Any
import json

from .base import DatabaseInterface
from .models import Interaction, Conversation
from ..core.config import settings


class SQLiteStorage(DatabaseInterface):
    """SQLite-based storage implementation"""

    def __init__(self, db_path: str = settings.DATABASE_URL):
        """Initialize the database connection"""
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create interactions table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            retrieved_documents TEXT,
            evaluation_metrics TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            conversation_id TEXT
        )
        """
        )

        conn.commit()
        conn.close()

    def store_interaction(self, interaction: Interaction) -> None:
        """Store a new interaction in the database.

        Args:
            interaction: The interaction to store
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO interactions 
        (query, response, retrieved_documents, evaluation_metrics, conversation_id)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                interaction.query,
                interaction.response,
                json.dumps(interaction.retrieved_documents),
                json.dumps(interaction.evaluation_metrics),
                interaction.conversation_id,
            ),
        )

        conn.commit()
        conn.close()

    def get_conversation_history(
        self, conversation_id: str, limit: int = 5
    ) -> List[Interaction]:
        """Retrieve conversation history for a given conversation ID.

        Args:
            conversation_id: The ID of the conversation
            limit: Maximum number of interactions to retrieve

        Returns:
            List of interactions in the conversation
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT query, response, retrieved_documents, evaluation_metrics, timestamp
        FROM interactions
        WHERE conversation_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
            (conversation_id, limit),
        )

        results = cursor.fetchall()
        conn.close()

        return [
            Interaction(
                query=row[0],
                response=row[1],
                retrieved_documents=json.loads(row[2]),
                evaluation_metrics=json.loads(row[3]),
                timestamp=row[4],
                conversation_id=conversation_id,
            )
            for row in results
        ]

    def get_all_interactions(self, limit: int = 100) -> List[Interaction]:
        """Retrieve all interactions with a limit.

        Args:
            limit: Maximum number of interactions to retrieve

        Returns:
            List of interactions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT query, response, retrieved_documents, evaluation_metrics, timestamp, conversation_id
        FROM interactions
        ORDER BY timestamp DESC
        LIMIT ?
        """,
            (limit,),
        )

        results = cursor.fetchall()
        conn.close()

        return [
            Interaction(
                query=row[0],
                response=row[1],
                retrieved_documents=json.loads(row[2]),
                evaluation_metrics=json.loads(row[3]),
                timestamp=row[4],
                conversation_id=row[5],
            )
            for row in results
        ]
