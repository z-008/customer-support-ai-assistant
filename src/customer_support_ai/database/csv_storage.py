"""
CSV-based storage implementation for the Customer Support AI Assistant
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .base import DatabaseInterface
from .models import Interaction, Conversation


class CSVStorage(DatabaseInterface):
    """CSV-based storage implementation"""

    def __init__(self, csv_path: str = "interactions.csv"):
        """Initialize CSV storage.

        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = Path(csv_path)
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize the CSV storage file if it doesn't exist"""
        if not self.csv_path.exists():
            # Create CSV with headers
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "conversation_id",
                        "query",
                        "response",
                        "retrieved_documents",
                        "evaluation_metrics",
                    ]
                )

    def store_interaction(self, interaction: Interaction) -> None:
        """Store a new interaction in CSV format.

        Args:
            interaction: The interaction to store
        """
        # Convert interaction to CSV row
        row = {
            "timestamp": interaction.timestamp or datetime.now().isoformat(),
            "conversation_id": interaction.conversation_id,
            "query": interaction.query,
            "response": interaction.response,
            "retrieved_documents": json.dumps(interaction.retrieved_documents),
            "evaluation_metrics": json.dumps(interaction.evaluation_metrics),
        }

        # Append to CSV
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode="a", header=False, index=False)

    def get_conversation_history(
        self, conversation_id: str, limit: int = 5
    ) -> List[Interaction]:
        """Retrieve conversation history from CSV.

        Args:
            conversation_id: The ID of the conversation
            limit: Maximum number of interactions to retrieve

        Returns:
            List of interactions in the conversation
        """
        # Read CSV and filter by conversation_id
        df = pd.read_csv(self.csv_path)
        df = df[df["conversation_id"] == conversation_id]

        # Sort by timestamp and limit
        df = df.sort_values("timestamp", ascending=False).head(limit)

        # Convert to Interaction objects
        interactions = []
        for _, row in df.iterrows():
            interaction = Interaction(
                query=row["query"],
                response=row["response"],
                retrieved_documents=json.loads(row["retrieved_documents"]),
                evaluation_metrics=json.loads(row["evaluation_metrics"]),
                conversation_id=row["conversation_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            interactions.append(interaction)

        return interactions

    def get_all_interactions(self, limit: int = 100) -> List[Interaction]:
        """Retrieve all interactions from CSV.

        Args:
            limit: Maximum number of interactions to retrieve

        Returns:
            List of interactions
        """
        # Read CSV and limit rows
        df = pd.read_csv(self.csv_path)
        df = df.sort_values("timestamp", ascending=False).head(limit)

        # Convert to Interaction objects
        interactions = []
        for _, row in df.iterrows():
            interaction = Interaction(
                query=row["query"],
                response=row["response"],
                retrieved_documents=json.loads(row["retrieved_documents"]),
                evaluation_metrics=json.loads(row["evaluation_metrics"]),
                conversation_id=row["conversation_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            interactions.append(interaction)

        return interactions
