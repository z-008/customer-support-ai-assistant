"""
Database module for the Customer Support AI Assistant
"""

from .base import DatabaseInterface
from .models import Interaction, Conversation
from .sqlite_storage import SQLiteStorage
from .csv_storage import CSVStorage

__all__ = [
    "DatabaseInterface",
    "Interaction",
    "Conversation",
    "SQLiteStorage",
    "CSVStorage",
]
