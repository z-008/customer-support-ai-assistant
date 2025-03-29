from .sqlite_storage import SQLiteStorage
from .models import Interaction


def init_and_test_db():
    """Initialize and test the SQLite database"""
    # Initialize the database
    storage = SQLiteStorage()

    # Test storing an interaction
    test_interaction = Interaction(
        query="Test question?",
        response="Test response",
        retrieved_documents=["doc1", "doc2"],
        evaluation_metrics={"accuracy": 0.95},
        conversation_id="test-conversation-1",
    )

    try:
        # Store the test interaction
        storage.store_interaction(test_interaction)
        print("✅ Successfully stored test interaction")

        # Retrieve the conversation history
        history = storage.get_conversation_history("test-conversation-1")
        print(
            f"✅ Successfully retrieved conversation history: {len(history)} interactions"
        )

        print("\nDatabase initialization and testing completed successfully!")

    except Exception as e:
        print(f"❌ Error during database testing: {str(e)}")


if __name__ == "__main__":
    init_and_test_db()
