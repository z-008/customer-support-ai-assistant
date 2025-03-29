import pytest
from unittest.mock import Mock, patch
from customer_support_ai.core.rag_engine import RAGEngine


@pytest.fixture
def mock_encoder():
    with patch("sentence_transformers.SentenceTransformer") as mock:
        # Mock encode method to return a simple vector
        mock.return_value.encode.return_value = [[0.1, 0.2, 0.3]]
        yield mock


@pytest.fixture
def mock_chromadb():
    with patch("chromadb.Client") as mock:
        # Mock collection methods
        collection = Mock()
        collection.count.return_value = 0
        collection.add.return_value = None
        collection.query.return_value = {
            "documents": [["Sample response"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.5]],
        }
        mock.return_value.get_or_create_collection.return_value = collection
        yield mock


@pytest.fixture
def rag_engine():
    # Mock all dependencies at once using multiple patches
    with patch("sentence_transformers.SentenceTransformer") as mock_transformer, patch(
        "chromadb.Client"
    ) as mock_chroma, patch("groq.Groq") as mock_groq:

        # Setup transformer mock
        mock_transformer.return_value.encode.return_value = [[0.1, 0.2, 0.3]]

        # Setup ChromaDB mock
        collection = Mock()
        collection.count.return_value = 0
        collection.query.return_value = {
            "documents": [["Sample response"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.5]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = collection

        # Setup Groq mock
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="Test response"))]
        mock_groq.return_value.chat.completions.create.return_value = mock_completion

        # Create RAG engine with all mocks in place
        engine = RAGEngine()
        yield engine


def test_add_documents(rag_engine):
    """Test adding documents to the vector database"""
    documents = [
        {
            "text": "Test query and response",
            "source": "test",
            "metadata": {"query": "test?", "response": "test response", "id": "1"},
        }
    ]

    rag_engine.add_documents(documents)

    # Verify document was added
    assert rag_engine.collection.add.called
    call_args = rag_engine.collection.add.call_args[1]
    assert len(call_args["documents"]) == 1
    assert call_args["documents"][0] == "Test query and response"


def test_retrieve_documents(rag_engine):
    """Test retrieving documents for a query"""
    query = "test query"
    results = rag_engine.retrieve(query)

    # Verify retrieval
    assert len(results) == 1
    assert results[0] == "Sample response"


def test_generate_response(rag_engine):
    """Test response generation"""
    result = rag_engine.generate_response("test query")
    assert result["response"] == "Test response"
