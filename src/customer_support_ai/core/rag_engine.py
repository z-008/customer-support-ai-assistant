"""
RAG Engine implementation for the Customer Support AI Assistant
"""

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import groq
from ..core.config import settings
import json
import os
import logging
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RAGEngine:
    """RAG Engine for customer support query processing"""

    def __init__(self):
        """Initialize the RAG engine with necessary components"""
        # Debug log to verify API key
        logger.info(
            f"Initializing RAG engine with API key present: {bool(settings.GROQ_API_KEY)}"
        )

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Updated ChromaDB client initialization
        self.client = chromadb.Client(
            Settings(
                persist_directory=settings.CHROMA_PERSIST_DIR,
                is_persistent=True,  # This ensures persistence
                anonymized_telemetry=False,
            )
        )

        self.collection = self.client.get_or_create_collection(
            name="customer_support", metadata={"hnsw:space": "cosine"}
        )
        self.groq_client = groq.Groq(api_key=settings.GROQ_API_KEY)

    def __del__(self):
        """Cleanup is handled automatically by ChromaDB"""
        pass

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector database."""
        print(f"Adding {len(documents)} documents to collection...")

        texts = [doc["text"] for doc in documents]
        print(
            f"Sample text: {texts[0][:100]}..."
        )  # Print first 100 chars of first document

        embeddings = self.encoder.encode(texts).tolist()
        print(f"Generated embeddings of shape: {len(embeddings)}x{len(embeddings[0])}")

        # Use unique IDs based on content hash or metadata ID
        ids = [f"doc_{doc['metadata']['id']}" for doc in documents]
        print(f"Generated {len(ids)} unique IDs")

        # Include all metadata for better retrieval
        metadatas = [
            {
                "source": doc.get("source", "unknown"),
                "query": doc["metadata"].get("query", ""),
                "response": doc["metadata"].get("response", ""),
                "id": doc["metadata"].get("id", ""),
            }
            for doc in documents
        ]

        try:
            before_count = self.collection.count()
            self.collection.add(
                embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
            )
            after_count = self.collection.count()
            print(f"Collection size before: {before_count}, after: {after_count}")
        except Exception as e:
            print(f"Error adding documents: {e}")

    def retrieve(
        self, query: str, n_results: int = settings.MAX_RETRIEVED_DOCS
    ) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query
            n_results: Number of results to retrieve

        Returns:
            List of retrieved document texts
        """
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],  # Include metadata in results
        )

        distances = results["distances"][0]
        # Add logging to check results
        print(f"Found {len(results['documents'][0])} documents")
        if len(results["documents"][0]) > 0:
            print(
                f"First document: {results['documents'][0][0][:100]}..."
            )  # Print first 100 chars

        return results["documents"][0], distances

    def generate_response(
        self, query: str, context: List[str] = None
    ) -> Dict[str, Any]:
        """Generate a response using RAG.

        Args:
            query: The customer's query
            context: Optional list of previous context

        Returns:
            Dictionary containing response and metadata
        """
        # Retrieve relevant documents
        retrieved_docs, distances = self.retrieve(query)

        # Prepare context
        context_str = "\n".join(retrieved_docs) if retrieved_docs else ""

        # Prepare prompt
        prompt = f"""You are a helpful customer support assistant. Use the following context to answer the customer's question. If the context is not relevant, provide a general helpful response.

Context:
{context_str}

Customer Question: {query}

Please provide a helpful response:"""

        # Generate response using Groq
        completion = self.groq_client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful customer support assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )

        response = completion.choices[0].message.content

        formatted_documents = [
            f"{doc}\nSimilarity Score: {1 - distance:.4f}"  # Convert distance to similarity
            for doc, distance in zip(retrieved_docs, distances)
        ]
        return {
            "response": response,
            "retrieved_documents": formatted_documents,
            "model": settings.MODEL_NAME,
        }

    def evaluate_response(
        self, query: str, response: str, retrieved_docs: List[str]
    ) -> Dict[str, float]:
        """Evaluate the quality of the response using multiple metrics.

        Args:
            query: The original query
            response: The generated response
            retrieved_docs: The retrieved documents

        Returns:
            Dictionary containing evaluation metrics
        """
        # 1. Basic word overlap metrics (existing)
        relevance_score = len(
            set(query.lower().split()) & set(response.lower().split())
        ) / len(query.split())
        context_usage = len(
            set(response.lower().split())
            & set(" ".join(retrieved_docs).lower().split())
        ) / len(response.split())

        # 2. BM25 scoring
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in retrieved_docs]
        bm25 = BM25Okapi(tokenized_docs)

        # Get BM25 score for response against retrieved docs
        response_tokens = response.lower().split()
        bm25_score = np.mean(bm25.get_scores(response_tokens))

        # 3. Semantic similarity using sentence embeddings
        query_embedding = self.encoder.encode([query])
        response_embedding = self.encoder.encode([response])
        semantic_similarity = cosine_similarity(query_embedding, response_embedding)[0][
            0
        ]

        return {
            "relevance_score": relevance_score,
            "context_usage": context_usage,
            "bm25_score": float(bm25_score),
            "semantic_similarity": float(semantic_similarity),
        }
