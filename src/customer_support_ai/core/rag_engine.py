"""
RAG Engine implementation for the Customer Support AI Assistant
"""

from typing import List, Dict, Any, Optional
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
from functools import lru_cache
from datetime import datetime, timedelta
from cachetools import TTLCache
from tqdm import tqdm
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create and configure stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO level to see the log messages
logger.addHandler(stream_handler)


class RAGEngine:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the RAG engine with necessary components"""

        print("Initializing RAG Engine...")
        # Debug log to verify API key
        logger.info(
            f"Initializing RAG engine with API key present: {bool(settings.GROQ_API_KEY)}"
        )

        # Create progress bar for overall initialization
        steps = ["API Check", "Model Loading", "ChromaDB Setup", "Cache Setup"]
        with tqdm(total=len(steps), desc="Overall Progress") as pbar:
            # API Key Check
            logger.info("Checking API key...")
            time.sleep(0.5)  # Add small delay to make progress visible
            pbar.set_description("Checking API key")
            logger.info(
                f"Initializing RAG engine with API key present: {bool(settings.GROQ_API_KEY)}"
            )
            pbar.update(1)

            # Model Loading
            pbar.set_description("Loading ML model")
            logger.info("Loading sentence transformer model...")
            with tqdm(total=100, desc="Loading Model", leave=False) as model_pbar:
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                model_pbar.update(100)
            logger.info("Sentence transformer model loaded")
            pbar.update(1)

            # ChromaDB Setup
            pbar.set_description("Setting up ChromaDB")
            logger.info("Initializing ChromaDB...")
            with tqdm(total=100, desc="ChromaDB Setup", leave=False) as db_pbar:
                self.client = chromadb.Client(
                    Settings(
                        persist_directory=settings.CHROMA_PERSIST_DIR,
                        is_persistent=True,  # This ensures persistence
                        anonymized_telemetry=False,
                    )
                )
                db_pbar.update(50)

                self.collection = self.client.get_or_create_collection(
                    name="customer_support", metadata={"hnsw:space": "cosine"}
                )
                db_pbar.update(50)
            logger.info("ChromaDB initialized")
            pbar.update(1)

            # Cache Setup
            pbar.set_description("Setting up cache")
            self.groq_client = groq.Groq(api_key=settings.GROQ_API_KEY)
            self.response_cache = None

            if settings.ENABLE_RESPONSE_CACHE:
                self.response_cache = TTLCache(
                    maxsize=settings.CACHE_MAX_SIZE,
                    ttl=timedelta(hours=settings.CACHE_TTL_HOURS).total_seconds(),
                )
                logger.info(
                    "Response cache enabled with TTL: %d hours, max size: %d",
                    settings.CACHE_TTL_HOURS,
                    settings.CACHE_MAX_SIZE,
                )
            else:
                logger.info("Response cache disabled")
            pbar.update(1)

        print("\nRAG Engine initialization complete! ✨")

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

    def _expand_query(self, query: str, context: str = None) -> str:
        """Expand the query using conversation context.

        Args:
            query: Original user query
            context: Optional list of previous responses

        Returns:
            str: Expanded query incorporating context
        """
        if not context:
            return query

        prompt = f"""Given the following conversation context and current query, 
        create a simplified, concise query that captures the core question. Focus on maintaining the original intent.
        Remove any unnecessary preamble, context, or verbose explanations while maintaining the essential question.
        Resolve any coreferences (pronouns, demonstratives, etc.) by replacing them with their specific referents.

        Previous Context:
        {' '.join(context)}

        Current Query: {query}

        Generate a clear, concise query. Strip away any unnecessary preamble. Do not include any other text than the query."""

        completion = self.groq_client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant tasked with expanding queries.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )

        expanded_query = completion.choices[0].message.content

        logger.info("Expanded query: %s", expanded_query)
        return expanded_query

    def generate_response(self, query: str, context: str = None) -> Dict[str, Any]:
        """Generate a response using RAG with optional caching.

        Args:
            query: The customer's query
            context: Optional list of previous context

        Returns:
            Dictionary containing response and metadata
        """

        # Expand query using context if available
        expanded_query = self._expand_query(query, context)

        # Check cache if enabled
        if self.response_cache is not None:
            cache_key = expanded_query.lower().strip()
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                logger.info("Cache hit for query: %s", cache_key)
                return cached_response

        # Retrieve relevant documents using expanded query
        retrieved_docs, distances = self.retrieve(expanded_query)

        # Use threshold from settings
        relevant_docs = []
        for doc, distance in zip(retrieved_docs, distances):
            if (1 - distance) >= settings.RELEVANCE_THRESHOLD:
                relevant_docs.append(doc)

        # Prepare context - only use documents that pass the threshold
        context_str = "\n".join(relevant_docs) if relevant_docs else ""

        # Prepare prompt
        prompt = f"""You are a helpful customer support assistant. First determine if the provided context is relevant to answering the customer's question.

        Context:
        {context_str}

        Customer Question: {expanded_query}

        If the context is relevant:
        - Provide a specific response based on the context directly
        If the context is NOT relevant:
        - Provide a general helpful response directly

        Important: Do not include any preamble about context relevance or your reasoning process. Start your response immediately with the answer."""

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
            if (1 - distance) >= settings.RELEVANCE_THRESHOLD
        ]

        result = {
            "expanded_query": expanded_query,
            "response": response,
            "retrieved_documents": formatted_documents,
            "model": settings.MODEL_NAME,
        }

        # Cache the response if enabled
        if self.response_cache is not None:
            self.response_cache[cache_key] = result
            logger.info("Cached response for query: %s", cache_key)

        return result

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

        # Metrics that depend on retrieved_docs
        if retrieved_docs:
            context_usage = len(
                set(response.lower().split())
                & set(" ".join(retrieved_docs).lower().split())
            ) / len(response.split())

            # BM25 scoring
            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in retrieved_docs]
            bm25 = BM25Okapi(tokenized_docs)
            # Get BM25 score for response against retrieved docs
            response_tokens = response.lower().split()
            bm25_score = float(np.mean(bm25.get_scores(response_tokens)))
        else:
            context_usage = 0.0
            bm25_score = 0.0

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
