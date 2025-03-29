"""
Dataset loading utility for the Customer Support AI Assistant
"""

from datasets import load_dataset
from ..core.rag_engine import RAGEngine
from tqdm import tqdm
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_customer_support_data(
    batch_size: int = 1000, max_docs: int = 10000, reset: bool = False
) -> None:
    """Load and process the customer support dataset.

    Args:
        batch_size: Number of documents to process in each batch
        max_docs: Maximum number of documents to load (default: 10000)
        reset: Whether to reset the collection before loading
    """
    logger.info("Starting dataset loading process...")
    logger.info("Attempting to download dataset from Hugging Face...")

    try:
        dataset = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")
        logger.info("Dataset successfully downloaded!")

        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag = RAGEngine()

        if reset:
            # Delete existing collection if it exists
            try:
                rag.client.delete_collection("customer_support")
                logger.info("Deleted existing collection")
            except:
                pass
            # Create new collection
            rag.collection = rag.client.create_collection(
                name="customer_support", metadata={"hnsw:space": "cosine"}
            )

        # Limit the total number of documents
        total_docs = min(len(dataset["train"]), max_docs)
        logger.info(
            f"Processing {total_docs} documents (limited from {len(dataset['train'])} total)"
        )

        for i in tqdm(range(0, total_docs, batch_size)):
            logger.debug(f"Processing batch starting at index {i}")
            batch = dataset["train"][i : min(i + batch_size, total_docs)]

            # Prepare documents
            documents: List[Dict[str, Any]] = []
            for idx, (input_text, output_text) in enumerate(
                zip(batch["input"], batch["output"])
            ):
                documents.append(
                    {
                        "text": f"Query: {input_text}\nResponse: {output_text}",
                        "source": "customer_support_tweets",
                        "metadata": {
                            "query": input_text,
                            "response": output_text,
                            "id": str(i + idx),
                        },
                    }
                )

            # Add documents to vector database
            rag.add_documents(documents)

            # Save progress
            if i % 1000 == 0:  # More frequent updates for smaller dataset
                logger.info(f"Progress: Processed {i}/{total_docs} documents")

        logger.info("Dataset loading completed successfully!")

    except Exception as e:
        logger.error(f"Error during dataset loading: {str(e)}")
        raise


if __name__ == "__main__":
    # Load only 10k documents with batch size of 1000
    load_customer_support_data(batch_size=1000, max_docs=10000, reset=True)
