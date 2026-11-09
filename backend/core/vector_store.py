from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Persistent ChromaDB storage
DB_PATH = BASE_DIR / "chroma_db"

# Collection name
COLLECTION_NAME = "autodev_knowledge"


def get_embedding_function():
    """Create the sentence transformer embedding function."""
    return SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        normalize_embeddings=True
    )


def get_collection():
    """Initialize ChromaDB and return the knowledge collection."""
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def add_documents(documents, metadatas, ids):
    """Add or update knowledge documents in ChromaDB."""
    if not (len(documents) == len(metadatas) == len(ids)):
        raise ValueError("Documents, metadatas and IDs must have equal lengths.")

    collection = get_collection()

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )


def seed_knowledge_base():
    """Add initial security and coding guidelines."""

    documents = [
        (
            "Validate all incoming API data. Use authentication and "
            "authorization checks. Never expose sensitive information "
            "in error messages. Apply rate limiting to protect APIs."
        ),
        (
            "Use meaningful variable and function names. Follow consistent "
            "indentation. Keep functions small and readable. Use clear "
            "comments and follow Python coding conventions."
        ),
        (
            "Use Pydantic models to validate request data in FastAPI. "
            "Use dependency injection for shared dependencies. Handle "
            "exceptions properly and never expose internal errors to users."
        )
    ]

    metadatas = [
        {
            "category": "security",
            "source": "OWASP API Security guidance",
            "source_url": "https://owasp.org/www-project-api-security/"
        },
        {
            "category": "python",
            "source": "Python PEP 8",
            "source_url": "https://peps.python.org/pep-0008/"
        },
        {
            "category": "fastapi",
            "source": "FastAPI documentation",
            "source_url": "https://fastapi.tiangolo.com/"
        }
    ]

    ids = [
        "owasp_api_001",
        "pep8_001",
        "fastapi_001"
    ]

    add_documents(documents, metadatas, ids)

    print("Knowledge base seeded successfully!")


if __name__ == "__main__":
    seed_knowledge_base()
