
from backend.core.vector_store import get_collection


def retrieve_guidelines(query: str, n_results: int = 3, category: str = None):
    """
    Retrieve relevant coding and security guidelines from ChromaDB.
    """

    if not query or not query.strip():
        return []

    collection = get_collection()

    total_documents = collection.count()

    if total_documents == 0:
        return []

    search_params = {
        "query_texts": [query],
        "n_results": min(n_results, total_documents)
    }

    if category:
        search_params["where"] = {"category": category}

    results = collection.query(**search_params)

    retrieved_documents = []

    for i in range(len(results["documents"][0])):
        retrieved_documents.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return retrieved_documents
