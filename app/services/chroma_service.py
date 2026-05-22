import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="property_documents"
)


def store_document_chunks(chunks, embeddings, filename):

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": filename
        }
        for _ in chunks
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids
    )


def search_document(query_embedding):

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )
    # Results from chromadb can have None or empty lists for keys.
    # Normalize to always return a dict with list values so callers
    # can safely iterate or index after checking length.
    documents = results.get("documents")
    metadatas = results.get("metadatas")

    if not documents or not metadatas:
        return {"documents": [], "metadatas": []}

    # chromadb returns lists-of-lists for batch queries; grab the first
    # inner list but guard against unexpected shapes.
    docs_first = documents[0] if isinstance(documents, (list, tuple)) and len(documents) > 0 else []
    metas_first = metadatas[0] if isinstance(metadatas, (list, tuple)) and len(metadatas) > 0 else []

    return {
        "documents": docs_first or [],
        "metadatas": metas_first or []
    }
