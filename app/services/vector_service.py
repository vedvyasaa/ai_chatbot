from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def property_to_text(property_item):

    return f"""
    {property_item.title}
    {property_item.location}
    {property_item.description}
     """


def create_embeddings(properties):

    texts = [
        property_to_text(p)
        for p in properties
    ]

    embeddings = model.encode(texts)

    return np.array(embeddings)


def build_faiss_index(embeddings):
    # Ensure embeddings is a 2D numpy array
    if embeddings is None or getattr(embeddings, 'size', 0) == 0:
        raise ValueError("Embeddings must be a non-empty numpy array")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def semantic_search(query, properties, index):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        k=3
    )

    result = []

    if indices is None or len(indices) == 0:
        return []

    for idx in indices[0]:
        if idx is None or idx < 0 or idx >= len(properties):
            continue

        result.append(properties[idx])

    return result
