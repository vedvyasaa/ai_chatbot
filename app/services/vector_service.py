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

    for idx in indices[0]:
        result.append(properties[idx])

    return result
