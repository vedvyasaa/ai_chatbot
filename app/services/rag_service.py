from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        text += page.extract_text()

    return text


def chunk_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        chunks.append(chunk)

    return chunks


def create_chunk_embeddings(chunks):

    embeddings = model.encode(chunks)

    return np.array(embeddings)


def build_chunk_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_similar_chunks(
        query,
        chunks,
        index,
        top_k=3
):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(chunks[idx])

        return results
