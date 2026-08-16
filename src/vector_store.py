import faiss
import numpy as np
import json
import os


def create_vector_store(embeddings):
    """
    Create a FAISS vector store using cosine similarity.
    """

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings cannot be empty.")

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Make sure embeddings are 2D
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D. Got shape: {embeddings.shape}"
        )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Inner Product on normalized vectors
    # = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_vector_store(
    index,
    query_embedding,
    top_k=3
):
    """
    Search the FAISS vector store.
    """

    if index is None:
        raise ValueError(
            "FAISS index cannot be None."
        )

    if index.ntotal == 0:
        return [], []

    # Convert query embedding to numpy
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # ------------------------------------------
    # Make sure query is 2D
    # ------------------------------------------

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    elif query_embedding.ndim != 2:
        raise ValueError(
            f"Query embedding must be 1D or 2D. "
            f"Got shape: {query_embedding.shape}"
        )

    # Normalize query for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Make sure top_k is valid
    top_k = min(
        int(top_k),
        index.ntotal
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    # Return first query's results
    return scores[0], indices[0]


def save_vector_store(
    index,
    chunks,
    directory="vectorstore"
):
    """
    Save FAISS index and corresponding chunks to disk.
    """

    os.makedirs(
        directory,
        exist_ok=True
    )

    index_path = os.path.join(
        directory,
        "resume.index"
    )

    chunks_path = os.path.join(
        directory,
        "chunks.json"
    )

    # Save FAISS index
    faiss.write_index(
        index,
        index_path
    )

    # Save chunks
    with open(
        chunks_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    return index_path, chunks_path


def load_vector_store(
    directory="vectorstore"
):
    """
    Load FAISS index and chunks from disk.
    """

    index_path = os.path.join(
        directory,
        "resume.index"
    )

    chunks_path = os.path.join(
        directory,
        "chunks.json"
    )

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    # Load FAISS index
    index = faiss.read_index(
        index_path
    )

    # Load chunks
    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return index, chunks