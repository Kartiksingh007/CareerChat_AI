from src.embeddings import create_embeddings
from src.vector_store import search_vector_store


def retrieve_relevant_chunks(
    index,
    chunks,
    query,
    top_k=5
):
    """
    Retrieve the most relevant resume chunks for a user query.

    Parameters:
        index: FAISS vector index
        chunks: List of resume text chunks
        query: User's question
        top_k: Number of chunks to retrieve

    Returns:
        List of dictionaries containing similarity score and chunk text.
    """

    # ------------------------------------------
    # Validate input
    # ------------------------------------------

    if not query or not query.strip():
        return []

    if index is None:
        return []

    if not chunks:
        return []

    # ------------------------------------------
    # Create embedding for user query
    # ------------------------------------------

    query_embeddings = create_embeddings([query])

    if query_embeddings is None or len(query_embeddings) == 0:
        return []

    # IMPORTANT:
    # create_embeddings([query]) returns:
    # [embedding]
    #
    # We need only the first embedding.
    query_embedding = query_embeddings[0]

    # ------------------------------------------
    # Search FAISS vector store
    # ------------------------------------------

    distances, indices = search_vector_store(
        index,
        query_embedding,
        top_k
    )

    # ------------------------------------------
    # Build results
    # ------------------------------------------

    results = []

    for score, index_id in zip(
        distances,
        indices
    ):

        index_id = int(index_id)

        # Safety check
        if index_id < 0 or index_id >= len(chunks):
            continue

        results.append(
            {
                "score": float(score),
                "chunk": chunks[index_id]
            }
        )

    return results