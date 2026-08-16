from src.embeddings import create_embeddings
from src.vector_store import (
    create_vector_store,
    search_vector_store,
    save_vector_store,
    load_vector_store
)


chunks = [
    "I have experience in Python, Pandas, NumPy and Machine Learning.",
    "I have worked with Laravel, PHP, REST APIs and MySQL.",
    "I have experience with AWS, Docker and CI/CD."
]


print("===== CREATE EMBEDDINGS =====")

embeddings = create_embeddings(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


print("\n===== CREATE VECTOR STORE =====")

index = create_vector_store(embeddings)

print("FAISS index created successfully")
print("Number of vectors:", index.ntotal)


print("\n===== SAVE VECTOR STORE =====")

save_vector_store(
    index,
    chunks,
    directory="vectorstore"
)

print("Vector store saved successfully")


print("\n===== LOAD VECTOR STORE =====")

loaded_index, loaded_chunks = load_vector_store(
    directory="vectorstore"
)

print("Vector store loaded successfully")
print("Loaded vectors:", loaded_index.ntotal)
print("Loaded chunks:", len(loaded_chunks))


print("\n===== SEMANTIC SEARCH =====")

query = "What programming and machine learning skills do I have?"

query_embedding = create_embeddings(
    [query]
)[0]

scores, indices = search_vector_store(
    loaded_index,
    query_embedding,
    top_k=5
)


for score, index_number in zip(scores, indices):

    print("\nSimilarity score:", score)

    print(
        "Chunk:",
        loaded_chunks[index_number]
    )