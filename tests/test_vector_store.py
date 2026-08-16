from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, search_vector_store


chunks = [
    "I have experience in Python, Pandas, NumPy and Machine Learning.",
    "I have worked with Laravel, PHP, REST APIs and MySQL.",
    "I have experience with AWS, Docker and CI/CD."
]


# Create embeddings
embeddings = create_embeddings(chunks)

print("===== VECTOR STORE TEST =====")
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


# Create FAISS index
index = create_vector_store(embeddings)

print("FAISS index created successfully")
print("Number of vectors:", index.ntotal)


# Test semantic search
query = "What programming and machine learning skills do I have?"

query_embedding = create_embeddings([query])[0]

distances, indices = search_vector_store(
    index,
    query_embedding,
    top_k=2
)


print("\n===== SEARCH RESULTS =====")

for distance, index_number in zip(distances, indices):
    print("\nDistance:", distance)
    print("Chunk:", chunks[index_number])