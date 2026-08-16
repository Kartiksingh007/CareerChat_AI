from src.resume_parser import extract_resume_text
from src.text_processor import clean_text, chunk_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.retriever import retrieve_relevant_chunks
from src.vector_store import save_vector_store, load_vector_store

# Resume path
file_path = r"C:\Users\asus\Downloads\Nikhil_CV.pdf"


# 1. Extract resume text
resume_text = extract_resume_text(file_path)

print("Resume extracted successfully")


# 2. Clean text
cleaned_text = clean_text(resume_text)

print("Text cleaned successfully")


# 3. Create chunks
chunks = chunk_text(cleaned_text)

print("Number of chunks:", len(chunks))


# 4. Create embeddings
embeddings = create_embeddings(chunks)

print("Embedding shape:", embeddings.shape)


# 5. Create FAISS vector store
index = create_vector_store(embeddings)


# 6. Save vector store
index_path, chunks_path = save_vector_store(
    index,
    chunks
)

print("Vector store saved successfully")
print("Index:", index_path)
print("Chunks:", chunks_path)

# 7. Load vector store
loaded_index, loaded_chunks = load_vector_store()

print("Vector store loaded successfully")
print("Loaded vectors:", loaded_index.ntotal)
print("Loaded chunks:", len(loaded_chunks))

# 8. Create query embedding
query = "Tell me about my work experience."

results = retrieve_relevant_chunks(
    query=query,
    index=index,
    chunks=chunks,
    top_k=5,
    score_threshold=0.20
)
print("\n===== RETRIEVAL RESULTS =====")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("Score:", result["score"])
    print("Chunk:")
    print(result["chunk"])