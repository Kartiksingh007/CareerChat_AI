from src.embeddings import create_embeddings


chunks = [
    "Python, Pandas and NumPy are used for data analysis.",
    "Machine learning models include Random Forest and XGBoost.",
    "Experienced in backend development using Laravel and REST APIs."
]


embeddings = create_embeddings(chunks)


print("===== EMBEDDINGS TEST =====")
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("First embedding:")
print(embeddings[0])