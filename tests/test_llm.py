from src.llm import generate_answer


context = """
I have experience in Python, Pandas, NumPy and Machine Learning.

I have experience with Random Forest, XGBoost, KNN, SVM and K-Means.

I have worked with Laravel, PHP, REST APIs and MySQL.
"""

question = "What machine learning skills do I have?"

answer = generate_answer(
    context,
    question
)

print("\n===== LLM ANSWER =====")
print(answer)