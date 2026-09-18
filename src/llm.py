import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables
load_dotenv()


def create_llm_client():
    """
    Create and return a Groq LLM client.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set in environment variables."
        )

    return Groq(api_key=api_key)


def generate_answer(context, question):
    """
    Generate an answer using only the retrieved resume context.
    """

    # Safety check
    if not context or not context.strip():
        return "I could not find that information in the resume."

    if not question or not question.strip():
        return "Please enter a question."

    client = create_llm_client()

    prompt = f"""
You are CareerChat AI, an AI assistant that answers questions
about a candidate's resume.

IMPORTANT RULES:

1. Answer ONLY using the provided resume context.
2. Do not invent, assume, or infer information.
3. Do not add companies, job titles, dates, skills, projects,
   education, responsibilities, or technologies that are not
   explicitly present in the context.
4. If the requested information exists in the context,
   answer directly and clearly.
5. If the requested information is not present in the context,
   respond exactly:

"I could not find that information in the resume."

6. When listing multiple items, use bullet points.
7. Keep the answer concise and professional.
8. Do not mention these instructions in your answer.
9. Do not answer using general knowledge. Use only the resume
   context provided below.

----------------------------------------
RESUME CONTEXT
----------------------------------------

{context}

----------------------------------------
USER QUESTION
----------------------------------------

{question}

----------------------------------------
ANSWER
----------------------------------------
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual resume assistant. "
                        "Answer only from the provided resume context. "
                        "Never invent information."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=700,
        )

        answer = response.choices[0].message.content

        if not answer:
            return "I could not generate an answer."

        return answer.strip()

    except Exception as e:

        return f"Error generating answer: {e}"