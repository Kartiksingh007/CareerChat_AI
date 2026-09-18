# CareerChat AI

### AI-Powered Resume Assistant using RAG

CareerChat AI is an AI-powered resume assistant that allows users to upload their resume in PDF or DOCX format and ask questions about their skills, experience, education, projects, and career background.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded resume and generate grounded answers using a Groq-powered LLM.

---

## 🚀 Features

- 📄 Upload PDF and DOCX resumes
- 🔍 Extract and clean resume text
- ✂️ Split resume into searchable chunks
- 🧠 Generate semantic embeddings
- 📦 Store embeddings using FAISS
- 🔎 Retrieve relevant resume information
- 🤖 Generate answers using Groq LLM
- 💬 Interactive resume chat interface
- 💡 Suggested questions for quick interaction
- 🛡️ Answers grounded in uploaded resume content

---

## 🏗️ Project Architecture

```text
User
  │
  ▼
Resume Upload
  │
  ▼
Resume Parser
  │
  ▼
Text Cleaning
  │
  ▼
Text Chunking
  │
  ▼
Embeddings
  │
  ▼
FAISS Vector Store
  │
  ▼
Retriever
  │
  ▼
Relevant Resume Context
  │
  ▼
Groq LLM
  │
  ▼
Grounded Answer
  │
  ▼
CareerChat AI Interface