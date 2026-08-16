import streamlit as st
from pathlib import Path

from src.resume_parser import extract_resume_text


st.set_page_config(
    page_title="CareerChat AI",
    page_icon="💼",
    layout="wide"
)


st.title("💼 CareerChat AI")
st.subheader("Your AI Resume & Career Assistant")

st.write(
    "Upload your resume and let CareerChat AI analyze your "
    "skills, experience, projects, and career profile."
)


uploaded_file = st.file_uploader(
    "📄 Upload your Resume",
    type=["pdf", "docx"]
)


if uploaded_file is not None:

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    try:

        resume_text = extract_resume_text(str(file_path))

        st.success("Resume processed successfully!")

        st.subheader("📋 Extracted Resume Text")

        st.text_area(
            "Resume Content",
            resume_text,
            height=400
        )

    except Exception as e:

        st.error(f"Error processing resume: {e}")