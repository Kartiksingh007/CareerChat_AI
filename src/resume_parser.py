import pymupdf
from docx import Document
from pathlib import Path


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF resume."""

    text = ""

    document = pymupdf.open(file_path)

    for page in document:
        text += page.get_text() + "\n"

    document.close()

    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX resume."""

    document = Document(file_path)

    text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)

    return "\n".join(text).strip()


def extract_resume_text(file_path: str) -> str:
    """Extract resume text based on the file extension."""

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    elif extension == ".docx":
        return extract_text_from_docx(file_path)

    else:
        raise ValueError(
            "Unsupported file format. Please upload a PDF or DOCX resume."
        )