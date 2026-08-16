import re


def clean_text(text: str) -> str:
    """
    Clean extracted resume text.
    """

    if not text:
        return ""

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove spaces at beginning/end of lines
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200
) -> list[str]:
    """
    Split resume text into overlapping chunks while
    trying to preserve paragraph and section boundaries.
    """

    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    # Split using blank lines first
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # If adding paragraph keeps us within the limit
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:

            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:

            if current_chunk:
                chunks.append(current_chunk.strip())

            # If paragraph itself is too large
            if len(paragraph) > chunk_size:

                start = 0

                while start < len(paragraph):

                    end = start + chunk_size

                    piece = paragraph[start:end].strip()

                    if piece:
                        chunks.append(piece)

                    start += chunk_size - chunk_overlap

                current_chunk = ""

            else:
                current_chunk = paragraph

    # Add remaining content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks