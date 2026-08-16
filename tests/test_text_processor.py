from src.resume_parser import extract_resume_text
from src.text_processor import clean_text, chunk_text


file_path = r"C:\Users\asus\Downloads\Nikhil_CV.pdf"


# Step 1: Extract
resume_text = extract_resume_text(file_path)

print("\n===== ORIGINAL TEXT =====\n")
print(resume_text)


# Step 2: Clean
cleaned_text = clean_text(resume_text)

print("\n===== CLEANED TEXT =====\n")
print(cleaned_text)


# Step 3: Chunk
chunks = chunk_text(cleaned_text)

print("\n===== CHUNKS =====")
print(f"Total chunks: {len(chunks)}")


for i, chunk in enumerate(chunks, start=1):

    print(f"\n===== CHUNK {i} =====\n")
    print(chunk)