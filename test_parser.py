from src.resume_parser import extract_resume_text


file_path = r"C:\Users\asus\Downloads\Nikhil_CV.pdf"

text = extract_resume_text(file_path)

print("\n===== EXTRACTED RESUME TEXT =====\n")
print(text)