
import pdfplumber

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file using pdfplumber
    """
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None 