from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import black
import os
import sys

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def text_to_pdf(input_text_file, output_pdf_file):
    """
    Convert a text file to a formatted PDF
    """
    print(f"Input text file: {os.path.abspath(input_text_file)}")
    print(f"Output PDF file: {os.path.abspath(output_pdf_file)}")
    
    # Check if input file exists
    if not os.path.exists(input_text_file):
        print(f"Error: Input file {input_text_file} does not exist.")
        return
        
    # Read the text file
    with open(input_text_file, 'r') as file:
        content = file.read()
    
    # Split the content into lines
    lines = content.splitlines()
    
    # Create PDF document
    doc = SimpleDocTemplate(output_pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    styles.add(ParagraphStyle(name='Name', 
                             parent=styles['Heading1'], 
                             fontSize=16,
                             alignment=TA_CENTER))
    
    styles.add(ParagraphStyle(name='Contact', 
                             parent=styles['Normal'], 
                             fontSize=10,
                             alignment=TA_CENTER))
    
    styles.add(ParagraphStyle(name='Heading', 
                             parent=styles['Heading2'], 
                             fontSize=12,
                             textColor=black,
                             spaceAfter=6))
    
    styles.add(ParagraphStyle(name='NormalText', 
                             parent=styles['Normal'],
                             fontSize=10,
                             spaceAfter=6))
    
    # Build the PDF content
    elements = []
    
    # Process lines
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # First line is the name
        if i == 0:
            elements.append(Paragraph(line, styles['Name']))
            i += 1
            continue
        
        # Second line is contact info
        if i == 1:
            elements.append(Paragraph(line, styles['Contact']))
            elements.append(Spacer(1, 12))
            i += 1
            continue
        
        # Headings (all caps lines)
        if line.isupper() and len(line) > 3:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(line, styles['Heading']))
            i += 1
            continue
        
        # Everything else as normal text
        elements.append(Paragraph(line, styles['NormalText']))
        i += 1
    
    # Build the PDF
    doc.build(elements)
    
    # Verify the file was created
    if os.path.exists(output_pdf_file):
        file_size = os.path.getsize(output_pdf_file)
        print(f"PDF created: {output_pdf_file} (Size: {file_size} bytes)")
    else:
        print(f"Error: PDF was not created at {output_pdf_file}")

if __name__ == "__main__":
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_root, "src", "data", "sample_resume.txt")
    output_file = os.path.join(project_root, "src", "data", "sample_resume.pdf")
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Print current directory for debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    
    text_to_pdf(input_file, output_file) 