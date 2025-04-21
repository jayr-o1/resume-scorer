#!/usr/bin/env python
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def check_file_exists(file_path):
    """Check if the file exists."""
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return False
    return True

def create_pdf_from_text(input_file, output_file):
    """Convert a text file to a formatted PDF using ReportLab."""
    if not check_file_exists(input_file):
        return False

    try:
        # Read content from the input file
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # Create the PDF document
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        styles = getSampleStyleSheet()

        # Create custom styles
        styles.add(ParagraphStyle(
            name='Name',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=6,
        ))
        
        styles.add(ParagraphStyle(
            name='Contact',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.darkblue,
            spaceAfter=12,
        ))
        
        styles.add(ParagraphStyle(
            name='Heading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.darkblue,
            spaceAfter=6,
            spaceBefore=12,
        ))
        
        styles.add(ParagraphStyle(
            name='Normal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
        ))

        # Process content into paragraphs
        lines = content.strip().split('\n')
        elements = []
        
        # Handle name (first line)
        if lines:
            name = lines[0]
            elements.append(Paragraph(name, styles['Name']))
            
        # Handle contact info (second line)
        if len(lines) > 1:
            contact = lines[1]
            elements.append(Paragraph(contact, styles['Contact']))
        
        # Process the rest of the content
        current_section = None
        
        for i in range(2, len(lines)):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if it's a section heading (all caps)
            if line.isupper():
                current_section = line
                elements.append(Paragraph(line, styles['Heading']))
            else:
                # Check if it's a bullet point
                if line.startswith('- '):
                    line = '&#8226; ' + line[2:]  # Replace dash with bullet
                    elements.append(Paragraph(line, styles['Normal']))
                else:
                    elements.append(Paragraph(line, styles['Normal']))
        
        # Build the PDF
        doc.build(elements)
        
        print(f"PDF created successfully at: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")
        return False

if __name__ == "__main__":
    # Set default file paths
    input_file = "tests/sample_resumes/sample_resume.txt"
    output_file = "tests/sample_resumes/sample_resume.pdf"
    
    # Allow custom file paths via command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create the PDF
    result = create_pdf_from_text(input_file, output_file)
    
    if result:
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
    else:
        sys.exit(1) 