import streamlit as st
import os
import tempfile
import sys
import altair as alt
import pandas as pd

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.pdf_extractor import extract_text_from_pdf
from utils.analyzer import analyze_resume, format_analysis_result

st.set_page_config(page_title="AI Resume Scorer", layout="wide")

def main():
    st.title("AI Resume Scorer")
    st.markdown("Upload a resume and job details to get AI-powered insights")
    
    # Industry selection
    industries = ["tech", "finance", "healthcare", "marketing"]
    selected_industry = st.selectbox(
        "Select Industry (Optional)", 
        ["Auto-detect"] + industries, 
        help="The system will auto-detect the industry, but you can override it here"
    )
    
    # File upload for resume
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    
    # Job details input
    st.subheader("Job Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_summary = st.text_area("Job Summary", height=150)
        key_duties = st.text_area("Key Duties", height=150)
    
    with col2:
        essential_skills = st.text_area("Essential Skills", height=150)
        qualifications = st.text_area("Qualifications", height=150)
    
    analyze_button = st.button("Analyze Resume")
    
    if analyze_button and uploaded_file is not None:
        if not (job_summary or key_duties or essential_skills or qualifications):
            st.error("Please provide at least some job details.")
            return
        
        with st.spinner("Analyzing resume..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            # Extract text from PDF
            resume_text = extract_text_from_pdf(temp_file_path)
            
            # Remove the temporary file
            os.unlink(temp_file_path)
            
            if resume_text:
                # Prepare job details
                job_details = {
                    "summary": job_summary,
                    "duties": key_duties,
                    "skills": essential_skills,
                    "qualifications": qualifications
                }
                
                # Add industry override if specified
                if selected_industry != "Auto-detect":
                    job_details["industry_override"] = selected_industry
                
                # Analyze the resume
                analysis = analyze_resume(resume_text, job_details)
                
                # Display the formatted results
                st.subheader("Analysis Results")
                
                # Create columns for the results
                result_col1, result_col2 = st.columns([2, 1])
                
                with result_col1:
                    # Show formatted results
                    formatted_results = format_analysis_result(analysis)
                    st.markdown(f"```\n{formatted_results}\n```")
                
                with result_col2:
                    # Show key metrics in a more visual way
                    if "match_percentage" in analysis:
                        st.metric("Match Percentage", f"{analysis['match_percentage']}%")
                    
                    if "skills_match" in analysis:
                        st.metric("Skills Match", analysis['skills_match']['match_ratio'])
                    
                    if "industry" in analysis:
                        st.metric("Industry", analysis['industry']['detected'].capitalize())
                    
                    if "recommendation" in analysis:
                        rec = analysis['recommendation']
                        color = "green" if "hire" in rec.lower() else "red"
                        st.markdown(f"<h3 style='color: {color};'>Recommendation: {rec}</h3>", unsafe_allow_html=True)
                    
                    # Show benchmark results if available
                    if "benchmark" in analysis and "benchmarks" in analysis["benchmark"]:
                        st.subheader("Industry Benchmarks")
                        benchmarks = analysis["benchmark"]["benchmarks"]
                        
                        # Create data for chart
                        chart_data = pd.DataFrame({
                            'category': ['Skills', 'Experience', 'Education', 'Overall'],
                            'percentage': [
                                benchmarks['skills'], 
                                benchmarks['experience'], 
                                benchmarks['education'],
                                benchmarks['overall']
                            ]
                        })
                        
                        # Create bar chart
                        chart = alt.Chart(chart_data).mark_bar().encode(
                            x=alt.X('percentage:Q', scale=alt.Scale(domain=[0, 100])),
                            y=alt.Y('category:N'),
                            color=alt.condition(
                                alt.datum.percentage >= 70,
                                alt.value('green'),
                                alt.value('orange')
                            ),
                            tooltip=['category', 'percentage']
                        ).properties(
                            title='Industry Benchmark Comparison'
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                
                # Show improvement suggestions if available
                if "improvement_suggestions" in analysis:
                    suggestions = analysis["improvement_suggestions"]
                    if any(suggestions.values()):
                        st.subheader("Improvement Suggestions")
                        
                        for category, items in suggestions.items():
                            if items:
                                with st.expander(f"{category.capitalize()} Improvements"):
                                    for item in items:
                                        st.markdown(f"- {item}")
                
                # Show missing skills as tags if available
                if "skills_match" in analysis and analysis["skills_match"].get("missing_skills"):
                    st.subheader("Missing Skills")
                    missing_skills = analysis["skills_match"]["missing_skills"]
                    
                    # Display as horizontal pills/tags
                    cols = st.columns(4)
                    for i, skill in enumerate(missing_skills):
                        col_idx = i % 4
                        cols[col_idx].markdown(f"<div style='background-color: #f0f2f6; padding: 8px 12px; border-radius: 16px; margin: 4px; display: inline-block;'>{skill}</div>", unsafe_allow_html=True)
                
                # Show extracted text (collapsible)
                with st.expander("View Extracted Resume Text"):
                    st.text(resume_text)
            else:
                st.error("Failed to extract text from the PDF. Please try another file.")
    
    # Show sample data if needed
    with st.expander("Need sample data?"):
        st.markdown("""
        ### Sample Job Details
        
        #### Job Summary
        We are seeking a Full Stack Developer with 5+ years of experience to join our team. The ideal candidate will be responsible for developing and maintaining web applications, collaborating with cross-functional teams, and ensuring high-quality code.
        
        #### Key Duties
        - Develop front-end and back-end components for web applications
        - Collaborate with UX/UI designers to implement user interfaces
        - Write clean, maintainable, and efficient code
        - Troubleshoot and debug applications
        - Optimize applications for maximum speed and scalability
        
        #### Essential Skills
        - React, Node.js, MongoDB, Express.js, JavaScript
        - RESTful API design and development
        - Git version control
        - Agile development methodologies
        - Unit testing and debugging
        
        #### Qualifications
        - Bachelor's degree in Computer Science or related field
        - 5+ years of experience in full-stack development
        - Strong understanding of web development principles
        - AWS certification is a plus
        - Experience with CI/CD pipelines
        """)

if __name__ == "__main__":
    main() 