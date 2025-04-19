import streamlit as st
import os
import tempfile
import sys
import altair as alt
import pandas as pd
import json
from pathlib import Path

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import the module functions directly
from utils.pdf_extractor import extract_text_from_pdf, extract_resume_sections
from utils.analyzer import analyze_resume, batch_process_resumes
from utils.skill_ontology import get_skill_ontology
from utils.visualizations import (
    create_skill_radar, 
    create_radar_chart_altair, 
    create_keyword_cloud,
    create_comparison_chart,
    create_missing_skills_chart,
    create_detailed_skills_breakdown
)

st.set_page_config(page_title="AI Resume Scorer", layout="wide")

def main():
    st.title("Enhanced AI Resume Scorer")
    st.markdown("Upload a resume and job details to get AI-powered insights with enhanced visualizations")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Single Resume Analysis", "Batch Processing", "Skill Ontology"])
    
    # Tab 1: Single Resume Analysis
    with tab1:
        # Industry selection
        industries = ["tech", "finance", "healthcare", "marketing"]
        selected_industry = st.selectbox(
            "Select Industry (Optional)", 
            ["Auto-detect"] + industries, 
            help="The system will auto-detect the industry, but you can override it here"
        )
        
        # File upload for resume
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        
        # Language options
        enable_translation = st.checkbox("Enable translation for non-English resumes", value=False)
        
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
                
                # Extract text from PDF with enhanced method
                extraction_result = extract_text_from_pdf(temp_file_path, translate=enable_translation)
                
                # Remove the temporary file
                os.unlink(temp_file_path)
                
                if extraction_result and extraction_result["text"]:
                    resume_text = extraction_result["text"]
                    # If sections are needed
                    resume_sections = extract_resume_sections(resume_text)
                    
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
                    analysis = analyze_resume(extraction_result, job_details)
                    
                    # Display the results
                    display_analysis_results(analysis, resume_text, job_details)
                else:
                    st.error("Failed to extract text from the PDF. Please try another file.")
    
    # Tab 2: Batch Processing
    with tab2:
        st.subheader("Batch Resume Processing")
        st.markdown("Upload multiple resumes to compare against the same job description")
        
        # Job details input for batch processing
        st.subheader("Job Details")
        batch_job_summary = st.text_area("Job Summary", key="batch_summary", height=100)
        batch_essential_skills = st.text_area("Essential Skills", key="batch_skills", height=100)
        
        # File uploader for multiple resumes
        uploaded_files = st.file_uploader("Upload Multiple Resumes (PDF)", type="pdf", accept_multiple_files=True)
        
        batch_analyze_button = st.button("Analyze Batch")
        
        if batch_analyze_button and uploaded_files:
            if not (batch_job_summary or batch_essential_skills):
                st.error("Please provide at least some job details.")
                return
            
            with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
                # Setup temp directory for PDFs
                temp_dir = tempfile.mkdtemp()
                pdf_paths = []
                
                # Save all uploaded files temporarily
                for uploaded_file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    pdf_paths.append(temp_file_path)
                
                # Prepare job details
                job_details = {
                    "summary": batch_job_summary,
                    "skills": batch_essential_skills
                }
                
                # Process in parallel
                results = batch_process_resumes(pdf_paths, job_details)
                
                # Clean up temp files
                for path in pdf_paths:
                    try:
                        os.unlink(path)
                    except:
                        pass
                
                # Display batch results
                display_batch_results(results)
    
    # Tab 3: Skill Ontology Management
    with tab3:
        st.subheader("Skill Ontology")
        st.markdown("View and manage the skill ontology used for skill normalization and detection")
        
        # Get skill ontology
        skill_ontology = get_skill_ontology()
        
        # Display existing skills
        st.markdown("### Current Skills in Ontology")
        
        # Group skills by category
        skills_by_category = {}
        for skill, info in skill_ontology.skills_map.items():
            category = info.get("category", "other")
            if category not in skills_by_category:
                skills_by_category[category] = []
            skills_by_category[category].append((skill, info))
        
        # Create expandable sections for each category
        for category, skills in skills_by_category.items():
            with st.expander(f"{category.title()} ({len(skills)})"):
                for skill, info in skills:
                    st.markdown(f"**{skill}**")
                    st.markdown(f"Aliases: {', '.join(info.get('aliases', []))}")
                    st.markdown(f"Related: {', '.join(info.get('related', []))}")
                    st.markdown("---")
        
        # Add new skill form
        st.markdown("### Add New Skill")
        
        skill_name = st.text_input("Skill Name")
        skill_aliases = st.text_input("Aliases (comma separated)")
        
        skill_categories = list(set([info.get("category", "other") for info in skill_ontology.skills_map.values()]))
        skill_category = st.selectbox("Category", skill_categories)
        
        skill_related = st.text_input("Related Skills (comma separated)")
        
        if st.button("Add Skill"):
            if skill_name:
                aliases = [a.strip() for a in skill_aliases.split(",")] if skill_aliases else []
                related = [r.strip() for r in skill_related.split(",")] if skill_related else []
                
                success = skill_ontology.add_skill(
                    skill_name, 
                    aliases=aliases, 
                    category=skill_category,
                    related=related
                )
                
                if success:
                    st.success(f"Added skill: {skill_name}")
                    st.rerun()
                else:
                    st.error("Failed to add skill")
            else:
                st.error("Skill name is required")

def display_analysis_results(analysis, resume_text, job_details):
    """Display the analysis results with enhanced visualizations"""
    st.subheader("Analysis Results")
    
    # Check for errors
    if "error" in analysis:
        st.error(f"Error during analysis: {analysis['error']}")
        return
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        match_percentage = analysis.get("match_percentage", "0")
        st.metric("Match Percentage", f"{match_percentage}%")
    
    with col2:
        skills_match = analysis.get("skills_match", {})
        st.metric("Skills Match", skills_match.get("match_ratio", "0/0"))
    
    with col3:
        industry = analysis.get("industry", {})
        st.metric("Industry", industry.get("detected", "Unknown").capitalize())
    
    with col4:
        recommendation = analysis.get("recommendation", "No recommendation")
        color = "green" if "hire" in recommendation.lower() else "red"
        st.markdown(f"<h3 style='color: {color};'>Recommendation: {recommendation}</h3>", unsafe_allow_html=True)
    
    # Create tabs for detailed results
    details_tab, viz_tab, improve_tab, data_tab = st.tabs(["Detailed Results", "Visualizations", "Improvements", "Raw Data"])
    
    with details_tab:
        # Skills section
        st.subheader("Skills Analysis")
        skills_col1, skills_col2 = st.columns(2)
        
        with skills_col1:
            st.markdown("#### Matched Skills")
            matched_skills = skills_match.get("matched_skills", [])
            for skill in matched_skills:
                st.markdown(f"- {skill}")
        
        with skills_col2:
            st.markdown("#### Missing Skills")
            missing_skills = skills_match.get("missing_skills", [])
            for skill in missing_skills:
                st.markdown(f"- {skill}")
        
        # Additional skills
        additional_skills = skills_match.get("additional_skills", [])
        if additional_skills:
            st.markdown("#### Additional Skills Detected")
            additional_col1, additional_col2 = st.columns(2)
            mid = len(additional_skills) // 2
            
            with additional_col1:
                for skill in additional_skills[:mid]:
                    st.markdown(f"- {skill}")
            
            with additional_col2:
                for skill in additional_skills[mid:]:
                    st.markdown(f"- {skill}")
        
        # Experience section
        st.subheader("Experience Analysis")
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            experience = analysis.get("experience", {})
            st.markdown(f"**Required Years:** {experience.get('required_years', 'Not specified')}")
            st.markdown(f"**Applicant Years:** {experience.get('applicant_years', 'Not specified')}")
            st.markdown(f"**Impact on Score:** {experience.get('percentage_impact', '0%')}")
        
        with exp_col2:
            # Display job titles if available
            job_titles = experience.get("job_titles", [])
            if job_titles:
                st.markdown("**Detected Job Titles:**")
                for title in job_titles:
                    st.markdown(f"- {title}")
            
            # Display employment durations if available
            emp_durations = experience.get("employment_durations", [])
            if emp_durations:
                st.markdown("**Detected Employment Periods:**")
                for duration in emp_durations:
                    st.markdown(f"- {duration.get('text', '')}")
        
        # Education and Certifications
        st.subheader("Qualifications")
        qual_col1, qual_col2 = st.columns(2)
        
        with qual_col1:
            education = analysis.get("education", {})
            st.markdown("#### Education")
            st.markdown(f"**Required:** {education.get('requirement', 'Not specified')}")
            st.markdown(f"**Applicant:** {education.get('applicant_education', 'Not specified')}")
            
            assessment = education.get("assessment", "")
            color = "green" if assessment == "Meets Requirement" else "red"
            st.markdown(f"**Assessment:** <span style='color: {color};'>{assessment}</span>", unsafe_allow_html=True)
        
        with qual_col2:
            certifications = analysis.get("certifications", {})
            st.markdown("#### Certifications")
            
            certs = certifications.get("relevant_certs", [])
            if certs:
                for cert in certs:
                    st.markdown(f"- {cert}")
            else:
                st.markdown("No certifications detected")
            
            st.markdown(f"**Impact:** {certifications.get('percentage_impact', '0%')}")
        
        # Salary estimate if available
        if "salary_estimate" in analysis:
            st.subheader("Salary Estimate")
            salary = analysis["salary_estimate"]
            st.markdown(f"**Estimated Range:** {salary.get('currency', 'USD')} {salary.get('min', 0):,} - {salary.get('max', 0):,}")
            st.markdown(f"*Note: {salary.get('note', '')}*")
    
    with viz_tab:
        # Create visualizations
        st.subheader("Score Breakdown")
        
        # Radar chart
        try:
            # Prepare data for radar chart
            radar_data = {
                "Skills": int(float(analysis.get("match_percentage", 0))),
                "Experience": min(100, int(float(analysis.get("experience", {}).get("applicant_years", 0))) * 10),
                "Education": 100 if analysis.get("education", {}).get("assessment") == "Meets Requirement" else 50,
                "Keywords": int(analysis.get("keywords", {}).get("matched", 0)) / max(1, int(analysis.get("keywords", {}).get("total", 1))) * 100,
                "Certifications": min(100, len(analysis.get("certifications", {}).get("relevant_certs", [])) * 25)
            }
            
            radar_chart = create_radar_chart_altair(radar_data)
            st.altair_chart(alt.Chart.from_dict(radar_chart), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating radar chart: {e}")
        
        # Create comparison to industry benchmark
        if "benchmark" in analysis:
            st.subheader("Industry Benchmark Comparison")
            try:
                benchmark_chart = create_comparison_chart(analysis, analysis["benchmark"])
                st.altair_chart(alt.Chart.from_dict(benchmark_chart), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating benchmark chart: {e}")
        
        # Create missing skills visualization if there are missing skills
        missing_skills = analysis.get("skills_match", {}).get("missing_skills", [])
        alternative_skills = analysis.get("skills_match", {}).get("alternative_skills", {})
        
        if missing_skills:
            st.subheader("Missing Skills Analysis")
            try:
                missing_chart = create_missing_skills_chart(missing_skills, alternative_skills)
                if "error" not in missing_chart:
                    # For network graph
                    if "nodes" in missing_chart:
                        st.json(missing_chart)  # Just show the data structure for now
                    else:
                        # For bar chart
                        st.altair_chart(alt.Chart.from_dict(missing_chart), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating missing skills chart: {e}")
        
        # Create skill proficiency breakdown if available
        skill_details = analysis.get("skills_match", {}).get("skill_details", [])
        if skill_details:
            st.subheader("Skill Proficiency Breakdown")
            try:
                skill_breakdown = create_detailed_skills_breakdown(skill_details)
                if "error" not in skill_breakdown:
                    # Convert to DataFrame for better display
                    rows = []
                    for category, proficiencies in skill_breakdown.items():
                        row = {"Category": category}
                        row.update(proficiencies)
                        rows.append(row)
                    
                    df = pd.DataFrame(rows)
                    st.dataframe(df.set_index("Category"), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating skill breakdown: {e}")
        
        # Create keyword cloud
        st.subheader("Keyword Comparison")
        try:
            # Combine job details for word cloud
            job_text = " ".join([
                job_details.get("summary", ""),
                job_details.get("duties", ""),
                job_details.get("skills", ""),
                job_details.get("qualifications", "")
            ])
            
            # Generate keyword cloud
            keyword_cloud = create_keyword_cloud(resume_text, job_text)
            st.image(f"data:image/png;base64,{keyword_cloud}", use_column_width=True)
        except Exception as e:
            st.error(f"Error creating keyword cloud: {e}")
    
    with improve_tab:
        st.subheader("Improvement Suggestions")
        
        # Display improvement suggestions
        suggestions = analysis.get("improvement_suggestions", {})
        
        if not any(suggestions.values()):
            st.success("No improvement suggestions - great job!")
        else:
            for category, items in suggestions.items():
                if items:
                    with st.expander(f"{category.title()} Improvements"):
                        for item in items:
                            st.markdown(f"- {item}")
        
        # Confidence scores if available
        if "confidence_scores" in analysis:
            st.subheader("Analysis Confidence")
            confidence = analysis["confidence_scores"]
            
            # Create a DataFrame for the confidence scores
            conf_data = {
                "Category": list(confidence.keys()),
                "Confidence": list(confidence.values())
            }
            
            df = pd.DataFrame(conf_data)
            
            # Create chart
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Confidence:Q', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Category:N', sort='-x'),
                color=alt.condition(
                    alt.datum.Confidence >= 70,
                    alt.value('green'),
                    alt.value('orange')
                ),
                tooltip=['Category:N', 'Confidence:Q']
            ).properties(
                title='Analysis Confidence Scores'
            )
            
            st.altair_chart(chart, use_container_width=True)
    
    with data_tab:
        st.subheader("Raw Data")
        st.json(analysis)

def display_batch_results(results):
    """Display batch processing results"""
    st.subheader("Batch Analysis Results")
    
    # Overall summary
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    avg_match = sum(int(r.get("match_percentage", 0)) for r in valid_results.values()) / max(1, len(valid_results))
    
    st.metric("Average Match Percentage", f"{avg_match:.1f}%")
    
    # Create DataFrame for comparison
    rows = []
    for filename, result in results.items():
        if "error" in result:
            continue
            
        rows.append({
            "Filename": filename,
            "Match %": int(result.get("match_percentage", 0)),
            "Skills Match": result.get("skills_match", {}).get("match_ratio", "0/0"),
            "Experience": result.get("experience", {}).get("applicant_years", "N/A"),
            "Education": result.get("education", {}).get("applicant_education", "N/A"),
            "Recommendation": result.get("recommendation", "N/A")
        })
    
    if rows:
        df = pd.DataFrame(rows)
        
        # Sort by match percentage
        df = df.sort_values(by="Match %", ascending=False)
        
        st.dataframe(df, use_container_width=True)
        
        # Create visualization of top resumes
        st.subheader("Top Candidates Comparison")
        
        chart_data = df.head(min(5, len(df)))
        
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Match %:Q', title='Match Percentage'),
            y=alt.Y('Filename:N', sort='-x', title='Resume'),
            color=alt.condition(
                alt.datum['Match %'] >= 70,
                alt.value('green'),
                alt.value('orange')
            ),
            tooltip=['Filename:N', 'Match %:Q', 'Skills Match:N', 'Experience:N', 'Recommendation:N']
        ).properties(
            title='Top Candidates by Match Percentage'
        )
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.error("No valid results to display")
    
    # Raw data
    with st.expander("Raw Results"):
        st.json(results)

# Show sample data in the sidebar
with st.sidebar:
    st.title("Resume Scorer")
    st.markdown("### How it works")
    st.markdown("""
    1. Upload a resume PDF
    2. Enter job details
    3. Click 'Analyze Resume'
    4. Get AI-powered insights
    """)
    
    st.markdown("---")
    
    st.markdown("### Enhanced Features")
    st.markdown("""
    - Multi-language resume support
    - Advanced skill ontology
    - Smart visualization 
    - Detailed improvement suggestions
    - Salary estimation
    """)
    
    st.markdown("---")
    
    # Sample data
    if st.button("Load Sample Data"):
        # This would populate the form with sample data
        st.success("Sample data loaded! Click 'Analyze Resume' to see results.")

if __name__ == "__main__":
    main() 