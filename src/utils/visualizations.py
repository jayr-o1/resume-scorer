"""
Visualization module for generating charts and visualizations for the resume analyzer
"""

import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import pandas as pd
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import altair as alt
import json

def create_radar_chart(categories: List[str], values: List[float], 
                       title: str = "Resume Score Breakdown", 
                       min_val: float = 0, max_val: float = 100) -> str:
    """
    Create a radar chart for the resume score breakdown
    
    Parameters:
    - categories: List of category names
    - values: List of values corresponding to each category
    - title: Chart title
    - min_val: Minimum value for the chart
    - max_val: Maximum value for the chart
    
    Returns:
    - Base64 encoded image string
    """
    # Number of variables
    N = len(categories)
    
    # Ensure we have values for all categories
    assert len(values) == N, "Categories and values must be the same length"
    
    # Repeat the first value to close the polygon
    values = np.append(values, values[0])
    
    # Calculate angles for each category
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Set up the figure
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Set y-axis limits
    ax.set_ylim(min_val, max_val)
    
    # Plot the data and fill the area
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Add title
    ax.set_title(title, size=15, pad=20)
    
    # Add grid lines
    ax.grid(True)
    
    # Use BytesIO to capture the image
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML/JSON
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def create_radar_chart_altair(data: Dict[str, float], title: str = "Resume Score Breakdown") -> Dict:
    """
    Create a radar chart using Altair for web rendering
    
    Parameters:
    - data: Dictionary mapping categories to values
    - title: Chart title
    
    Returns:
    - Altair chart specification as JSON
    """
    # Simplified approach using a basic bar chart instead of a radar chart
    # Convert data to DataFrame
    df = pd.DataFrame({"category": list(data.keys()), "value": list(data.values())})
    
    # Create a simple bar chart that will work reliably
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('category:N', sort='-x', title='Category'),
        x=alt.X('value:Q', scale=alt.Scale(domain=[0, 100]), title='Score'),
        color=alt.Color('category:N', legend=None),
        tooltip=['category:N', 'value:Q']
    ).properties(
        title=title,
        width=400,
        height=300
    )
    
    return json.loads(chart.to_json())

def create_skill_radar(analysis_result: Dict) -> str:
    """
    Create a radar chart specifically for skills breakdown
    
    Parameters:
    - analysis_result: Analysis result dictionary
    
    Returns:
    - Base64 encoded image string
    """
    # Extract data from the analysis result
    skills_match = analysis_result.get("skills_match", {})
    matched = len(skills_match.get("matched_skills", []))
    total = matched + len(skills_match.get("missing_skills", []))
    
    # Get experience data
    exp = analysis_result.get("experience", {})
    required_years = exp.get("required_years", "0")
    applicant_years = exp.get("applicant_years", "0")
    
    # Convert to numeric if possible
    if isinstance(required_years, str) and required_years.isdigit():
        required_years = int(required_years)
    else:
        required_years = 0
        
    if isinstance(applicant_years, str) and applicant_years.isdigit():
        applicant_years = int(applicant_years)
    else:
        applicant_years = 0
    
    # Calculate experience score (capped at 150% of required)
    exp_score = min(100, (applicant_years / max(1, required_years)) * 100) if required_years > 0 else 50
    
    # Get education score
    edu = analysis_result.get("education", {})
    edu_score = 100 if edu.get("assessment") == "Meets Requirement" else 50
    
    # Get certifications score
    certs = analysis_result.get("certifications", {})
    cert_count = len(certs.get("relevant_certs", []))
    cert_score = min(100, cert_count * 25)
    
    # Get keyword match score
    keywords = analysis_result.get("keywords", {})
    matched_kw = int(keywords.get("matched", 0))
    total_kw = int(keywords.get("total", 1))
    keyword_score = (matched_kw / max(1, total_kw)) * 100
    
    # Create the radar chart
    categories = ["Skills", "Experience", "Education", "Certifications", "Keywords"]
    values = [
        (matched / max(1, total)) * 100, 
        exp_score, 
        edu_score, 
        cert_score, 
        keyword_score
    ]
    
    return create_radar_chart(categories, values, "Resume Skill Breakdown")

def create_detailed_skills_breakdown(skill_details: List[Dict]) -> Dict:
    """
    Create a detailed breakdown of skills by category and proficiency
    
    Parameters:
    - skill_details: List of skill detail dictionaries
    
    Returns:
    - Chart data for visualizing skills by category and proficiency
    """
    if not skill_details:
        return {"error": "No skill details available"}
    
    # Group skills by category
    categories = {}
    for skill in skill_details:
        category = skill.get("category", "other")
        if category not in categories:
            categories[category] = []
        categories[category].append(skill)
    
    # Count skills by proficiency in each category
    result = {}
    for category, skills in categories.items():
        proficiency_counts = {
            "expert": 0,
            "intermediate": 0,
            "beginner": 0,
            "unknown": 0
        }
        
        for skill in skills:
            prof = skill.get("proficiency", "unknown")
            proficiency_counts[prof] += 1
        
        result[category] = proficiency_counts
    
    return result

def create_keyword_cloud(text: str, job_text: str, title: str = "Keyword Density", 
                        width: int = 800, height: int = 400) -> str:
    """
    Create a word cloud showing keyword density
    
    Parameters:
    - text: Text from the resume
    - job_text: Text from the job description
    - title: Cloud title
    - width: Image width
    - height: Image height
    
    Returns:
    - Base64 encoded image string
    """
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width/100, height/100), dpi=100)
    
    # Generate word cloud for resume
    wordcloud1 = WordCloud(width=width//2, height=height, background_color='white', 
                         max_words=100, contour_width=1, contour_color='steelblue').generate(text)
    
    # Generate word cloud for job description
    wordcloud2 = WordCloud(width=width//2, height=height, background_color='white',
                         max_words=100, contour_width=1, contour_color='firebrick').generate(job_text)
    
    # Display the clouds
    ax1.imshow(wordcloud1, interpolation='bilinear')
    ax1.set_title('Resume Keywords')
    ax1.axis('off')
    
    ax2.imshow(wordcloud2, interpolation='bilinear')
    ax2.set_title('Job Description Keywords')
    ax2.axis('off')
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(pad=3)
    
    # Use BytesIO to capture the image
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML/JSON
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def create_comparison_chart(resume_data: Dict, benchmark_data: Dict, 
                           title: str = "Resume vs. Industry Standard") -> Dict:
    """
    Create a comparison bar chart between resume scores and industry benchmarks
    
    Parameters:
    - resume_data: Resume analysis data
    - benchmark_data: Industry benchmark data
    - title: Chart title
    
    Returns:
    - Altair chart specification as JSON
    """
    # Check for empty inputs and provide defaults if needed
    if resume_data is None:
        resume_data = {}
    if benchmark_data is None:
        benchmark_data = {"benchmarks": {}}
        
    # Extract benchmark data
    benchmarks = benchmark_data.get("benchmarks", {})
    if benchmarks is None:
        benchmarks = {}
    
    # Prepare data
    categories = ["Skills", "Experience", "Education", "Overall"]
    resume_values = []
    benchmark_values = []
    
    # Calculate resume values
    skills_match = resume_data.get("skills_match", {}) or {}
    matched = len(skills_match.get("matched_skills", []) or [])
    total = matched + len(skills_match.get("missing_skills", []) or [])
    skills_score = (matched / max(1, total)) * 100
    
    exp = resume_data.get("experience", {}) or {}
    if isinstance(exp.get("applicant_years"), str) and exp.get("applicant_years").isdigit():
        applicant_years = int(exp.get("applicant_years"))
    else:
        applicant_years = 0
        
    if isinstance(exp.get("required_years"), str) and exp.get("required_years").isdigit():
        required_years = int(exp.get("required_years"))
    else:
        required_years = 1
    
    exp_score = min(100, (applicant_years / max(1, required_years)) * 100)
    
    edu = resume_data.get("education", {}) or {}
    edu_score = 100 if edu.get("assessment") == "Meets Requirement" else 50
    
    overall_score = int(resume_data.get("match_percentage", 0) or 0)
    
    resume_values = [skills_score, exp_score, edu_score, overall_score]
    benchmark_values = [
        benchmarks.get("skills", 70),  # Default values if not provided
        benchmarks.get("experience", 60),
        benchmarks.get("education", 80),
        benchmarks.get("overall", 75)
    ]
    
    # Create DataFrame for Altair - using a simpler long-form format
    data = []
    for i, category in enumerate(categories):
        data.append({"category": category, "value": resume_values[i], "source": "Resume"})
        data.append({"category": category, "value": benchmark_values[i], "source": "Industry Benchmark"})
    
    df = pd.DataFrame(data)
    
    # Create a simple grouped bar chart that will work reliably
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('source:N', title='Source'),
        y=alt.Y('value:Q', title='Score', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('source:N', 
                       scale=alt.Scale(domain=['Resume', 'Industry Benchmark'],
                                     range=['#5276A7', '#57A44C'])),
        column=alt.Column('category:N', title='Category'),
        tooltip=['category:N', 'source:N', 'value:Q']
    ).properties(
        title=title,
        width=150
    )
    
    return json.loads(chart.to_json())

def create_missing_skills_chart(missing_skills: List[str], 
                                alternative_skills: Dict[str, List[str]]) -> Dict:
    """
    Create a visualization of missing skills and their alternatives
    
    Parameters:
    - missing_skills: List of missing skills
    - alternative_skills: Dictionary mapping missing skills to alternative skills
    
    Returns:
    - Altair chart specification as JSON
    """
    if not missing_skills:
        return {"error": "No missing skills to visualize"}
    
    # Create a simpler bar chart visualization of missing skills
    data = []
    
    # Add missing skills
    for skill in missing_skills:
        data.append({
            "skill": skill,
            "type": "Missing Skill",
            "count": 1
        })
        
        # Add alternative skills if available
        if skill in alternative_skills and alternative_skills[skill]:
            for alt_skill in alternative_skills[skill][:3]:  # Limit to top 3 alternatives
                data.append({
                    "skill": alt_skill,
                    "type": "Alternative to " + skill,
                    "count": 0.7  # Slightly smaller bars for alternatives
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create a simple bar chart
    if not df.empty:
        # Prepare domain list for color scale
        color_domains = ['Missing Skill']
        for s in missing_skills:
            color_domains.append(f'Alternative to {s}')
        
        # Prepare range list with appropriate colors
        color_ranges = ['#E45756'] + ['#54A24B' for _ in missing_skills]
        
        chart = alt.Chart(df).mark_bar().encode(
            y=alt.Y('skill:N', title='Skills'),
            x=alt.X('count:Q', title='', axis=None),
            color=alt.Color('type:N', 
                         scale=alt.Scale(domain=color_domains,
                                        range=color_ranges)),
            tooltip=['skill:N', 'type:N']
        ).properties(
            title="Missing Skills and Alternatives",
            width=500,
            height=min(500, len(data) * 25)  # Dynamic height based on number of skills
        )
        
        return json.loads(chart.to_json())
    else:
        return {"error": "No data available for visualization"} 