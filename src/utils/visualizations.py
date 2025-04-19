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
    # Convert data to long format
    df = pd.DataFrame({
        'category': list(data.keys()),
        'value': list(data.values())
    })
    
    # Calculate the angle for each category
    N = len(data)
    df['angle'] = [i * (360 / N) for i in range(N)]
    df['angle_rad'] = df['angle'] * np.pi / 180
    
    # Calculate x, y coordinates
    df['x'] = df['value'] * np.cos(df['angle_rad'])
    df['y'] = df['value'] * np.sin(df['angle_rad'])
    
    # Create the chart - using newer Altair syntax
    base = alt.Chart(df).encode(
        theta=alt.Theta('angle:Q', scale=alt.Scale(domain=[0, 360])),
        radius=alt.Radius('value:Q', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('category:N', legend=alt.Legend(title="Categories"))
    )
    
    # Points
    points = base.mark_point(size=100).encode(
        tooltip=['category:N', 'value:Q']
    )
    
    # Create a separate dataframe for the line to connect first and last points
    line_data = pd.DataFrame({
        'x': df['x'].tolist() + [df['x'].iloc[0]],
        'y': df['y'].tolist() + [df['y'].iloc[0]],
        'angle': df['angle'].tolist() + [df['angle'].iloc[0]]
    })
    
    # Lines - using the dataframe directly
    line = alt.Chart(line_data).mark_line(color='gray', strokeWidth=1).encode(
        x='x:Q',
        y='y:Q',
        order='angle:Q'
    )
    
    # Area
    area = base.mark_area(opacity=0.3)
    
    # Labels for categories
    labels = base.mark_text(align='center', baseline='middle', fontSize=12).encode(
        text='category:N',
        radius=alt.value(150)  # Place labels outside the chart
    )
    
    # Combine all layers
    chart = alt.layer(points, line, area, labels).properties(
        width=400,
        height=400,
        title=title
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
    # Extract benchmark data
    benchmarks = benchmark_data.get("benchmarks", {})
    
    # Prepare data
    categories = ["Skills", "Experience", "Education", "Overall"]
    resume_values = []
    benchmark_values = []
    
    # Calculate resume values
    skills_match = resume_data.get("skills_match", {})
    matched = len(skills_match.get("matched_skills", []))
    total = matched + len(skills_match.get("missing_skills", []))
    skills_score = (matched / max(1, total)) * 100
    
    exp = resume_data.get("experience", {})
    if isinstance(exp.get("applicant_years"), str) and exp.get("applicant_years").isdigit():
        applicant_years = int(exp.get("applicant_years"))
    else:
        applicant_years = 0
        
    if isinstance(exp.get("required_years"), str) and exp.get("required_years").isdigit():
        required_years = int(exp.get("required_years"))
    else:
        required_years = 1
    
    exp_score = min(100, (applicant_years / max(1, required_years)) * 100)
    
    edu = resume_data.get("education", {})
    edu_score = 100 if edu.get("assessment") == "Meets Requirement" else 50
    
    overall_score = int(resume_data.get("match_percentage", 0))
    
    resume_values = [skills_score, exp_score, edu_score, overall_score]
    benchmark_values = [
        benchmarks.get("skills", 0),
        benchmarks.get("experience", 0),
        benchmarks.get("education", 0),
        benchmarks.get("overall", 0)
    ]
    
    # Create DataFrame for Altair
    data = []
    for i, category in enumerate(categories):
        data.append({"category": category, "value": resume_values[i], "source": "Resume"})
        data.append({"category": category, "value": benchmark_values[i], "source": "Industry Benchmark"})
    
    df = pd.DataFrame(data)
    
    # Create Altair chart with updated syntax
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('source:N', title=None),
        y=alt.Y('value:Q', title='Score', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('source:N', scale=alt.Scale(
            domain=['Resume', 'Industry Benchmark'],
            range=['#5276A7', '#57A44C']
        )),
        column=alt.Column('category:N', title=None),
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
    
    # Prepare data for visualization
    nodes = []
    links = []
    
    # Add missing skills as nodes
    for i, skill in enumerate(missing_skills):
        nodes.append({
            "id": skill,
            "name": skill,
            "type": "missing"
        })
        
        # Add alternative skills and connections
        if skill in alternative_skills and alternative_skills[skill]:
            for alt_skill in alternative_skills[skill]:
                # Add node if not already present
                if not any(n["id"] == alt_skill for n in nodes):
                    nodes.append({
                        "id": alt_skill,
                        "name": alt_skill,
                        "type": "alternative"
                    })
                
                # Add link
                links.append({
                    "source": skill,
                    "target": alt_skill,
                    "value": 1
                })
    
    # Create dataframes for nodes and links
    nodes_df = pd.DataFrame(nodes)
    links_df = pd.DataFrame(links)
    
    # If using Altair for network visualization (basic version)
    if not links_df.empty and not nodes_df.empty:
        # Create a chart for nodes
        node_chart = alt.Chart(nodes_df).mark_circle(size=100).encode(
            x=alt.X('id:N', axis=None),
            y=alt.Y('type:N', axis=alt.Axis(title='Type')),
            color=alt.Color('type:N', scale=alt.Scale(
                domain=['missing', 'alternative'],
                range=['#E45756', '#54A24B']
            )),
            tooltip=['name:N', 'type:N']
        )
        
        # Create text labels
        text_chart = alt.Chart(nodes_df).mark_text(dy=-10).encode(
            x=alt.X('id:N', axis=None),
            y=alt.Y('type:N'),
            text='name:N'
        )
        
        # Combine
        chart = alt.layer(node_chart, text_chart).properties(
            width=600,
            height=200,
            title="Missing Skills and Alternatives"
        )
        
        return json.loads(chart.to_json())
    else:
        # Fallback to simple text representation
        return {
            "nodes": nodes,
            "links": links,
            "error": "Not enough data for visualization"
        } 