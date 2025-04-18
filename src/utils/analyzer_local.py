import os
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_experiences(text):
    """Extract years of experience from resume text"""
    # Look for patterns like "X years of experience" or "X+ years"
    patterns = [
        r'(\d+)\+?\s+years?\s+(?:of\s+)?experience',
        r'experience\s+(?:of\s+)?(\d+)\+?\s+years?',
        r'(\d+)-year\s+(?:of\s+)?experience'
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            years.extend([int(y) for y in matches])
    
    return max(years) if years else None

def extract_education(text):
    """Extract education level from resume text"""
    education_levels = {
        'phd': "PhD",
        'doctor': "Doctorate",
        'master': "Master's degree",
        'mba': "MBA",
        'bachelor': "Bachelor's degree",
        'bs ': "Bachelor of Science",
        'ba ': "Bachelor of Arts",
        'associate': "Associate degree",
        'high school': "High School"
    }
    
    found_levels = []
    for keyword, level in education_levels.items():
        if keyword in text.lower():
            found_levels.append(level)
    
    # Return the highest education level found
    if 'PhD' in found_levels or 'Doctorate' in found_levels:
        return "PhD/Doctorate"
    elif "Master's degree" in found_levels or "MBA" in found_levels:
        return "Master's degree"
    elif any(level for level in found_levels if "Bachelor" in level):
        return "Bachelor's degree"
    elif "Associate degree" in found_levels:
        return "Associate degree"
    elif "High School" in found_levels:
        return "High School"
    else:
        return "Not specified"

def extract_certifications(text):
    """Extract certifications from resume text"""
    # First normalize the text to handle encoding issues
    text = normalize_text(text)
    
    # Look for specific certifications by name
    specific_certs = [
        'aws certified', 'azure certified', 'gcp certified', 'google cloud certified',
        'comptia', 'cisco certified', 'ccna', 'ccnp', 'cissp', 'ceh', 'security+',
        'pmp', 'capm', 'scrum master', 'safe', 'itil', 'prince2',
        'cka', 'ckad', 'terraform', 'aws developer', 'aws solutions architect',
        'mongodb certified', 'oracle certified', 'mcsa', 'mcse', 'mcts',
        'rhce', 'rhcsa', 'lpic', 'cfa', 'cpa', 'six sigma'
    ]
    
    found_certs = []
    for cert in specific_certs:
        if cert in text:
            # Clean up and format the certification name
            clean_cert = cert.title()
            if cert.startswith('aws'):
                clean_cert = cert.upper().replace('AWS', 'AWS ')
            elif cert.startswith('azure'):
                clean_cert = cert.replace('azure', 'Azure')
            elif cert.startswith('gcp') or cert.startswith('google'):
                clean_cert = cert.replace('gcp', 'GCP').replace('google cloud', 'Google Cloud')
            
            found_certs.append(clean_cert)
    
    # Also use regex to find certification patterns
    cert_patterns = [
        r'(\w+)\s+certified\s+(developer|architect|professional|associate|engineer|administrator)',
        r'certified\s+(\w+)\s+(developer|architect|professional|associate|engineer|administrator)',
        r'(aws|azure|gcp|google cloud|comptia|cisco|oracle)\s+(certification|certified)',
    ]
    
    for pattern in cert_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                clean_cert = ' '.join([part.title() for part in match if part])
                if 'aws' in clean_cert.lower():
                    clean_cert = clean_cert.replace('Aws', 'AWS')
                found_certs.append(clean_cert)
    
    # Remove duplicates and very short entries
    found_certs = [cert for cert in found_certs if len(cert) > 3]
    found_certs = list(set(found_certs))
    
    return found_certs

def normalize_text(text):
    """Normalize text to handle encoding issues"""
    # Replace common encoding issues
    text = text.replace('â€¢', '•')
    text = text.replace('â€"', '-')
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    text = text.replace('â', '')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.lower().strip()

def check_skill_match(resume_text, skill):
    """Check if a skill is present in the resume text using more robust matching"""
    # Normalize both texts
    resume_text = normalize_text(resume_text)
    skill = normalize_text(skill)
    
    # Direct match
    if skill in resume_text:
        return True
    
    # Common variants for specific technologies
    skill_variants = {
        'react': ['reactjs', 'react.js'],
        'node.js': ['nodejs', 'node'],
        'express.js': ['expressjs', 'express'],
        'javascript': ['js', 'ecmascript'],
        'typescript': ['ts'],
        'mongodb': ['mongo'],
        'postgresql': ['postgres', 'psql'],
        'rest': ['restful', 'rest api', 'restapi'],
        'git': ['github', 'gitlab', 'version control'],
        'agile': ['scrum', 'kanban', 'sprint'],
        'ci/cd': ['continuous integration', 'continuous deployment', 'jenkins', 'github actions'],
        'unit testing': ['jest', 'mocha', 'testing', 'test driven']
    }
    
    # Check for variants
    for base_skill, variants in skill_variants.items():
        if skill == base_skill or skill in variants:
            # If the skill we're looking for matches this base_skill or is one of its variants
            if base_skill in resume_text:
                return True
            # Check if any variant is in the resume
            for variant in variants:
                if variant in resume_text:
                    return True
                    
    return False

def analyze_resume(resume_text, job_details):
    """
    Analyze resume against job details using a local model
    
    Parameters:
    - resume_text: Extracted text from the resume
    - job_details: Dictionary containing job summary, duties, skills, etc.
    
    Returns:
    - Dictionary with analysis results
    """
    try:
        # Load sentence-transformers model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Extract job requirements
        job_text = " ".join([
            job_details.get('summary', ''),
            job_details.get('duties', ''),
            job_details.get('skills', ''),
            job_details.get('qualifications', '')
        ])
        
        # Extract skills from job description
        skills_text = job_details.get('skills', '')
        skills_list = [s.strip().lower() for s in re.findall(r'[-•]?\s*([\w\s\+\#\.\-]+)(?:,|\n|$)', skills_text)]
        skills_list = [s for s in skills_list if len(s) > 2]  # Filter out very short items
        
        # Create embeddings
        resume_embedding = model.encode([resume_text])[0]
        job_embedding = model.encode([job_text])[0]
        
        # Calculate overall similarity
        similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        match_percentage = int(similarity * 100)
        
        # Extract skills matches using the improved matcher
        matched_skills = []
        missing_skills = []
        
        for skill in skills_list:
            if check_skill_match(resume_text, skill):
                matched_skills.append(skill)
            else:
                # Fall back to embedding similarity only if direct match fails
                skill_embedding = model.encode([skill])[0]
                skill_similarity = cosine_similarity([resume_embedding], [skill_embedding])[0][0]
                
                if skill_similarity > 0.5:  # Threshold for skill match
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)
        
        # Calculate match ratio
        total_skills = len(skills_list)
        matched_count = len(matched_skills)
        match_ratio = f"{matched_count}/{total_skills}"
        
        # Extract years of experience
        applicant_years = extract_experiences(resume_text) or "Not specified"
        required_years_match = re.search(r'(\d+)\+?\s+years?', job_details.get('qualifications', ''))
        required_years = required_years_match.group(1) if required_years_match else "Not specified"
        
        # Calculate experience impact on score
        if isinstance(applicant_years, int) and isinstance(required_years, str) and required_years.isdigit():
            req_years = int(required_years)
            if applicant_years < req_years:
                percentage_impact = f"-{min(20, (req_years - applicant_years) * 5)}%"
            elif applicant_years > req_years:
                percentage_impact = f"+{min(10, (applicant_years - req_years) * 2)}%"
            else:
                percentage_impact = "0%"
        else:
            percentage_impact = "0%"
        
        # Extract education
        applicant_education = extract_education(resume_text)
        required_education = extract_education(job_details.get('qualifications', ''))
        
        education_levels = ["Not specified", "High School", "Associate degree", "Bachelor's degree", "Master's degree", "PhD/Doctorate"]
        applicant_level = education_levels.index(applicant_education)
        required_level = education_levels.index(required_education)
        
        if applicant_level >= required_level:
            education_assessment = "Meets Requirement"
        else:
            education_assessment = "Below Requirement"
        
        # Extract certifications
        certifications = extract_certifications(resume_text)
        cert_impact = f"+{min(10, len(certifications) * 5)}%" if certifications else "0%"
        
        # Calculate keyword matching
        important_keywords = []
        for section in [job_details.get('summary', ''), job_details.get('duties', ''), job_details.get('qualifications', '')]:
            words = re.findall(r'\b\w{3,}\b', section.lower())
            important_keywords.extend([w for w in words if len(w) > 3])
        
        important_keywords = list(set(important_keywords))
        
        # Normalize resume text for keyword matching
        normalized_resume = normalize_text(resume_text)
        resume_words = set(re.findall(r'\b\w{3,}\b', normalized_resume))
        
        matched_keywords = [k for k in important_keywords if k in resume_words]
        keyword_ratio = f"{len(matched_keywords)}/{len(important_keywords)}"
        
        # Determine recommendation
        if match_percentage >= 85:
            recommendation = "Strong Hire"
        elif match_percentage >= 70:
            recommendation = "Hire"
        elif match_percentage >= 50:
            recommendation = "Consider"
        else:
            recommendation = "Reject"
        
        # Return the analysis results
        return {
            "match_percentage": str(match_percentage),
            "skills_match": {
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "match_ratio": match_ratio
            },
            "experience": {
                "required_years": str(required_years),
                "applicant_years": str(applicant_years),
                "percentage_impact": percentage_impact
            },
            "education": {
                "requirement": required_education,
                "applicant_education": applicant_education,
                "assessment": education_assessment
            },
            "certifications": {
                "relevant_certs": certifications,
                "percentage_impact": cert_impact
            },
            "keywords": {
                "matched": str(len(matched_keywords)),
                "total": str(len(important_keywords)),
                "match_ratio": keyword_ratio
            },
            "recommendation": recommendation
        }
        
    except Exception as e:
        print(f"Error during local analysis: {e}")
        return {
            "error": str(e),
            "match_percentage": "0",
            "recommendation": "Error during analysis"
        }

def format_analysis_result(analysis):
    """
    Format the analysis results into a readable string
    """
    if "error" in analysis:
        return f"Error: {analysis['error']}"
    
    result = "AI Insights\n"
    result += f"Match Percentage:\n{analysis['match_percentage']}%\n\n"
    
    # Skills match
    skills = analysis['skills_match']
    result += f"Skills Match:\n{', '.join(skills['matched_skills'])} ({skills['match_ratio']})\n\n"
    
    # Experience
    exp = analysis['experience']
    result += f"Experience:\nRequired: {exp['required_years']} | Applicant: {exp['applicant_years']} ({exp['percentage_impact']})\n\n"
    
    # Education
    edu = analysis['education']
    result += f"Education:\n{edu['applicant_education']} ({edu['assessment']})\n\n"
    
    # Certifications
    certs = analysis['certifications']
    if certs['relevant_certs']:
        result += f"Certifications:\n{', '.join(certs['relevant_certs'])} ({certs['percentage_impact']})\n\n"
    else:
        result += "Certifications:\nNone\n\n"
    
    # Keywords
    kw = analysis['keywords']
    result += f"Resume Keywords:\nMatched {kw['match_ratio']}\n\n"
    
    # Recommendation
    rec = analysis['recommendation']
    emoji = "✅" if "hire" in rec.lower() else "❌"
    result += f"Recommendation:\n{rec} {emoji}"
    
    return result 