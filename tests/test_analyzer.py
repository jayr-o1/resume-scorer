import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from utils.analyzer import (
    normalize_text,
    extract_experiences,
    extract_education,
    extract_certifications,
    extract_job_titles,
    extract_employment_durations,
    check_skill_match
)

class TestAnalyzer(unittest.TestCase):
    """Test cases for the analyzer module"""
    
    def test_normalize_text(self):
        """Test text normalization function"""
        input_text = "This is a test with some â€¢ weird characters  and â€" spacing."
        expected = "this is a test with some • weird characters and - spacing."
        result = normalize_text(input_text)
        self.assertEqual(result, expected)
    
    def test_extract_experiences(self):
        """Test experience extraction function"""
        # Test various patterns
        test_cases = [
            ("I have 5 years of experience in Python.", 5),
            ("With over 7 years experience in software development.", 7),
            ("My career of 10+ years in data science.", 10),
            ("I have a 3-year experience in project management.", 3),
            ("No experience mentioned here.", None)
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = extract_experiences(text)
                self.assertEqual(result, expected)
    
    def test_extract_education(self):
        """Test education level extraction"""
        # Test various education levels
        test_cases = [
            ("I have a PhD in Computer Science from MIT.", "PhD/Doctorate"),
            ("Master's degree in Engineering from Stanford.", "Master's degree"),
            ("Bachelor of Science in Physics from Harvard.", "Bachelor's degree"),
            ("Associate degree in Business Administration.", "Associate degree"),
            ("High School diploma from Lincoln High.", "High School"),
            ("No education mentioned here.", "Not specified")
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = extract_education(text)
                self.assertEqual(result, expected)
    
    def test_extract_certifications(self):
        """Test certification extraction"""
        # Test various certification patterns
        text = """
        I am AWS Certified Solutions Architect and hold the CCNA certification.
        Other certifications include Security+ and Project Management Professional (PMP).
        I recently completed the Certified Kubernetes Administrator (CKA) program.
        """
        
        result = extract_certifications(text)
        
        # Check some expected certifications
        self.assertIn("AWS Certified Solutions Architect", result)
        self.assertIn("CCNA", result)
        self.assertIn("Security+", result)
        self.assertIn("PMP", result)
        self.assertIn("CKA", result)
    
    def test_extract_job_titles(self):
        """Test job title extraction"""
        text = """
        Career History:
        
        Senior Software Engineer at Google (2018-2020)
        Led a team of developers and implemented new features.
        
        Frontend Developer at Facebook (2015-2018)
        Developed responsive user interfaces using React.
        
        Junior QA Engineer at Microsoft (2013-2015)
        Performed testing and quality assurance tasks.
        """
        
        result = extract_job_titles(text)
        
        # Check expected job titles
        self.assertIn("Senior Software Engineer", result)
        self.assertIn("Frontend Developer", result)
        self.assertIn("Junior QA Engineer", result)
    
    def test_extract_employment_durations(self):
        """Test employment duration extraction"""
        text = """
        Work Experience:
        
        Google - Jan 2018 - Present
        Software Engineer
        
        Facebook - 05/2015 - 12/2017
        Frontend Developer
        
        Microsoft - 01.2013 - 04.2015
        QA Engineer
        """
        
        result = extract_employment_durations(text)
        
        # Should find three employment periods
        self.assertEqual(len(result), 3)
        
        # Check that we have start and end dates
        for duration in result:
            self.assertIn("start_date", duration)
            self.assertIn("end_date", duration)
    
    def test_check_skill_match(self):
        """Test skill matching function"""
        resume_text = """
        Skills:
        - Python programming
        - JavaScript, React.js
        - AWS cloud services
        - MongoDB database
        - Git version control
        """
        
        # Test direct matches
        self.assertTrue(check_skill_match(resume_text, "python"))
        self.assertTrue(check_skill_match(resume_text, "javascript"))
        self.assertTrue(check_skill_match(resume_text, "aws"))
        
        # Test aliases/variants
        self.assertTrue(check_skill_match(resume_text, "react"))
        self.assertTrue(check_skill_match(resume_text, "mongo"))
        
        # Test non-matches
        self.assertFalse(check_skill_match(resume_text, "docker"))
        self.assertFalse(check_skill_match(resume_text, "java"))

if __name__ == "__main__":
    unittest.main() 