import unittest
import sys
import os
import re

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.analyzer import check_skill_match, normalize_text

class TestSkillMatching(unittest.TestCase):
    """Test cases for skill matching functionality"""

    def test_tech_skill_exact_matching(self):
        """Test that technical skills need exact matching"""
        # Marketing resume text
        marketing_resume = """
        SENIOR MARKETING MANAGER
        ABC Brands, New York, NY | June 2019 - Present
        - Developed and implemented comprehensive marketing strategies resulting in 35% increase in brand awareness
        - Managed a team of 5 marketing specialists and a $1.2M annual marketing budget
        - Led digital marketing campaigns across social media, email, and content platforms
        - Conducted market research and competitor analysis to identify trends and opportunities
        - Collaborated with sales team to align marketing efforts with sales goals

        SKILLS
        - Digital Marketing (Social Media, Email, Content)
        - Brand Management
        - Marketing Strategy Development
        - Campaign Management
        - Market Research & Analytics
        - Budget Management
        - Team Leadership
        - SEO/SEM
        - Content Marketing
        - Communication & Presentation
        """
        
        # Technical skills that shouldn't match
        technical_skills = [
            "typescript", "node.js", "express", "graphql", "postgresql", 
            "mongodb", "redis", "aws", "azure", "docker", "kubernetes", 
            "ci/cd", "microservices", "rest api"
        ]
        
        for skill in technical_skills:
            self.assertFalse(
                check_skill_match(marketing_resume, skill),
                f"Marketing resume should NOT match technical skill: {skill}"
            )
        
        # Marketing skills that should match
        marketing_skills = [
            "marketing", "digital marketing", "social media", "brand management", 
            "team leadership", "market research", "analytics"
        ]
        
        for skill in marketing_skills:
            self.assertTrue(
                check_skill_match(marketing_resume, skill),
                f"Marketing resume should match marketing skill: {skill}"
            )
    
    def test_strict_technical_word_boundaries(self):
        """Test that technical skill matches respect word boundaries"""
        tech_resume = "I have experience with MongoDB and PostgreSQL databases."
        
        # These should match
        self.assertTrue(check_skill_match(tech_resume, "mongodb"))
        self.assertTrue(check_skill_match(tech_resume, "postgresql"))
        
        # This should NOT match "express" inside "experience"
        experience_text = "I have extensive experience in project management."
        self.assertFalse(check_skill_match(experience_text, "express"))
        
        # Test for TypeScript (which should only match TypeScript, not just "script")
        typescript_text = "I use JavaScript and TypeScript in my projects."
        script_text = "I wrote a shell script to automate the process."
        
        self.assertTrue(check_skill_match(typescript_text, "typescript"))
        self.assertFalse(check_skill_match(script_text, "typescript"))
    
    def test_variant_matching(self):
        """Test matching of skill variants"""
        # Test Node.js variations
        nodejs_text = "Built REST APIs using Node.js and Express."
        node_text = "Experienced with Node and Express frameworks."
        
        self.assertTrue(check_skill_match(nodejs_text, "node.js"))
        self.assertTrue(check_skill_match(node_text, "node.js"))
        
        # Test PostgreSQL variations
        postgres_text = "Database experience includes Postgres and MongoDB."
        
        self.assertTrue(check_skill_match(postgres_text, "postgresql"))

if __name__ == "__main__":
    unittest.main() 