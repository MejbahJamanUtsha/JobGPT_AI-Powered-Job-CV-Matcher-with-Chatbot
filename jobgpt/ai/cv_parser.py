"""
CV Parser Module for JobGPT

This module handles extraction of text from PDF resumes/CVs and uses
a spaCy NER model to identify key entities like skills, education, etc.
"""

import os
import re
import PyPDF2
import pdfplumber
from typing import Dict, List, Any, Optional, Tuple
import spacy
from pathlib import Path

# Will load custom NER model once trained
# For now we'll use a base model and extend it
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

class CVParser:
    """Parse resume/CV documents and extract structured information."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the CV parser.
        
        Args:
            model_path: Path to a custom spaCy NER model for CV parsing
        """
        self.nlp = nlp
        if model_path and os.path.exists(model_path):
            print(f"Loading custom NER model from {model_path}")
            self.ner_model = spacy.load(model_path)
        else:
            self.ner_model = self.nlp
            
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        
        # Try with PyPDF2 first
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            
        # If PyPDF2 extraction yields little text, try pdfplumber
        if len(text.strip()) < 100:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            except Exception as e:
                print(f"pdfplumber extraction failed: {e}")
                
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess the extracted text.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Preprocessed text
        """
        # Replace multiple newlines with a single one
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with parsing
        text = re.sub(r'[^\w\s@.,;:()\-\']', ' ', text)
        
        return text.strip()
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """
        Extract contact information from the text.
        
        Args:
            text: Preprocessed text from CV/resume
            
        Returns:
            Dictionary containing contact information
        """
        contact_info = {
            'email': None,
            'phone': None,
            'linkedin': None
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            contact_info['email'] = email_matches[0]
            
        # Phone pattern (various formats)
        phone_pattern = r'(\+?[\d\s\(\)\-\.]{10,20})'
        phone_matches = re.findall(phone_pattern, text)
        if phone_matches:
            # Clean up phone number format
            phone = re.sub(r'[^\d+]', '', phone_matches[0])
            if len(phone) >= 10:  # Ensure it's a valid length
                contact_info['phone'] = phone
                
        # LinkedIn URL pattern
        linkedin_pattern = r'linkedin\.com/in/[\w\-]+'
        linkedin_matches = re.findall(linkedin_pattern, text)
        if linkedin_matches:
            contact_info['linkedin'] = "https://" + linkedin_matches[0]
            
        return contact_info
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from the text.
        
        Args:
            text: Preprocessed text from CV/resume
            
        Returns:
            List of extracted skills
        """
        # This will be enhanced with the custom NER model
        # For now, using a basic approach with spaCy entities and keywords
        
        doc = self.ner_model(text)
        skills = []
        
        # Common technical skills keywords (to be expanded)
        tech_skills = [
            "python", "javascript", "java", "c++", "c#", "ruby", "php", "swift",
            "react", "angular", "vue", "node.js", "django", "flask", "fastapi", 
            "docker", "kubernetes", "aws", "azure", "gcp", "sql", "nosql", "mongodb",
            "postgresql", "mysql", "oracle", "git", "machine learning", "deep learning",
            "nlp", "data science", "data analysis", "tensorflow", "pytorch", "keras",
            "scikit-learn", "pandas", "numpy", "matplotlib", "tableau", "power bi",
            "excel", "word", "powerpoint", "photoshop", "illustrator", "figma",
            "html", "css", "sass", "less", "webpack", "babel", "typescript"
        ]
        
        # Check for skills mentions
        for skill in tech_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                skills.append(skill)
                
        # Extract skills from "Skills" section if present
        skills_section_pattern = r'(?i)skills?.*?(?:\n\n|\Z)'
        skills_section_match = re.search(skills_section_pattern, text, re.DOTALL)
        
        if skills_section_match:
            skills_text = skills_section_match.group(0)
            # Extract structured skills from the section
            # This is a simplified approach and will be enhanced with NER
            skill_candidates = re.findall(r'\b[A-Za-z+#]+(?:\s[A-Za-z+#]+){0,2}\b', skills_text)
            for skill in skill_candidates:
                if len(skill) > 2 and skill.lower() not in ['and', 'the', 'skills', 'skill']:
                    skills.append(skill.strip())
                    
        # Remove duplicates and sort
        return sorted(list(set(skills)))
    
    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """
        Extract education information from the text.
        
        Args:
            text: Preprocessed text from CV/resume
            
        Returns:
            List of dictionaries containing education details
        """
        education = []
        
        # Look for education section
        education_pattern = r'(?i)education|academic|qualification'
        education_match = re.search(education_pattern, text)
        
        if education_match:
            # Get text from education section start to next section or end
            start_idx = education_match.start()
            next_section_pattern = r'\n\s*(?:experience|work|employment|skills|projects|publications)'
            next_section_match = re.search(next_section_pattern, text[start_idx:], re.IGNORECASE)
            
            if next_section_match:
                education_text = text[start_idx:start_idx + next_section_match.start()]
            else:
                # If no next section found, take a reasonable chunk
                education_text = text[start_idx:start_idx + 1500]
                
            # Extract degree information
            degree_patterns = [
                r'(?i)(ph\.?d\.?|doctor of philosophy)',
                r'(?i)(m\.?s\.?|master of science)',
                r'(?i)(m\.?b\.?a\.?|master of business administration)',
                r'(?i)(b\.?s\.?|bachelor of science)',
                r'(?i)(b\.?a\.?|bachelor of arts)',
                r'(?i)(b\.?tech\.?|bachelor of technology)',
                r'(?i)(m\.?tech\.?|master of technology)'
            ]
            
            for pattern in degree_patterns:
                matches = re.finditer(pattern, education_text)
                for match in matches:
                    # Try to extract year and institution near the degree
                    context = education_text[max(0, match.start() - 100):min(len(education_text), match.end() + 200)]
                    
                    year_pattern = r'(19|20)\d{2}'
                    years = re.findall(year_pattern, context)
                    
                    institution_pattern = r'(?i)(?:university|college|institute|school) of ([A-Z][a-z]+(?: [A-Z][a-z]+){0,5})'
                    institutions = re.findall(institution_pattern, context)
                    
                    education_entry = {
                        'degree': match.group(0),
                        'institution': institutions[0] if institutions else None,
                        'year': years[0] if years else None
                    }
                    
                    education.append(education_entry)
        
        return education
    
    def extract_experience(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract work experience information from the text.
        
        Args:
            text: Preprocessed text from CV/resume
            
        Returns:
            List of dictionaries containing work experience details
        """
        experience = []
        
        # Look for experience section
        experience_pattern = r'(?i)experience|employment|work history'
        experience_match = re.search(experience_pattern, text)
        
        if experience_match:
            # Get text from experience section start to next section or end
            start_idx = experience_match.start()
            next_section_pattern = r'\n\s*(?:education|academic|skills|projects|publications|references)'
            next_section_match = re.search(next_section_pattern, text[start_idx:], re.IGNORECASE)
            
            if next_section_match:
                experience_text = text[start_idx:start_idx + next_section_match.start()]
            else:
                # If no next section found, take a reasonable chunk
                experience_text = text[start_idx:start_idx + 2000]
                
            # Extract company and position patterns
            # This is a simplified approach and will be enhanced with NER
            companies = re.findall(r'(?i)(?:at|with|for)?\s*([A-Z][a-zA-Z0-9&\s,]+(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company|Co\.?))', experience_text)
            positions = re.findall(r'(?i)(Senior|Junior|Lead|Principal|Software|Data|Product|Project|Engineering|Manager|Director|Developer|Engineer|Architect|Analyst|Consultant|Specialist|Associate)(?:\s+[A-Z][a-z]+){0,3}', experience_text)
            dates = re.findall(r'(?i)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4}\s*(?:-|–|to)\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4}|Present|Current)', experience_text)
            
            # Pair companies with positions and dates if possible
            for i in range(min(len(companies), len(positions))):
                experience_entry = {
                    'company': companies[i].strip(),
                    'position': positions[i].strip(),
                    'dates': dates[i] if i < len(dates) else None
                }
                experience.append(experience_entry)
        
        return experience
    
    def parse_cv(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a CV/resume PDF and extract structured information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing structured CV/resume information
        """
        # Extract and preprocess text
        raw_text = self.extract_text_from_pdf(pdf_path)
        processed_text = self.preprocess_text(raw_text)
        
        # Extract information
        contact_info = self.extract_contact_info(processed_text)
        skills = self.extract_skills(processed_text)
        education = self.extract_education(processed_text)
        experience = self.extract_experience(processed_text)
        
        # Combine into structured CV data
        cv_data = {
            'contact_info': contact_info,
            'skills': skills,
            'education': education,
            'experience': experience,
            'raw_text': raw_text
        }
        
        return cv_data


if __name__ == "__main__":
    # Example usage
    parser = CVParser()
    sample_pdf_path = "path_to_sample_cv.pdf"
    
    if os.path.exists(sample_pdf_path):
        cv_data = parser.parse_cv(sample_pdf_path)
        print("Extracted CV Data:")
        print(f"Contact: {cv_data['contact_info']}")
        print(f"Skills: {cv_data['skills']}")
        print(f"Education: {cv_data['education']}")
        print(f"Experience: {cv_data['experience']}")
    else:
        print(f"Sample PDF not found at {sample_pdf_path}") 