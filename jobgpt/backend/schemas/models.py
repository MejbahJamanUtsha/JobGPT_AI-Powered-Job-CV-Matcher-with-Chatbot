"""
Pydantic Models for JobGPT

This module contains all the Pydantic models used for data validation and serialization
throughout the JobGPT application.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, EmailStr, validator, HttpUrl
from datetime import datetime
import uuid

class User(BaseModel):
    """User model representing a job seeker or employer."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    role: str = "job_seeker"  # or "employer"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "name": "John Doe",
                "role": "job_seeker"
            }
        }

class Education(BaseModel):
    """Education entry in a CV."""
    
    degree: str
    institution: Optional[str] = None
    year: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "degree": "B.S. Computer Science",
                "institution": "University of Technology",
                "year": "2018"
            }
        }

class Experience(BaseModel):
    """Work experience entry in a CV."""
    
    position: str
    company: str
    dates: Optional[str] = None
    description: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "position": "Software Engineer",
                "company": "Tech Solutions Inc.",
                "dates": "Jan 2020 - Present",
                "description": "Developed full-stack web applications using React and Node.js."
            }
        }

class ContactInfo(BaseModel):
    """Contact information in a CV."""
    
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "email": "john.doe@example.com",
                "phone": "+1234567890",
                "linkedin": "https://linkedin.com/in/johndoe"
            }
        }

class CV(BaseModel):
    """CV model representing a parsed resume."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    filename: str
    skills: List[str] = []
    education: List[Dict[str, Any]] = []
    experience: List[Dict[str, Any]] = []
    contact_info: Dict[str, Any] = {}
    raw_text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "filename": "john_doe_resume.pdf",
                "skills": ["Python", "JavaScript", "React", "Machine Learning"],
                "education": [
                    {
                        "degree": "B.S. Computer Science",
                        "institution": "University of Technology",
                        "year": "2018"
                    }
                ],
                "experience": [
                    {
                        "position": "Software Engineer",
                        "company": "Tech Solutions Inc.",
                        "dates": "Jan 2020 - Present"
                    }
                ],
                "contact_info": {
                    "email": "john.doe@example.com",
                    "phone": "+1234567890"
                },
                "raw_text": "John Doe\nSoftware Engineer\n..."
            }
        }

class Job(BaseModel):
    """Job model representing a job listing."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    company: str
    location: Optional[str] = None
    description: str
    requirements: Optional[str] = None
    responsibilities: Optional[str] = None
    qualifications: Optional[str] = None
    skills: List[str] = []
    salary_range: Optional[str] = None
    job_type: Optional[str] = None  # full-time, part-time, contract, etc.
    remote: bool = False
    url: Optional[str] = None
    posted_at: datetime = Field(default_factory=datetime.utcnow)
    external_id: Optional[str] = None  # ID from external job API
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Senior Frontend Developer",
                "company": "Tech Innovations Inc.",
                "location": "San Francisco, CA",
                "description": "Join our team to build modern web applications...",
                "requirements": "5+ years of experience with JavaScript and React...",
                "skills": ["JavaScript", "React", "HTML", "CSS", "Redux"],
                "salary_range": "$120,000 - $150,000",
                "job_type": "full-time",
                "remote": True
            }
        }

class SkillOverlap(BaseModel):
    """Model representing skill overlap between a CV and a job."""
    
    common_skills: List[str]
    missing_skills: List[str]
    overlap_percentage: float
    
    class Config:
        schema_extra = {
            "example": {
                "common_skills": ["JavaScript", "React"],
                "missing_skills": ["Redux", "TypeScript"],
                "overlap_percentage": 0.67
            }
        }

class JobMatch(BaseModel):
    """Model representing a match between a CV and a job."""
    
    job: Job
    similarity_score: float
    rank: int
    matching_method: str  # 'sbert' or 'tfidf'
    skill_overlap: SkillOverlap
    
    class Config:
        schema_extra = {
            "example": {
                "job": {
                    "id": "job123",
                    "title": "Senior Frontend Developer",
                    "company": "Tech Innovations Inc."
                },
                "similarity_score": 0.85,
                "rank": 1,
                "matching_method": "sbert",
                "skill_overlap": {
                    "common_skills": ["JavaScript", "React"],
                    "missing_skills": ["Redux", "TypeScript"],
                    "overlap_percentage": 0.67
                }
            }
        }

class ChatbotQuery(BaseModel):
    """Model representing a query to the chatbot."""
    
    query: str
    user_id: Optional[str] = None
    cv_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Find me jobs that match my skills in Python and machine learning",
                "user_id": "user123",
                "cv_id": "cv456"
            }
        }

class ChatbotResponse(BaseModel):
    """Model representing a response from the chatbot."""
    
    text: str
    suggestions: Optional[List[str]] = None
    job_matches: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I found 3 jobs that match your skills in Python and machine learning.",
                "suggestions": ["Add Docker to your skills", "Consider roles in data science"],
                "job_matches": [
                    {
                        "title": "Data Scientist",
                        "company": "AI Solutions Inc."
                    }
                ]
            }
        }
