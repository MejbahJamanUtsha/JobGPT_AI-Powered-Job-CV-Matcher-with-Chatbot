"""
Semantic Matcher Module for JobGPT

This module handles the semantic matching between CVs and job descriptions
using sentence transformers (SBERT) for embeddings and cosine similarity
for ranking. Also includes fallback to TF-IDF if embeddings fail.
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import joblib
import shap

# Sentence Transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    has_sbert = True
except ImportError:
    has_sbert = False
    
# Fallback using scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobMatcher:
    """Match CVs with job descriptions using semantic similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = False):
        """
        Initialize the job matcher.
        
        Args:
            model_name: Name of the SBERT model to use for embeddings
            use_gpu: Whether to use GPU for embeddings
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.sbert_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Try to load SBERT model
        if has_sbert:
            try:
                self.sbert_model = SentenceTransformer(model_name)
                if use_gpu and self.sbert_model.device != "cuda":
                    self.sbert_model.to("cuda")
                logger.info(f"Loaded SBERT model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load SBERT model: {e}")
                self.sbert_model = None
        else:
            logger.warning("SentenceTransformers not available, using TF-IDF fallback")
            
    def preprocess_job(self, job: Dict[str, Any]) -> str:
        """
        Preprocess job information for matching.
        
        Args:
            job: Job dictionary containing title, description, requirements, etc.
            
        Returns:
            Preprocessed job text for embedding
        """
        # Combine relevant fields with appropriate weighting (repeated for emphasis)
        job_text = ""
        
        # Title is very important, include multiple times
        title = job.get('title', '')
        job_text += f"{title} {title} {title} "
        
        # Other fields
        job_text += job.get('description', '') + " "
        job_text += job.get('requirements', '') + " "
        job_text += job.get('qualifications', '') + " "
        job_text += job.get('responsibilities', '') + " "
        job_text += job.get('skills', '') + " "
        
        return job_text.strip()
    
    def preprocess_cv(self, cv_data: Dict[str, Any]) -> str:
        """
        Preprocess CV information for matching.
        
        Args:
            cv_data: CV dictionary containing skills, experience, education, etc.
            
        Returns:
            Preprocessed CV text for embedding
        """
        cv_text = ""
        
        # Add skills (important, repeat for emphasis)
        skills = cv_data.get('skills', [])
        skills_text = " ".join(skills)
        cv_text += f"{skills_text} {skills_text} "
        
        # Add experience information
        experiences = cv_data.get('experience', [])
        for exp in experiences:
            position = exp.get('position', '')
            company = exp.get('company', '')
            cv_text += f"{position} at {company} "
            
        # Add education information
        education = cv_data.get('education', [])
        for edu in education:
            degree = edu.get('degree', '')
            institution = edu.get('institution', '')
            cv_text += f"{degree} from {institution} "
            
        # Add raw text with lower weight
        cv_text += cv_data.get('raw_text', '')
        
        return cv_text.strip()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text using SBERT.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.sbert_model is not None:
            try:
                return self.sbert_model.encode(text, show_progress_bar=False)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
                return None
        return None
    
    def calculate_similarity_sbert(self, cv_embedding: np.ndarray, job_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between CV and job embeddings.
        
        Args:
            cv_embedding: CV embedding vector
            job_embeddings: Job embedding vectors
            
        Returns:
            Array of similarity scores
        """
        # Reshape if needed
        if len(cv_embedding.shape) == 1:
            cv_embedding = cv_embedding.reshape(1, -1)
            
        # Calculate cosine similarity
        return cosine_similarity(cv_embedding, job_embeddings)[0]
    
    def calculate_similarity_tfidf(self, cv_text: str, job_texts: List[str]) -> np.ndarray:
        """
        Calculate TF-IDF similarity between CV and job texts.
        
        Args:
            cv_text: Preprocessed CV text
            job_texts: List of preprocessed job texts
            
        Returns:
            Array of similarity scores
        """
        # Combine CV and jobs for fitting the vectorizer
        all_texts = [cv_text] + job_texts
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity
        cv_vector = tfidf_matrix[0:1]
        job_vectors = tfidf_matrix[1:]
        
        return cosine_similarity(cv_vector, job_vectors)[0]
    
    def match_cv_with_jobs(self, cv_data: Dict[str, Any], 
                          jobs: List[Dict[str, Any]],
                          top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Match a CV with a list of jobs and return top matches.
        
        Args:
            cv_data: CV dictionary containing skills, experience, education, etc.
            jobs: List of job dictionaries
            top_n: Number of top matches to return
            
        Returns:
            List of top job matches with similarity scores
        """
        # Preprocess CV and jobs
        cv_text = self.preprocess_cv(cv_data)
        job_texts = [self.preprocess_job(job) for job in jobs]
        
        # Try SBERT embeddings first
        scores = None
        using_sbert = False
        
        if self.sbert_model is not None:
            try:
                cv_embedding = self.get_embedding(cv_text)
                job_embeddings = [self.get_embedding(job_text) for job_text in job_texts]
                job_embeddings = np.vstack(job_embeddings)
                
                scores = self.calculate_similarity_sbert(cv_embedding, job_embeddings)
                using_sbert = True
                logger.info("Using SBERT embeddings for matching")
            except Exception as e:
                logger.warning(f"SBERT matching failed: {e}")
                scores = None
        
        # Fallback to TF-IDF if SBERT fails
        if scores is None:
            try:
                scores = self.calculate_similarity_tfidf(cv_text, job_texts)
                logger.info("Using TF-IDF fallback for matching")
            except Exception as e:
                logger.error(f"TF-IDF matching failed: {e}")
                return []
        
        # Sort jobs by similarity score
        job_scores = list(zip(jobs, scores))
        job_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N matches
        top_matches = []
        for i, (job, score) in enumerate(job_scores[:top_n]):
            match = job.copy()
            match['similarity_score'] = float(score)
            match['rank'] = i + 1
            match['matching_method'] = 'sbert' if using_sbert else 'tfidf'
            
            # Calculate skill overlap
            cv_skills = set(s.lower() for s in cv_data.get('skills', []))
            job_skills = set(s.lower() for s in self._extract_skills_from_job(job))
            
            common_skills = cv_skills.intersection(job_skills)
            missing_skills = job_skills - cv_skills
            
            match['skill_overlap'] = {
                'common_skills': list(common_skills),
                'missing_skills': list(missing_skills),
                'overlap_percentage': len(common_skills) / len(job_skills) if job_skills else 0
            }
            
            top_matches.append(match)
        
        return top_matches
    
    def _extract_skills_from_job(self, job: Dict[str, Any]) -> List[str]:
        """
        Extract skills from a job description.
        
        Args:
            job: Job dictionary
            
        Returns:
            List of skills from the job
        """
        # Use skills field if available
        if 'skills' in job and isinstance(job['skills'], list):
            return job['skills']
        
        # Extract from requirements if no explicit skills field
        # This is a placeholder - in a real system, this would use NER or other extraction
        requirements = job.get('requirements', '')
        if isinstance(requirements, str) and requirements:
            # Very simplified skill extraction
            skills = []
            common_tech_skills = [
                "python", "javascript", "java", "c++", "react", "angular", "vue", 
                "nodejs", "django", "flask", "docker", "kubernetes", "aws", "azure",
                "sql", "mongodb", "postgresql", "mysql", "git", "machine learning",
                "tensorflow", "pytorch", "data science", "analytics"
            ]
            
            for skill in common_tech_skills:
                if skill.lower() in requirements.lower():
                    skills.append(skill)
                    
            return skills
            
        return []
    
    def explain_match(self, cv_data: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain why a job matches a CV using SHAP values.
        
        Args:
            cv_data: CV dictionary
            job: Job dictionary
            
        Returns:
            Dictionary with explanations
        """
        # For this simplified version, we'll focus on skill overlap
        # In a full system, SHAP values would be used to explain the model's decision
        
        cv_skills = set(s.lower() for s in cv_data.get('skills', []))
        job_skills = set(s.lower() for s in self._extract_skills_from_job(job))
        
        common_skills = cv_skills.intersection(job_skills)
        missing_skills = job_skills - cv_skills
        
        explanation = {
            'job_title': job.get('title', 'Untitled Job'),
            'matching_skills': list(common_skills),
            'missing_skills': list(missing_skills),
            'skill_overlap_percentage': len(common_skills) / len(job_skills) if job_skills else 0,
            'recommendation': 'Strong Match' if len(common_skills) / len(job_skills) > 0.7 else 'Partial Match'
        }
        
        return explanation


if __name__ == "__main__":
    # Example usage
    matcher = JobMatcher()
    
    # Sample CV data
    cv_data = {
        'skills': ['Python', 'JavaScript', 'React', 'Machine Learning', 'SQL'],
        'experience': [
            {'position': 'Software Engineer', 'company': 'Tech Solutions Inc.'}
        ],
        'education': [
            {'degree': 'B.S. Computer Science', 'institution': 'University of Technology'}
        ],
        'raw_text': 'Experienced software engineer with focus on web development and data science.'
    }
    
    # Sample jobs
    jobs = [
        {
            'title': 'Senior Frontend Developer',
            'description': 'Join our team to build modern web applications.',
            'requirements': 'Experience with React, JavaScript, and modern web technologies.',
            'skills': ['JavaScript', 'React', 'HTML', 'CSS', 'Redux']
        },
        {
            'title': 'Data Scientist',
            'description': 'Analyze large datasets to derive insights.',
            'requirements': 'Proficiency in Python, Machine Learning, and SQL required.',
            'skills': ['Python', 'Machine Learning', 'SQL', 'TensorFlow', 'Data Analysis']
        }
    ]
    
    # Match CV with jobs
    matches = matcher.match_cv_with_jobs(cv_data, jobs)
    
    # Display results
    print("Top Job Matches:")
    for match in matches:
        print(f"Rank {match['rank']}: {match['title']}")
        print(f"Similarity Score: {match['similarity_score']:.4f}")
        print(f"Matching Method: {match['matching_method']}")
        print(f"Common Skills: {', '.join(match['skill_overlap']['common_skills'])}")
        print(f"Missing Skills: {', '.join(match['skill_overlap']['missing_skills'])}")
        print(f"Skill Overlap: {match['skill_overlap']['overlap_percentage']:.2%}")
        print("-" * 50) 