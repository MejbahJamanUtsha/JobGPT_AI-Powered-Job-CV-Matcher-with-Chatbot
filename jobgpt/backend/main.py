"""
JobGPT Backend API

Main FastAPI application with endpoints for CV processing and job matching.
"""

import os
import json
import tempfile
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import services and models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ai.cv_parser import CVParser
from ai.matcher import JobMatcher

# Import database service (to be implemented)
from backend.services.db_service import MongoDBService
from backend.services.job_service import JobService
from backend.services.chatbot_service import ChatbotService
from backend.schemas.models import User, CV, Job, JobMatch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="JobGPT API",
    description="API for processing CVs and matching with job listings",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
db_service = MongoDBService(
    connection_string=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
    database_name=os.getenv("MONGODB_DB", "jobgpt")
)

job_service = JobService(db_service=db_service)
chatbot_service = ChatbotService(api_key=os.getenv("OPENAI_API_KEY"))
cv_parser = CVParser()
job_matcher = JobMatcher()

# Define API routes

@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {"message": "JobGPT API is running"}

@app.post("/users/", response_model=User)
async def create_user(user: User):
    """Create a new user."""
    return await db_service.create_user(user.dict())

@app.post("/upload-cv/", response_model=Dict[str, Any])
async def upload_cv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Upload and process a CV/resume.
    
    Args:
        file: PDF file containing the CV/resume
        user_id: Optional user ID to associate with the CV
        
    Returns:
        Parsed CV data
    """
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # Process CV
        cv_data = cv_parser.parse_cv(temp_file_path)
        
        # Store in database if user_id provided
        if user_id:
            cv_model = CV(
                user_id=user_id,
                filename=file.filename,
                skills=cv_data.get('skills', []),
                education=cv_data.get('education', []),
                experience=cv_data.get('experience', []),
                contact_info=cv_data.get('contact_info', {}),
                raw_text=cv_data.get('raw_text', '')
            )
            
            # Store in background to improve response time
            background_tasks.add_task(db_service.create_cv, cv_model.dict())
            cv_data['saved_to_db'] = True
        
        # Clean up temp file
        background_tasks.add_task(os.unlink, temp_file_path)
        
        return cv_data
        
    except Exception as e:
        logger.error(f"Error processing CV: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")

@app.get("/jobs/", response_model=List[Job])
async def get_jobs(
    skills: Optional[List[str]] = Query(None),
    location: Optional[str] = None,
    limit: int = 10
):
    """
    Get job listings, optionally filtered by skills and location.
    
    Args:
        skills: Optional list of skills to filter by
        location: Optional location to filter by
        limit: Maximum number of jobs to return
        
    Returns:
        List of job listings
    """
    return await job_service.get_jobs(skills=skills, location=location, limit=limit)

@app.post("/match-jobs/", response_model=List[Dict[str, Any]])
async def match_jobs(
    cv_data: Dict[str, Any],
    limit: int = Query(5, gt=0, le=20),
    min_score: float = Query(0.5, ge=0, le=1.0)
):
    """
    Match a CV with job listings.
    
    Args:
        cv_data: Parsed CV data
        limit: Maximum number of matches to return
        min_score: Minimum similarity score for matches
        
    Returns:
        List of job matches with similarity scores
    """
    try:
        # Get jobs from database
        jobs = await job_service.get_jobs(limit=50)  # Get more jobs than needed for better matching
        
        # Match CV with jobs
        matches = job_matcher.match_cv_with_jobs(cv_data, jobs, top_n=limit)
        
        # Filter by minimum score
        matches = [match for match in matches if match['similarity_score'] >= min_score]
        
        return matches
        
    except Exception as e:
        logger.error(f"Error matching jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Error matching jobs: {str(e)}")

@app.post("/chatbot/query/", response_model=Dict[str, Any])
async def chatbot_query(
    query: str = Form(...),
    user_id: Optional[str] = Form(None),
    cv_id: Optional[str] = Form(None),
    context: Optional[Dict[str, Any]] = None
):
    """
    Send a query to the chatbot.
    
    Args:
        query: User's query text
        user_id: Optional user ID for personalization
        cv_id: Optional CV ID for context
        context: Optional additional context
        
    Returns:
        Chatbot response
    """
    try:
        # Get user's CV data if available
        cv_data = None
        if user_id and cv_id:
            cv_data = await db_service.get_cv(cv_id=cv_id, user_id=user_id)
        
        # Process query with chatbot
        response = await chatbot_service.process_query(
            query=query,
            user_id=user_id,
            cv_data=cv_data,
            context=context
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chatbot query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/explain-match/{job_id}/", response_model=Dict[str, Any])
async def explain_match(
    job_id: str,
    cv_id: Optional[str] = None,
    cv_data: Optional[Dict[str, Any]] = None
):
    """
    Explain why a job matches a CV.
    
    Args:
        job_id: ID of the job to explain
        cv_id: Optional ID of the CV to compare with
        cv_data: Optional CV data (if not providing cv_id)
        
    Returns:
        Explanation of the match
    """
    try:
        # Get job data
        job = await job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get CV data if cv_id provided
        if cv_id and not cv_data:
            cv_data = await db_service.get_cv(cv_id=cv_id)
            if not cv_data:
                raise HTTPException(status_code=404, detail="CV not found")
        
        # Ensure we have CV data
        if not cv_data:
            raise HTTPException(status_code=400, detail="Either cv_id or cv_data must be provided")
        
        # Get explanation
        explanation = job_matcher.explain_match(cv_data, job)
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining match: {e}")
        raise HTTPException(status_code=500, detail=f"Error explaining match: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
