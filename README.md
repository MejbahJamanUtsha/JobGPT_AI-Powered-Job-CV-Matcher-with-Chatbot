# JobGPT - AI-Powered Job-CV Matcher with Chatbot

JobGPT is an intelligent platform that matches job seekers with ideal roles by analyzing their resumes/CVs using advanced NLP and machine learning models.

## Features

- **CV/Resume Analysis**: Extracts and processes information from PDFs using custom spaCy NER models
- **Semantic Matching**: Uses sentence embeddings (SBERT) to find the best job matches
- **Conversational Chatbot**: GPT-4 powered chatbot for natural language job searches
- **Skill Gap Analysis**: Visualizes differences between your skills and job requirements
- **Real-time Job Data**: Integration with job search APIs for up-to-date listings

## Tech Stack

- **Frontend**: React with Tailwind CSS, Three.js for 3D elements
- **Backend**: Python (FastAPI), NLP/ML processing 
- **Database**: MongoDB for storing CVs, job listings, and user data
- **AI Models**: spaCy NER, SBERT embeddings, GPT-4 for the chatbot
- **Deployment**: Docker containers, deployable to Render (backend) and Vercel (frontend)

## Project Structure

```
jobgpt/
├───ai/                       # Python AI Core  
│   ├───cv_parser.py          # PDF → Text  
│   ├───nlp_model/            # Custom spaCy NER  
│   └───matcher.py            # SBERT + TF-IDF  
├───backend/                  # FastAPI
│   ├───schemas/              # Pydantic models  
│   └───services/             # MongoDB queries  
├───client/                   # React + Tailwind  
│   ├───public/               # 3D assets  
│   └───src/components/       # Animated UI  
└───docs/                     # Documentation
```

## Getting Started

1. Clone this repository
2. Install backend dependencies: `pip install -r requirements.txt`
3. Install frontend dependencies: `cd client && npm install`
4. Set up MongoDB and configure connection in `.env`
5. Start the backend: `uvicorn backend.main:app --reload`
6. Start the frontend: `cd client && npm run dev`

## License

MIT
 
