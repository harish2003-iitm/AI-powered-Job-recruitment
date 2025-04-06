# TalentSphere AI

AI-powered recruitment platform that intelligently matches candidates to jobs by analyzing resumes and job descriptions. Uses advanced language models to evaluate skills, experience, and education relevance, eliminating bias and reducing screening time. Features intuitive dashboard, automated matching, and interview scheduling.

## Features

- **Job Description Analysis**: Parse and extract key requirements from job descriptions
- **CV/Resume Analysis**: Extract candidate information including skills, experience, and education
- **Smart Matching Algorithm**: Match candidates to jobs based on multiple criteria with a sophisticated scoring system
- **Interview Request Generation**: Automatically generate personalized interview requests for candidates
- **Web Interface**: User-friendly dashboard to manage the entire recruitment process

## System Architecture

The system consists of two main components:

1. **Backend API**: FastAPI-based RESTful API that handles data processing and AI-powered analysis
2. **Frontend**: React-based web interface for user interaction

## Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn
- Ollama (for local LLM)

### Backend Setup

1. Clone the repository

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with necessary configurations:
   ```
   OPENAI_API_KEY=your_api_key  # Optional, if using OpenAI
   EMAIL_HOST=smtp.example.com
   EMAIL_PORT=587
   EMAIL_USERNAME=your_username
   EMAIL_PASSWORD=your_password
   EMAIL_FROM=your_email@example.com
   ```

5. Run the FastAPI server:
   ```
   python main.py
   ```
   The API will be available at http://127.0.0.1:8000

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```
   The web interface will be available at http://localhost:3000

## Usage

1. **Upload Job Descriptions**:
   - Go to the "Upload Job" page
   - Upload a CSV file containing job descriptions
   - The system will automatically extract requirements

2. **Upload Candidate CVs**:
   - Go to the "Upload Candidate" page
   - Upload PDF files of candidate CVs
   - The system will automatically extract candidate information

3. **View Matches**:
   - Go to the "Matches" page to see how candidates match with job positions
   - Filter by match score to find the best candidates
   - Review detailed match breakdown by skills, experience, and education

4. **Dashboard**:
   - View overall statistics and recruitment insights

## Quick Start

For Windows users:
```
setup.bat
start.bat
```

For Unix/Linux users:
```
chmod +x setup.sh
./setup.sh
chmod +x start.sh
./start.sh
```

## API Endpoints

- `POST /upload/job-descriptions`: Upload job description files
- `POST /upload/candidates`: Upload candidate CV files
- `GET /stats`: Get system statistics for the dashboard
- `GET /jobs`: Get list of all jobs
- `GET /candidates`: Get list of all candidates
- `GET /matches`: Get all matches with option to filter by minimum score

## Technology Stack

- **Backend**: 
  - FastAPI
  - SQLite
  - Langchain
  - LLM (Ollama/OpenAI)
  
- **Frontend**:
  - React
  - Material UI
  - React Router
  - Recharts

## Development

To run the system in development mode:

1. Start the backend:
   ```
   python main.py
   ```

2. In a separate terminal, start the frontend:
   ```
   cd frontend
   npm start
   ```

## License

[MIT License](LICENSE) 