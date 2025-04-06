from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List, Dict, Any, Optional
import json
import sqlite3
from datetime import datetime
from database import Database
from agents import JobDescriptionAgent, CandidateAgent, MatchingAgent
from recruitment_graph import app as recruitment_graph_app, GraphState
from email_service import EmailService, send_email
import chardet
import time
import traceback
import re
from langchain.schema import HumanMessage

def get_education_level_name(level_value, education_levels):
    """Helper function to get the name of an education level from its value"""
    try:
        # Try to find an exact match
        for name, value in education_levels.items():
            if value == level_value:
                return name
        
        # If no exact match, return the closest match
        level_value = int(level_value)
        closest_match = "Unknown"
        closest_diff = float('inf')
        
        for name, value in education_levels.items():
            diff = abs(value - level_value)
            if diff < closest_diff:
                closest_diff = diff
                closest_match = name
                
        return closest_match
    except (ValueError, TypeError):
        # If conversion to int fails, return Unknown
        return "Unknown"

app = FastAPI(title="AI-Powered Recruitment System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
job_agent = JobDescriptionAgent()
candidate_agent = CandidateAgent()
matching_agent = MatchingAgent()
email_service = EmailService()

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db_connection(max_retries=5, retry_delay=0.3):
    """
    Get a SQLite database connection with retry logic to handle 'database is locked' errors.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        SQLite connection object
    """
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('recruitment.db', timeout=20.0)
            return conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                print(f"Database locked, retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise
                
    # If we get here, all retries failed
    raise sqlite3.OperationalError("Could not connect to database after multiple attempts")

def init_db():
    """Initialize database"""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create Job Descriptions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            requirements_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add requirements_json column if it doesn't exist
    try:
        c.execute("PRAGMA table_info(jobs)")
        columns = [col[1] for col in c.fetchall()]
        if 'requirements_json' not in columns:
            c.execute('ALTER TABLE jobs ADD COLUMN requirements_json TEXT')
            print("Added requirements_json column to jobs table")
    except Exception as e:
        print(f"Error checking or adding column: {e}")
    
    # Create Candidates table
    c.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            education TEXT,
            experience TEXT,
            skills TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create Match Scores table
    c.execute('''
        CREATE TABLE IF NOT EXISTS match_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            overall_score REAL NOT NULL,
            skills_match REAL DEFAULT 0,
            experience_match REAL DEFAULT 0,
            education_match REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if interview_requests table exists and has job_id column
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interview_requests'")
    if not c.fetchone():
        # Create Interview Requests table with job_id column
        c.execute('''
            CREATE TABLE IF NOT EXISTS interview_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                proposed_dates TEXT NOT NULL,
                interview_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (candidate_id) REFERENCES candidates (id),
                FOREIGN KEY (job_id) REFERENCES jobs (id)
            )
        ''')
        print("Created interview_requests table with job_id column")
    else:
        # Check if job_id column exists in interview_requests table
        c.execute("PRAGMA table_info(interview_requests)")
        columns = [col[1] for col in c.fetchall()]
        if 'job_id' not in columns:
            try:
                # Add job_id column if it doesn't exist
                c.execute('ALTER TABLE interview_requests ADD COLUMN job_id INTEGER NOT NULL DEFAULT 0')
                print("Added job_id column to interview_requests table")
            except Exception as e:
                print(f"Error adding job_id column to interview_requests table: {e}")
                # If the above fails, recreate the table with the correct schema
                c.execute('DROP TABLE interview_requests')
                c.execute('''
                    CREATE TABLE interview_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        candidate_id INTEGER NOT NULL,
                        job_id INTEGER NOT NULL,
                        subject TEXT NOT NULL,
                        body TEXT NOT NULL,
                        proposed_dates TEXT NOT NULL,
                        interview_type TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                print("Recreated interview_requests table with job_id column")
    
    conn.commit()
    conn.close()
    
    print("Database initialized")

# Initialize database
init_db()

def detect_encoding(file_path: str) -> str:
    """Detect the encoding of a file"""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

@app.post("/clean-database")
async def clean_database():
    """Clean database by removing duplicate job entries"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get job count before cleaning
        c.execute('SELECT COUNT(*) FROM jobs')
        before_count = c.fetchone()[0]
        
        # Find all duplicates
        c.execute('''
            SELECT title, COUNT(*) as count
            FROM jobs
            GROUP BY title
            HAVING count > 1
        ''')
        duplicates = c.fetchall()
        
        total_removed = 0
        for title, count in duplicates:
            # Keep the newest entry for each duplicate title (highest ID)
            c.execute('''
                DELETE FROM jobs
                WHERE title = ? AND id NOT IN (
                    SELECT MAX(id)
                    FROM jobs
                    WHERE title = ?
                )
            ''', (title, title))
            removed = count - 1
            total_removed += removed
            print(f"Removed {removed} duplicate entries for '{title}'")
        
        conn.commit()
        
        # Get job count after cleaning
        c.execute('SELECT COUNT(*) FROM jobs')
        after_count = c.fetchone()[0]
        
        conn.close()
        
        return {
            "message": f"Database cleaned. Removed {total_removed} duplicate job entries.",
            "before_count": before_count,
            "after_count": after_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/job-descriptions")
async def upload_job_descriptions(files: List[UploadFile] = File(...)):
    """Upload job description files"""
    try:
        processed_jobs = []
        skipped_jobs = []
        for file in files:
            # Save file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Process job descriptions
            jobs = job_agent.parse_jd_csv(file_path)
            
            # Store in database
            conn = get_db_connection()
            c = conn.cursor()
            
            for job in jobs:
                # Check if job with same title already exists
                c.execute('SELECT id FROM jobs WHERE title = ?', (job['title'],))
                existing_job = c.fetchone()
                
                if existing_job:
                    # Skip this job
                    skipped_jobs.append(job['title'])
                    print(f"Skipping duplicate job: {job['title']}")
                    continue
                
                # Extract requirements before saving
                requirements = job_agent.extract_key_requirements(job['description'])
                requirements_json = requirements.get('requirements_json', json.dumps(requirements))
                
                # Insert into jobs table with requirements_json
                c.execute('''
                    INSERT INTO jobs (title, description, requirements_json)
                    VALUES (?, ?, ?)
                ''', (job['title'], job['description'], requirements_json))
                
                job_id = c.lastrowid
                processed_jobs.append(job_id)
            
            conn.commit()
            conn.close()
        
        return {
            "message": f"Successfully processed {len(processed_jobs)} job descriptions. Skipped {len(skipped_jobs)} duplicates.",
            "job_ids": processed_jobs,
            "skipped": skipped_jobs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/candidates")
async def upload_candidates(files: List[UploadFile] = File(...)):
    """Upload candidate CV files"""
    try:
        processed_candidates = []
        shortlisted_candidates = []
        emails_sent = []
        
        # Update match threshold from 40% to 65%
        MATCH_THRESHOLD = 80  # Changed from 80% to 65% (above 60% as requested)
        
        # Define education levels for qualification scoring
        education_levels = {
            'High School': 1,
            'Associate': 2,
            'Bachelor': 3, 
            'Master': 4,
            'PhD': 5,
            'Doctorate': 5
        }
        
        # Update weights to be equal (approximately 1/3 each)
        weights = {
            'skills': 0.34,        # Changed from 0.5 to 0.34 (~1/3)
            'experience': 0.33,    # Changed from 0.3 to 0.33 (~1/3)
            'qualification': 0.33, # Changed from 0.2 to 0.33 (~1/3)
            'bonus': 0.0
        }
        
        # First clean up the database before processing
        conn = get_db_connection()
        c = conn.cursor()
        
        # Fix interview_requests table if needed
        try:
            # Check if interview_requests table exists and has job_id column
            c.execute("PRAGMA table_info(interview_requests)")
            columns = [col[1] for col in c.fetchall()]
            
            if 'job_id' not in columns:
                # Table exists but doesn't have job_id column
                print("interview_requests table is missing job_id column, fixing...")
                try:
                    # Try to add the column
                    c.execute('ALTER TABLE interview_requests ADD COLUMN job_id INTEGER NOT NULL DEFAULT 0')
                    print("Added job_id column to interview_requests table")
                except Exception as e:
                    # If we can't add column, drop and recreate the table
                    print(f"Error adding column: {e}, recreating table")
                    c.execute('DROP TABLE interview_requests')
                    c.execute('''
                        CREATE TABLE interview_requests (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            candidate_id INTEGER NOT NULL,
                            job_id INTEGER NOT NULL,
                            subject TEXT NOT NULL,
                            body TEXT NOT NULL,
                            proposed_dates TEXT NOT NULL,
                            interview_type TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    print("Recreated interview_requests table with job_id column")
            else:
                print("interview_requests table has job_id column")
        except Exception as e:
            # Table probably doesn't exist, create it
            print(f"Error checking interview_requests table: {e}, creating table")
            c.execute('''
                CREATE TABLE IF NOT EXISTS interview_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    job_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    proposed_dates TEXT NOT NULL,
                    interview_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            print("Created interview_requests table with job_id column")
        
        conn.commit()
        
        # Debug: Check how many rows are in the jobs table
        c.execute('SELECT COUNT(*) FROM jobs')
        job_count = c.fetchone()[0]
        print(f"Total rows in jobs table before cleanup: {job_count}")
        
        # Clean up duplicate jobs
        print("Cleaning up duplicate job entries...")
        
        # Get duplicates
        c.execute('''
            SELECT title, COUNT(*) as count
            FROM jobs
            GROUP BY title
            HAVING count > 1
        ''')
        duplicates = c.fetchall()
        
        total_removed = 0
        for title, count in duplicates:
            # Keep the newest entry for each duplicate title (highest ID)
            c.execute('''
                DELETE FROM jobs
                WHERE title = ? AND id NOT IN (
                    SELECT MAX(id)
                    FROM jobs
                    WHERE title = ?
                )
            ''', (title, title))
            removed = count - 1
            total_removed += removed
            print(f"Removed {removed} duplicate entries for '{title}'")
        
        conn.commit()
        
        # Get job count after cleaning
        c.execute('SELECT COUNT(*) FROM jobs')
        after_job_count = c.fetchone()[0]
        print(f"Total rows in jobs table after cleanup: {after_job_count} (removed {total_removed} duplicates)")
        
        # Fetch all jobs once at the beginning
        c.execute('SELECT id, title, description, requirements_json FROM jobs')
        all_jobs = c.fetchall()
        conn.close()
        
        # Step 1: Group jobs by title and extract requirements just once per unique job type
        job_requirements_by_title = {}
        job_details = []
        
        # First count unique job titles
        unique_job_titles = set()
        for _, job_title, _, _ in all_jobs:
            unique_job_titles.add(job_title)
            
        print(f"Processing {len(unique_job_titles)} unique job types...")
        
        for job_id, job_title, job_desc, requirements_json in all_jobs:
            if job_title not in job_requirements_by_title:
                # Process this job type only once
                print(f"Extracting requirements for job type: {job_title}")
                
                # Use stored requirements if available, otherwise extract them
                if requirements_json and requirements_json.strip():
                    try:
                        requirements = json.loads(requirements_json)
                        print(f"Using pre-stored requirements for {job_title}")
                    except:
                        print(f"Failed to parse stored requirements, extracting fresh for {job_title}")
                        requirements = job_agent.extract_key_requirements(job_desc)
                else:
                    requirements = job_agent.extract_key_requirements(job_desc)
                
                # Store requirements for this job title
                job_requirements_by_title[job_title] = requirements
            
            # Keep track of all job details
            job_details.append({
                'id': job_id,
                'title': job_title,
                'requirements': job_requirements_by_title[job_title]
            })
        
        print(f"Finished processing {len(job_requirements_by_title)} unique job types. Moving to candidate processing...")
        
        # Step 2: Process each candidate CV
        for file in files:
            # Save file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            print(f"Processing candidate CV: {file.filename}")
            # Process CV
            cv_text = candidate_agent.parse_cv(file_path)
            candidate_info = candidate_agent.extract_candidate_info(cv_text)

            # --- HANDLE CANDIDATE NAME (AGENT OUTPUT OR FILENAME FALLBACK) ---
            candidate_name = candidate_info.get('name')
            # Check if the name is missing, empty, or a generic placeholder
            if not candidate_name or candidate_name.strip() == "" or candidate_name.lower() == "unknown":
                candidate_name = os.path.splitext(file.filename)[0] # Use filename (without extension) as fallback
                print(f"Warning: CandidateAgent failed to extract name from {file.filename}. Using filename '{candidate_name}' as identifier.")
                # Optionally update candidate_info dict if needed downstream, though using the variable is cleaner
                # candidate_info['name'] = candidate_name 
            # --- END NAME HANDLING ---

            # Store candidate in database
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""
                INSERT INTO candidates (name, email, phone, education, experience, skills)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                candidate_name, # Use the determined name
                candidate_info.get('email', ''), # Use .get for safety
                candidate_info.get('phone', ''),
                json.dumps(candidate_info.get('education', [])),
                json.dumps(candidate_info.get('experience', [])),
                json.dumps(candidate_info.get('skills', []))
            ))
            candidate_id = c.lastrowid
            processed_candidates.append(candidate_id)
            
            # Step 3: Match candidate against all jobs (using pre-processed requirements)
            print(f"Matching candidate {candidate_name} against all jobs...") # Use determined name
            match_results = []
            
            for job in job_details:
                job_id = job['id']
                job_title = job['title']
                requirements = job['requirements']
                
                # Extract requirements for scoring
                required_skills = requirements.get('required_skills', [])
                preferred_skills = requirements.get('preferred_skills', [])
                required_experience = requirements.get('experience_years', None) # Get value from agent, default to None if missing
                required_education = requirements.get('education_level', 'Bachelor')  # Default to Bachelor's if not specified

                # --- PARSE EXPERIENCE REQUIREMENT FROM AGENT OUTPUT (NO HARDCODED DEFAULTS) ---
                required_experience_num = None # Initialize as None

                if isinstance(required_experience, (int, float)):
                    required_experience_num = float(required_experience)
                    print(f"Numeric experience requirement: {required_experience_num}")
                elif isinstance(required_experience, str):
                    # Check if the string is empty or just whitespace
                    if not required_experience or required_experience.strip() == '':
                        print(f"Empty string for experience requirement, defaulting to 0 years")
                        required_experience_num = 0.0
                    else:
                        match = re.search(r'(\d+(?:\.\d+)?)', required_experience)
                        if match:
                            try:
                                required_experience_num = float(match.group(1))
                            except ValueError:
                                print(f"Warning: Agent provided experience ('{required_experience}') - could not parse extracted number. Experience requirement unset.")
                                required_experience_num = None # Parsing failed
                        else:
                            # If no number found in string, set to 0 rather than None
                            print(f"No numeric value found in experience requirement: '{required_experience}', defaulting to 0 years")
                            required_experience_num = 0.0
                elif required_experience is None:
                    # If None is explicitly provided, default to 0 years
                    print(f"None value for experience requirement, defaulting to 0 years")
                    required_experience_num = 0.0
                else:
                    print(f"Unexpected experience requirement type: {type(required_experience)}, defaulting to 0 years")
                    required_experience_num = 0.0

                # Ensure non-negative if a number was successfully parsed
                if required_experience_num is not None:
                     required_experience_num = max(0.0, required_experience_num)
                     print(f"Final experience requirement: {required_experience_num} years")
                else:
                     print(f"Failed to parse experience requirement, will default to 0 during scoring")
                # --- END PARSING ---

                # ADVANCED MATCHING ALGORITHM
                
                # 1. Skills match score (using LLM for semantic matching)
                candidate_skills = candidate_info.get('skills', [])
                skills_matched = 0
                preferred_matched = 0
                skills_score = 0.0 # Initialize skills score
                matched_required = [] # Initialize list for matched skills log
                matched_preferred = [] # Initialize list for matched skills log

                # Prepare skills lists (even if empty)
                candidate_skills_processed = []
                for skill in candidate_skills:
                    if isinstance(skill, dict) and 'name' in skill:
                        candidate_skills_processed.append(skill['name'])
                    elif isinstance(skill, str):
                        candidate_skills_processed.append(skill)
                
                # Ensure LLM skill evaluation always runs
                try:
                    # Prepare data for LLM prompt (handles empty lists)
                    candidate_skills_text = ", ".join(candidate_skills_processed) if candidate_skills_processed else "None"
                    required_skills_text = ", ".join(required_skills) if required_skills else "None"
                    preferred_skills_text = ", ".join(preferred_skills) if preferred_skills else "None"

                    # Create prompt for LLM skill evaluation
                    skill_match_prompt = f"""
                    Evaluate the semantic match between the candidate's skills and the job requirements.

                    JOB TITLE: {job_title}
                    REQUIRED SKILLS: {required_skills_text}
                    PREFERRED SKILLS: {preferred_skills_text}

                    CANDIDATE SKILLS: {candidate_skills_text}

                    Analyze:
                    1. Direct matches (e.g., Python == Python).
                    2. Semantic matches (e.g., 'Penetration Testing' matches required 'Cybersecurity').
                    3. Transferable skills relevant to the {job_title} role.

                    Return ONLY JSON with:
                    {{
                        "matched_required_skills": [list of required skills found in candidate skills],
                        "matched_preferred_skills": [list of preferred skills found in candidate skills],
                        "overall_skill_score": 0.0 to 1.0 (based on required skills match, bonus for preferred),
                        "reasoning": "Brief explanation of the match."
                    }}
                    """
                    
                    # Call LLM for skill matching analysis - USE candidate_agent.llm
                    # Use abatch with a list containing the raw prompt string
                    skill_response_list = await candidate_agent.llm.abatch([skill_match_prompt])
                    # Extract content from AIMessage object
                    skill_response_text = skill_response_list[0].content if skill_response_list else ""
                    
                    # Initialize reasoning for the inner try/except
                    reasoning = ''

                    try:
                        # Parse JSON response
                        json_match = re.search(r'```(?:json)?(.*)```', skill_response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1).strip()
                        else:
                            json_str = skill_response_text.strip()
                            
                        match_data = json.loads(json_str)
                        
                        matched_required = match_data.get('matched_required_skills', [])
                        matched_preferred = match_data.get('matched_preferred_skills', [])
                        llm_skill_score = float(match_data.get('overall_skill_score', 0.0))
                        reasoning = match_data.get('reasoning', '')
                        
                        # Use LLM's assessment directly
                        skills_score = llm_skill_score
                        skills_matched = len(matched_required) # For logging
                        preferred_matched = len(matched_preferred) # For logging
                        
                        print(f"LLM Skills Score: {skills_score:.2f} - Matched Required: {skills_matched}, Matched Preferred: {preferred_matched}")
                        print(f"  Matched Required Skills: {matched_required}")
                        print(f"  Matched Preferred Skills: {matched_preferred}")
                        print(f"LLM Reasoning: {reasoning}")
                        
                    except json.JSONDecodeError:
                        print(f"Failed to parse LLM skill match JSON: {skill_response_text}. Falling back to basic matching.")
                        # Fallback calculation
                        candidate_skills_lower = {s.lower() for s in candidate_skills_processed if isinstance(s, str)}
                        required_skills_lower = [skill.lower() for skill in required_skills if isinstance(skill, str)]
                        preferred_skills_lower = [skill.lower() for skill in preferred_skills if isinstance(skill, str)]
                        skills_matched = sum(1 for skill in required_skills_lower if skill in candidate_skills_lower or any(skill in cs for cs in candidate_skills_lower))
                        preferred_matched = sum(1 for skill in preferred_skills_lower if skill in candidate_skills_lower or any(skill in cs for cs in candidate_skills_lower))
                        required_count = len(required_skills) if required_skills else 0 # Use 0 if list is empty
                        preferred_count = len(preferred_skills) if preferred_skills else 0 # Use 0 if list is empty
                        if required_count + preferred_count > 0:
                            skills_score = (skills_matched + 0.5 * preferred_matched) / (required_count + 0.5 * preferred_count)
                        else:
                            skills_score = 0.0 # If no skills required/preferred, fallback score is 0.0
                        print(f"Fallback Skills Score: {skills_score:.2f}")

                except Exception as e:
                    print(f"Error during LLM skill matching: {e}. Falling back to basic matching.")
                    # Fallback calculation (same as above)
                    candidate_skills_lower = {s.lower() for s in candidate_skills_processed if isinstance(s, str)}
                    required_skills_lower = [skill.lower() for skill in required_skills if isinstance(skill, str)]
                    preferred_skills_lower = [skill.lower() for skill in preferred_skills if isinstance(skill, str)]
                    skills_matched = sum(1 for skill in required_skills_lower if skill in candidate_skills_lower or any(skill in cs for cs in candidate_skills_lower))
                    preferred_matched = sum(1 for skill in preferred_skills_lower if skill in candidate_skills_lower or any(skill in cs for cs in candidate_skills_lower))
                    required_count = len(required_skills) if required_skills else 0 # Use 0 if list is empty
                    preferred_count = len(preferred_skills) if preferred_skills else 0 # Use 0 if list is empty
                    if required_count + preferred_count > 0:
                        skills_score = (skills_matched + 0.5 * preferred_matched) / (required_count + 0.5 * preferred_count)
                    else:
                        skills_score = 0.0 # If no skills required/preferred, fallback score is 0.0
                    print(f"Fallback Skills Score: {skills_score:.2f}")
                
                skills_score = min(1.0, skills_score)  # Cap at 1.0

                # 2. Experience match score (Considering position relevance)
                candidate_experience = candidate_info.get('experience', [])
                relevant_experience_years = 0.0  # Initialize relevant years
                relevant_positions_details = [] # Store details of relevant positions
                
                # Use LLM to evaluate position relevance for experience entries
                for exp in candidate_experience:
                    if isinstance(exp, dict):
                        # Extract position information
                        exp_title = exp.get('title', '')
                        exp_company = exp.get('company', '')
                        exp_description = exp.get('description', '')
                        exp_years = 0.0
                        
                        # First check if the experience entry has calculated years
                        if 'years' in exp and exp.get('years') is not None:
                            try:
                                if isinstance(exp['years'], (int, float)):
                                    exp_years = float(exp['years'])
                                elif isinstance(exp['years'], str) and exp['years'].strip():
                                    exp_years = float(exp['years'])
                            except (ValueError, TypeError):
                                pass
                        
                        # If no pre-calculated years, calculate from start/end dates
                        if exp_years == 0.0 and 'start_date' in exp and 'end_date' in exp:
                            start_date = exp.get('start_date', '')
                            end_date = exp.get('end_date', '')
                            
                            # Use LLM to calculate years between dates
                            date_prompt = f"""
                            Calculate the number of years between these two dates:
                            Start date: {start_date}
                            End date: {end_date if end_date.lower() != 'present' else datetime.now().strftime('%Y-%m')}
                            
                            If end date is 'Present' or similar, use current date.
                            Return ONLY the numeric value representing total years (e.g., 4.5).
                            """
                            
                            try:
                                # Call LLM to calculate years from dates - USE candidate_agent.llm
                                # Use abatch with a list containing the raw prompt string
                                years_response_list = await candidate_agent.llm.abatch([date_prompt])
                                # Extract content from AIMessage object
                                content = years_response_list[0].content if years_response_list else ""
                                years_match = re.search(r'(\d+(?:\.\d+)?)', content)
                                if years_match:
                                    exp_years = float(years_match.group(1))
                                    print(f"LLM calculated {exp_years} years from {start_date} to {end_date}")
                            except Exception as e:
                                print(f"Error calculating years from dates with LLM: {e}")
                        
                        # If still no years, try to parse from duration field
                        if exp_years == 0.0 and 'duration' in exp:
                            duration_str = exp.get('duration', '')
                            
                            # Use LLM to extract years from duration text
                            duration_prompt = f"""
                            Extract the total number of years from this duration text: "{duration_str}"
                            
                            Examples:
                            - "2019-2023" → 4
                            - "Jan 2020 - March 2023" → 3.2
                            - "2 years 3 months" → 2.25
                            
                            Return ONLY the numeric value of years (e.g., 4.5).
                            """
                            
                            try:
                                # Call LLM to extract years from duration - USE candidate_agent.llm
                                # Use abatch with a list containing the raw prompt string
                                duration_response_list = await candidate_agent.llm.abatch([duration_prompt])
                                # Extract content from AIMessage object
                                content = duration_response_list[0].content if duration_response_list else ""
                                duration_match = re.search(r'(\d+(?:\.\d+)?)', content)
                                if duration_match:
                                    exp_years = float(duration_match.group(1))
                                    print(f"LLM extracted {exp_years} years from duration: {duration_str}")
                            except Exception as e:
                                print(f"Error extracting years from duration with LLM: {e}")
                        
                        # Use LLM to check position relevance if years > 0
                        if exp_years > 0:
                            # Create prompt for LLM to determine position relevance
                            position_prompt = f"""
                            Evaluate if this experience position is relevant to the job requirements.

                            JOB REQUIREMENTS:
                            - Job Title: {job_title}
                            - Required Skills: {', '.join(required_skills) if required_skills else 'Not specified'}

                            CANDIDATE EXPERIENCE:
                            - Position Title: {exp_title}
                            - Company: {exp_company}
                            - Description: {exp_description}

                            Consider ALL possible relevance factors:
                            1. Direct job title similarities
                            2. Overlapping or transferable skills between positions
                            3. Related industry/domain experience
                            4. Underlying technologies or methodologies
                            5. Relevant responsibilities regardless of job title

                            Please evaluate generously - focus on actual relevance of experience rather than exact title matches.
                            For example, a Software Engineer position could be relevant to Cybersecurity Analyst if they worked on security features.

                            Return JSON with:
                            {{  # Escaped literal brace
                                "is_relevant": true/false,
                                "relevance_score": 0.0 to 1.0 (decimal representing relevance),
                                "reasoning": "Brief explanation of relevance factors"
                            }}  # Escaped literal brace
                            """
                            
                            try:
                                # Call LLM to evaluate position relevance - USE candidate_agent.llm
                                # Use abatch with a list containing the raw prompt string
                                relevance_response_list = await candidate_agent.llm.abatch([position_prompt])
                                # Extract content from AIMessage object
                                relevance_response_text = relevance_response_list[0].content if relevance_response_list else ""

                                try:
                                    # Parse JSON response
                                    # Attempt to find JSON within potential markdown code blocks
                                    json_match = re.search(r'```(?:json)?(.*)```', relevance_response_text, re.DOTALL) # Use _text variable
                                    if json_match:
                                        json_str = json_match.group(1).strip()
                                    else:
                                        json_str = relevance_response_text.strip() # Use _text variable

                                    relevance_data = json.loads(json_str)
                                    is_relevant = relevance_data.get('is_relevant', False)
                                    relevance_score = float(relevance_data.get('relevance_score', 0.0))
                                    reasoning = relevance_data.get('reasoning', 'No explanation provided')

                                    if is_relevant and relevance_score > 0:
                                        weighted_exp_years = exp_years * relevance_score
                                        relevant_experience_years += weighted_exp_years
                                        relevant_positions_details.append(f"{exp_title} (Score: {relevance_score:.2f}, Yrs: {weighted_exp_years:.2f})")
                                        print(f"Position '{exp_title}' relevance: {relevance_score:.2f} - Added {weighted_exp_years:.2f} years - {reasoning}")
                                    else:
                                        print(f"Position '{exp_title}' not relevant - {reasoning}")
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, log it and treat as NOT relevant
                                    print(f"Failed to parse JSON response for position relevance: {relevance_response_text}") # Log _text variable
                                    print(f"Position '{exp_title}' treated as not relevant due to parsing error.")
                                    # Do NOT add any years here
                                    
                            except Exception as e:
                                print(f"Error evaluating position relevance for '{exp_title}': {e}")
                                # If the LLM call itself fails, treat as NOT relevant
                                print(f"LLM evaluation failed - Treating position '{exp_title}' as not relevant. No years added.")
                                # Ensure NO years are added if the LLM fails
                                # REMOVED: relevant_experience_years += exp_years 
                
                # Calculate experience score based on revised logic
                experience_score = 0.0
                required_experience_num_calc = 0.0 # Initialize calculated requirement

                if relevant_experience_years == 0:
                    # If candidate has no relevant experience, score is 0
                    experience_score = 0.0
                    print(f"Candidate has no relevant experience. Experience score: 0.0")
                else:
                    # Candidate has some relevant experience
                    if required_experience_num is None:
                        # If requirement couldn't be parsed, treat as 0 years required
                        print(f"Warning: Could not determine numerical experience requirement from agent output ('{required_experience}'). Treating as entry-level (0 years required).")
                        required_experience_num_calc = 0.0
                    else:
                         # Ensure required_experience_num is a non-negative float
                         try:
                              req_exp_float = float(required_experience_num)
                              required_experience_num_calc = max(0.0, req_exp_float)
                         except (ValueError, TypeError):
                              print(f"Warning: Could not convert required experience '{required_experience_num}' to float. Treating as 0 years required.")
                              required_experience_num_calc = 0.0

                    if required_experience_num_calc > 0:
                        # Job requires experience, calculate ratio
                        experience_score = min(1.0, relevant_experience_years / required_experience_num_calc)
                        print(f"Experience score calculated: {experience_score:.2f} (Candidate has {relevant_experience_years:.1f} relevant yrs, Job requires {required_experience_num_calc} yrs)")
                    else:
                        # Job requires 0 years (entry-level) or requirement was unparsable
                        # Since candidate has relevant experience (>0), they meet/exceed the requirement
                        experience_score = 1.0
                        print(f"Entry-level position ({required_experience_num_calc} yrs required), candidate has {relevant_experience_years:.1f} relevant yrs. Experience score: 1.0")

                # 3. Education/qualification match score
                # Ensure required_education is a string and convert to a level
                if not isinstance(required_education, str):
                    required_education = str(required_education)
                    
                # Get the required level with a default of 3 (Bachelor's level)
                required_level = 3  # Default to Bachelor's level
                for key, value in education_levels.items():
                    if isinstance(key, str) and key.lower() in required_education.lower():
                        required_level = value
                        print(f"Found education requirement match: '{key}' in '{required_education}', level={value}")
                        break
                
                candidate_education = candidate_info.get('education', [])
                highest_education_level = 0 # Use a different variable name
                
                print(f"Candidate education entries: {candidate_education}")
                for edu in candidate_education:
                    if not isinstance(edu, dict):
                        continue
                        
                    degree = edu.get('degree', '')
                    if not isinstance(degree, str):
                        continue
                    
                    print(f"Checking education entry: {degree}")
                    # Use LLM to determine the education level instead of hardcoded mapping
                    edu_level_prompt = f"""
                    Determine the education level for this degree: "{degree}"
                    
                    Education levels are:
                    1 = High School
                    2 = Associate's
                    3 = Bachelor's
                    4 = Master's
                    5 = PhD/Doctorate
                    
                    Return only a single number representing the level (1-5).
                    """
                    
                    try:
                        # Call LLM to determine education level
                        # Use abatch with a list containing the raw prompt string
                        level_response_list = await candidate_agent.llm.abatch([edu_level_prompt])
                        # Extract content from AIMessage object
                        content = level_response_list[0].content if level_response_list else ""
                        level_match = re.search(r'(\d+)', content) # Search in the extracted content
                        if level_match:
                            level = int(level_match.group(1))
                            print(f"LLM determined education level for '{degree}': {level}")
                        else:
                            # Fallback to keyword matching
                            level = 0
                            for key, value in education_levels.items():
                                if isinstance(key, str) and key.lower() in degree.lower():
                                    level = value
                                    print(f"Fallback: Found education match: '{key}' in '{degree}', level={value}")
                                    break
                    except Exception as e:
                        print(f"Error determining education level with LLM: {e}")
                        # Fallback to keyword matching
                        level = 0
                        for key, value in education_levels.items():
                            if isinstance(key, str) and key.lower() in degree.lower():
                                level = value
                                print(f"Fallback: Found education match: '{key}' in '{degree}', level={value}")
                                break
                
                    highest_education_level = max(highest_education_level, level)
                
                print(f"Final education levels - Required: {required_level}, Candidate highest: {highest_education_level}")
                
                # Compare education levels numerically
                try:
                    # Ensure both are integers before comparison
                    candidate_level_int = int(highest_education_level)
                    required_level_int = int(required_level)
                    
                    if candidate_level_int >= required_level_int:
                        qualification_score = 1.0
                        print(f"Education qualification match: Candidate ({candidate_level_int}) >= Required ({required_level_int})")
                    else:
                        qualification_score = candidate_level_int / required_level_int if required_level_int > 0 else 0.0
                        print(f"Partial education qualification match: {qualification_score:.2f} (Candidate {candidate_level_int} < Required {required_level_int})")
                except (ValueError, TypeError):
                    print(f"Warning: Could not compare education levels numerically: Cand={highest_education_level}, Req={required_level}")
                    # Fallback: Check if strings match if numeric comparison fails
                    if str(highest_education_level) == str(required_level):
                        qualification_score = 1.0
                    else:
                        qualification_score = 0.0 # Assign 0 if types are incompatible
                
                # 4. Optional: Certifications or achievements bonus (small impact)
                bonus_score = 0.0
                # Could check for certifications or achievements here
                
                # Combine scores with weights
                overall_score = (
                    weights['skills'] * skills_score +
                    weights['experience'] * experience_score +
                    weights['qualification'] * qualification_score +
                    weights['bonus'] * bonus_score
                )
                
                # Convert to percentage (0-100)
                overall_score_pct = round(overall_score * 100)
                
                # --- DETAILED LOGGING --- 
                print(f"--- Match Score Details for Candidate {candidate_id} vs Job {job_id} ({job_title}) ---")
                print(f"  Skills Score (Weight {weights['skills']:.1f}): {skills_score*100:.1f}% (Matched: {skills_matched}/{len(required_skills)}, Preferred: {preferred_matched}/{len(preferred_skills)})")
                print(f"  Experience Score (Weight {weights['experience']:.1f}): {experience_score*100:.1f}% (Candidate Relevant: {relevant_experience_years:.1f} yrs, Required: {required_experience_num_calc} yrs)")
                # Add experience breakdown log
                if relevant_positions_details:
                    print(f"    Relevant Positions: [{', '.join(relevant_positions_details)}]")
                print(f"  Qualification Score (Weight {weights['qualification']:.1f}): {qualification_score*100:.1f}% (Candidate Lvl: {highest_education_level}, Required Lvl: {required_level})")
                print(f"  Bonus Score (Weight {weights['bonus']:.1f}): {bonus_score*100:.1f}%")
                print(f"  Overall Score: {overall_score_pct}%")
                print("-------------------------------------------------------------")
                # --- END DETAILED LOGGING --- 
                
                # Create match score object
                match_score = {
                    'overall_score': overall_score_pct,
                    'skills_match': round(skills_score * 100),
                    'experience_match': round(experience_score * 100),
                    'education_match': round(qualification_score * 100),
                    'threshold_met': overall_score_pct >= MATCH_THRESHOLD
                }
                
                # Store match in database
                c.execute('''
                    INSERT INTO match_scores (job_id, candidate_id, overall_score, skills_match, experience_match, education_match)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (job_id, candidate_id, match_score['overall_score'], match_score['skills_match'], 
                      match_score['experience_match'], match_score['education_match']))
                
                # Track match results
                match_results.append({
                    'job_id': job_id,
                    'job_title': job_title,
                    'match_score': match_score,
                    'requirements': requirements
                })
            
            # Step 4: Shortlisting stage
            print(f"Shortlisting candidate {candidate_info['name']}...")
            shortlisted_for_jobs = []
            for match in match_results:
                # Check if score meets our threshold (40%)
                if match['match_score']['overall_score'] >= MATCH_THRESHOLD:
                    # Mark as shortlisted
                    shortlisted_for_jobs.append(match['job_title'])
                    print(f"Candidate {candidate_info['name']} shortlisted for {match['job_title']} with score {match['match_score']['overall_score']}%")
                    
                    # Generate interview request - force it to generate an interview request
                    print(f"Generating interview request for {match['job_title']} position...")
                    interview_request = matching_agent.generate_interview_request(
                        candidate_info, match['requirements'], match['match_score'])
                    
                    if not interview_request:
                        print(f"WARNING: Failed to generate interview request despite meeting threshold. Generating default request.")
                        # Create a default interview request if the agent fails to generate one
                        interview_request = {
                            "subject": f"Interview Invitation for {match['job_title']} Position",
                            "body": f"Dear {candidate_info['name']},\n\nWe are pleased to invite you to an interview for the {match['job_title']} position. Your profile has been reviewed and we believe you could be a good match for this role.\n\nPlease let us know your availability for the coming week.\n\nBest regards,\nRecruitment Team",
                            "proposed_dates": ["Monday 10:00 AM", "Tuesday 2:00 PM", "Thursday 11:00 AM"],
                            "interview_type": "Video Call"
                        }
                    
                    # Save to database
                    c.execute('''
                        INSERT INTO interview_requests (candidate_id, job_id, subject, body, proposed_dates, interview_type)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        candidate_id,
                        match['job_id'],
                        interview_request['subject'],
                        interview_request['body'],
                        json.dumps(interview_request['proposed_dates']),
                        interview_request['interview_type']
                    ))
                    
                    # Automatically send email to candidate
                    try:
                        print(f"Sending interview request email to {candidate_info['email']}...")
                        
                        # Use the direct send_email function from the module
                        email_sent = send_email(
                            recipient_email=candidate_info['email'],
                            subject=interview_request['subject'],
                            body=interview_request['body']
                        )
                        
                        # Always consider it sent in this simulation
                        emails_sent.append({
                            'candidate_name': candidate_info['name'],
                            'job_title': match['job_title'],
                            'email': candidate_info['email'],
                            'interview_details': {
                                'subject': interview_request['subject'],
                                'dates': interview_request['proposed_dates'],
                                'type': interview_request['interview_type']
                            }
                        })
                        print(f"Email sending {'successful' if email_sent else 'failed'} for {candidate_info['email']}")
                    except Exception as e:
                        print(f"Error with email: {e}")
            
            if shortlisted_for_jobs:
                shortlisted_candidates.append({
                    'candidate_id': candidate_id,
                    'name': candidate_info['name'],
                    'email': candidate_info['email'],
                    'shortlisted_for': shortlisted_for_jobs
                })
            
            conn.commit()
            conn.close()
        
        print(f"Completed processing all candidates.")
        return {
            "message": f"Successfully processed {len(processed_candidates)} candidates",
            "candidate_ids": processed_candidates,
            "shortlisted": shortlisted_candidates,
            "emails_sent": emails_sent
        }
    except Exception as e:
        print(f"Error processing candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics for the dashboard"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get job count
        c.execute('SELECT COUNT(*) FROM jobs')
        total_jobs = c.fetchone()[0]
        
        # Get candidate count
        c.execute('SELECT COUNT(*) FROM candidates')
        total_candidates = c.fetchone()[0]
        
        # Get total match count
        c.execute('SELECT COUNT(*) FROM match_scores')
        total_matches = c.fetchone()[0]
        
        # Get high match count (>= 80%)
        c.execute('SELECT COUNT(*) FROM match_scores WHERE overall_score >= 80')
        high_matches = c.fetchone()[0]
        
        conn.close()
        
        return {
            "totalJobs": total_jobs,
            "totalCandidates": total_candidates,
            "totalMatches": total_matches,
            "highMatches": high_matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
async def get_jobs():
    """Get all jobs"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id, title, description, created_at FROM jobs ORDER BY created_at DESC')
        jobs = []
        for row in c.fetchall():
            jobs.append({
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "created_at": row[3]
            })
        conn.close()
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidates")
async def get_candidates():
    """Get all candidates"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id, name, email, created_at FROM candidates ORDER BY created_at DESC')
        candidates = []
        for row in c.fetchall():
            candidates.append({
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "created_at": row[3]
            })
        conn.close()
        return candidates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/matches")
async def get_matches(min_score: int = 0):
    """Get all matches with option to filter by minimum score"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        query = '''
            SELECT 
                m.id,
                m.job_id,
                j.title AS job_title,
                m.candidate_id,
                c.name AS candidate_name,
                c.email AS candidate_email,
                m.overall_score,
                m.skills_match,
                m.experience_match,
                m.education_match,
                m.created_at
            FROM match_scores m
            JOIN jobs j ON m.job_id = j.id
            JOIN candidates c ON m.candidate_id = c.id
            WHERE m.overall_score >= ?
            ORDER BY m.overall_score DESC
        '''
        
        c.execute(query, (min_score,))
        
        matches = []
        for row in c.fetchall():
            matches.append({
                "id": row[0],
                "job_id": row[1],
                "job_title": row[2],
                "candidate_id": row[3],
                "candidate_name": row[4],
                "candidate_email": row[5],
                "overall_score": row[6],
                "skills_match": row[7],
                "experience_match": row[8],
                "education_match": row[9],
                "created_at": row[10]
            })
        
        conn.close()
        return matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidates/{candidate_id}/interview-request")
async def get_interview_request(candidate_id: int, job_id: int = None):
    """Get interview request for a candidate"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        query = '''
            SELECT id, candidate_id, job_id, subject, body, proposed_dates, interview_type
            FROM interview_requests
            WHERE candidate_id = ?
        '''
        
        params = [candidate_id]
        
        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)
            
        # Get the most recent interview request
        query += " ORDER BY created_at DESC LIMIT 1"
        
        c.execute(query, params)
        row = c.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Interview request not found")
        
        interview_request = {
            "id": row[0],
            "candidate_id": row[1],
            "job_id": row[2],
            "subject": row[3],
            "body": row[4],
            "proposed_dates": json.loads(row[5]),
            "interview_type": row[6]
        }
        
        conn.close()
        return interview_request
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 