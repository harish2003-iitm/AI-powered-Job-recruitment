import sqlite3
from typing import Optional
import json
from datetime import datetime

class Database:
    def __init__(self, db_path: str = "recruitment.db"):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create Job Descriptions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    description TEXT NOT NULL,
                    required_skills TEXT NOT NULL,
                    experience_years INTEGER NOT NULL,
                    qualifications TEXT NOT NULL,
                    responsibilities TEXT NOT NULL,
                    requirements_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Add requirements_json column if it doesn't exist
            try:
                cursor.execute("PRAGMA table_info(job_descriptions)")
                columns = [col[1] for col in cursor.fetchall()]
                if 'requirements_json' not in columns:
                    cursor.execute('ALTER TABLE job_descriptions ADD COLUMN requirements_json TEXT')
                    print("Added requirements_json column to job_descriptions table")
            except Exception as e:
                print(f"Error checking or adding column: {e}")

            # Create Candidates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT,
                    education TEXT NOT NULL,
                    experience TEXT NOT NULL,
                    skills TEXT NOT NULL,
                    certifications TEXT NOT NULL,
                    cv_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create Match Scores table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    candidate_id INTEGER NOT NULL,
                    overall_score REAL NOT NULL,
                    skills_match REAL NOT NULL,
                    experience_match REAL NOT NULL,
                    education_match REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
                    FOREIGN KEY (candidate_id) REFERENCES candidates (id)
                )
            ''')

            # Create Interview Requests table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interview_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_score_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    proposed_dates TEXT NOT NULL,
                    interview_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_score_id) REFERENCES match_scores (id)
                )
            ''')

            conn.commit()

    def insert_job_description(self, job_data: dict) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO job_descriptions 
                (title, company, description, required_skills, experience_years, 
                qualifications, responsibilities)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_data['title'],
                job_data['company'],
                job_data['description'],
                json.dumps(job_data['required_skills']),
                job_data['experience_years'],
                json.dumps(job_data['qualifications']),
                json.dumps(job_data['responsibilities'])
            ))
            conn.commit()
            return cursor.lastrowid

    def insert_candidate(self, candidate_data: dict) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO candidates 
                (name, email, phone, education, experience, skills, certifications, cv_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                candidate_data['name'],
                candidate_data['email'],
                candidate_data.get('phone'),
                json.dumps(candidate_data['education']),
                json.dumps(candidate_data['experience']),
                json.dumps(candidate_data['skills']),
                json.dumps(candidate_data['certifications']),
                candidate_data['cv_path']
            ))
            conn.commit()
            return cursor.lastrowid

    def insert_match_score(self, match_data: dict) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO match_scores 
                (job_id, candidate_id, overall_score, skills_match, experience_match, education_match)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                match_data['job_id'],
                match_data['candidate_id'],
                match_data['overall_score'],
                match_data['skills_match'],
                match_data['experience_match'],
                match_data['education_match']
            ))
            conn.commit()
            return cursor.lastrowid

    def insert_interview_request(self, interview_data: dict) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interview_requests 
                (match_score_id, status, proposed_dates, interview_type)
                VALUES (?, ?, ?, ?)
            ''', (
                interview_data['match_score_id'],
                interview_data['status'],
                json.dumps([d.isoformat() for d in interview_data['proposed_dates']]),
                interview_data['interview_type']
            ))
            conn.commit()
            return cursor.lastrowid

    def get_job_description(self, job_id: int) -> Optional[dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM job_descriptions WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'title': row[1],
                    'company': row[2],
                    'description': row[3],
                    'required_skills': json.loads(row[4]),
                    'experience_years': row[5],
                    'qualifications': json.loads(row[6]),
                    'responsibilities': json.loads(row[7]),
                    'created_at': row[8]
                }
            return None

    def get_candidate(self, candidate_id: int) -> Optional[dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM candidates WHERE id = ?', (candidate_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'email': row[2],
                    'phone': row[3],
                    'education': json.loads(row[4]),
                    'experience': json.loads(row[5]),
                    'skills': json.loads(row[6]),
                    'certifications': json.loads(row[7]),
                    'cv_path': row[8],
                    'created_at': row[9]
                }
            return None 

# Direct database access helper functions for use with the recruitment graph
# These act as a simpler interface to the Database class

def get_db_connection():
    """Get a database connection"""
    return sqlite3.connect("recruitment.db")

def add_job_description(title: str, description: str, requirements_json: str) -> int:
    """Add a job description to the database and return its ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # First, check if the table has the expected columns
        cursor.execute("PRAGMA table_info(job_descriptions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Parse requirements if provided
        if requirements_json:
            requirements = json.loads(requirements_json)
        else:
            requirements = {}

        # If we're using the main.py schema instead of Database class schema
        if 'company' not in columns:
            cursor.execute('''
                INSERT INTO jobs (title, description)
                VALUES (?, ?)
            ''', (
                title,
                description
            ))
        else:
            # Check if requirements_json column exists
            has_requirements_json = 'requirements_json' in columns
            
            # Using the Database class schema
            if has_requirements_json:
                cursor.execute('''
                    INSERT INTO job_descriptions 
                    (title, company, description, required_skills, experience_years, 
                    qualifications, responsibilities, requirements_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    title,
                    requirements.get('company', 'Unknown Company'),
                    description,
                    json.dumps(requirements.get('required_skills', [])),
                    requirements.get('experience_years', 0),
                    json.dumps(requirements.get('qualifications', [])),
                    json.dumps(requirements.get('responsibilities', [])),
                    requirements_json
                ))
            else:
                cursor.execute('''
                    INSERT INTO job_descriptions 
                    (title, company, description, required_skills, experience_years, 
                    qualifications, responsibilities)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    title,
                    requirements.get('company', 'Unknown Company'),
                    description,
                    json.dumps(requirements.get('required_skills', [])),
                    requirements.get('experience_years', 0),
                    json.dumps(requirements.get('qualifications', [])),
                    json.dumps(requirements.get('responsibilities', []))
                ))
        conn.commit()
        new_id = cursor.lastrowid
        return new_id
    except Exception as e:
        print(f"Error adding job description: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def add_candidate(name: str, email: str, phone: str, cv_filepath: str, extracted_info_json: str) -> int:
    """Add a candidate to the database and return their ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # First, check if the table has the expected columns
        cursor.execute("PRAGMA table_info(candidates)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Parse extracted info if provided 
        if extracted_info_json:
            info = json.loads(extracted_info_json)
        else:
            info = {}
        
        # If we're using the main.py schema
        if 'certifications' not in columns:
            cursor.execute('''
                INSERT INTO candidates 
                (name, email, phone, education, experience, skills)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                name,
                email,
                phone,
                json.dumps(info.get('education', [])),
                json.dumps(info.get('experience', [])),
                json.dumps(info.get('skills', []))
            ))
        else:
            # Using the Database class schema
            cursor.execute('''
                INSERT INTO candidates 
                (name, email, phone, education, experience, skills, certifications, cv_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name,
                email,
                phone,
                json.dumps(info.get('education', [])),
                json.dumps(info.get('experience', [])),
                json.dumps(info.get('skills', [])),
                json.dumps(info.get('certifications', [])),
                cv_filepath
            ))
        conn.commit()
        new_id = cursor.lastrowid
        return new_id
    except Exception as e:
        print(f"Error adding candidate: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def add_match_result(job_id: int, candidate_email: str, score: float, match_details_json: str) -> int:
    """Add a match result to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # First get the candidate ID from their email
        cursor.execute('SELECT id FROM candidates WHERE email = ?', (candidate_email,))
        candidate_row = cursor.fetchone()
        if not candidate_row:
            # If candidate not found in candidates table, try using a default ID or create a new one
            print(f"Candidate with email {candidate_email} not found, using dummy ID")
            candidate_id = 1  # Use a default ID
        else:
            candidate_id = candidate_row[0]
        
        # Check which table to use
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='match_scores'")
        if cursor.fetchone():
            table_name = 'match_scores'
        else:
            table_name = 'matches'
        
        # Parse match details if provided
        if match_details_json:
            details = json.loads(match_details_json)
        else:
            details = {"skills_match": 0, "experience_match": 0, "education_match": 0}
        
        # Determine which table structure to use
        if table_name == 'match_scores':
            cursor.execute(f'''
                INSERT INTO {table_name}
                (job_id, candidate_id, overall_score, skills_match, experience_match, education_match)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                job_id,
                candidate_id,
                score,
                details.get('skills_match', 0),
                details.get('experience_match', 0),
                details.get('education_match', 0)
            ))
        else:
            cursor.execute(f'''
                INSERT INTO {table_name}
                (job_id, candidate_id, overall_score)
                VALUES (?, ?, ?)
            ''', (
                job_id,
                candidate_id,
                score
            ))
        
        conn.commit()
        new_id = cursor.lastrowid
        return new_id
    except Exception as e:
        print(f"Error adding match result: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def get_job_description_by_id(job_id: int) -> dict:
    """Get a job description by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # First try the job_descriptions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='job_descriptions'")
        if cursor.fetchone():
            cursor.execute('SELECT * FROM job_descriptions WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            if row:
                # Get column names
                cursor.execute('PRAGMA table_info(job_descriptions)')
                columns = [col[1] for col in cursor.fetchall()]
                job_data = {}
                for i, col in enumerate(columns):
                    job_data[col] = row[i]
                
                # Process specific fields
                if 'required_skills' in job_data and job_data['required_skills']:
                    job_data['required_skills'] = json.loads(job_data['required_skills'])
                if 'qualifications' in job_data and job_data['qualifications']:
                    job_data['qualifications'] = json.loads(job_data['qualifications'])
                if 'responsibilities' in job_data and job_data['responsibilities']:
                    job_data['responsibilities'] = json.loads(job_data['responsibilities'])
                
                return job_data
        
        # If not found or table doesn't exist, try the jobs table
        cursor.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Job with ID {job_id} not found")
        
        # Get column names
        cursor.execute('PRAGMA table_info(jobs)')
        columns = [col[1] for col in cursor.fetchall()]
        
        job_data = {}
        for i, col in enumerate(columns):
            job_data[col] = row[i]
        
        # Parse requirements_json if available
        if 'requirements_json' in job_data and job_data['requirements_json']:
            try:
                requirements = json.loads(job_data['requirements_json'])
                # Add requirements fields to the job data
                for key, value in requirements.items():
                    if key not in job_data or not job_data[key]:
                        job_data[key] = value
            except json.JSONDecodeError:
                print(f"Error parsing requirements_json for job {job_id}")
        
        return job_data
    except Exception as e:
        print(f"Error getting job description: {e}")
        raise
    finally:
        conn.close()
        
# Alias for compatibility with recruitment_graph.py
get_job_description = get_job_description_by_id 