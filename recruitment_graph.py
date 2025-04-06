import os
from typing import Dict, List, TypedDict, Annotated, Sequence, Optional, Any, Union
import operator
import json
import hashlib
import pickle
from pathlib import Path
from langgraph.graph import StateGraph, END
import re
import time

# Import agent classes and Pydantic models from agents.py
from agents import JobDescriptionAgent, CandidateAgent, MatchingAgent, JobRequirements, CandidateInfo, InterviewRequest
from database import add_job_description, add_candidate, add_match_result, get_job_description
from email_service import send_email  # Assuming email_service has a send_email function

# --- Performance Optimizations ---

# Create cache directory if it doesn't exist
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Clear the cache if we want to force fresh processing
def clear_cache():
    """Clear all cached data"""
    print("Clearing all cached data to ensure fresh processing...")
    for cache_file in CACHE_DIR.glob("*.pkl"):
        try:
            cache_file.unlink()
            print(f"Deleted cache file: {cache_file}")
        except Exception as e:
            print(f"Error deleting cache file {cache_file}: {e}")

# Force clear the cache on startup
clear_cache()

def get_cache_key(data_str: str) -> str:
    """Generate a cache key from input data"""
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()

def save_to_cache(key: str, data: Any) -> None:
    """Save data to cache"""
    cache_file = CACHE_DIR / f"{key}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(key: str) -> Optional[Any]:
    """Load data from cache if it exists"""
    cache_file = CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading from cache: {e}")
    return None

# --- State Definition ---

class GraphState(TypedDict, total=False):
    """
    Represents the state of our graph.

    Attributes:
        job_description_filepath: Path to the job description CSV file.
        candidate_cv_filepath: Path to the candidate CV PDF file.
        job_id: Database ID of the processed job description.
        job_requirements: Extracted requirements from the job description.
        candidate_cv_text: Text extracted from the candidate's CV.
        candidate_info: Structured information extracted from the CV.
        match_score: Calculated match score between candidate and job.
        interview_request_email: Generated email content for the interview request.
        error: Any error messages encountered during processing.
    """
    job_description_filepath: Optional[str]
    candidate_cv_filepath: Optional[str]
    job_id: Optional[int]
    job_requirements: Optional[JobRequirements]
    candidate_cv_text: Optional[str]
    candidate_info: Optional[CandidateInfo]
    match_score: Optional[Dict[str, Any]]
    interview_request_email: Optional[InterviewRequest]
    error: Optional[str]

# --- Node Definitions ---

# Instantiate agents (consider making models/base URLs configurable)
jd_agent = JobDescriptionAgent()
candidate_agent = CandidateAgent()
matching_agent = MatchingAgent()

def parse_job_description(state: GraphState) -> GraphState:
    """Parses the job description CSV and extracts requirements for the first JD found."""
    print("--- PARSING JOB DESCRIPTION --- \n")
    filepath = state.get("job_description_filepath")
    if not filepath:
        return {"error": "Job description filepath not provided."}

    try:
        # Check cache first
        cache_key = get_cache_key(f"jd_parse_{filepath}")
        cached_result = load_from_cache(cache_key)
        
        if cached_result:
            print("Using cached job description parsing results")
            return cached_result
        
        parsed_jds = jd_agent.parse_jd_csv(filepath)
        if not parsed_jds:
            return {"error": "No job descriptions found in the CSV."}

        # For simplicity, process only the first JD in the file
        first_jd = parsed_jds[0]
        print(f"Processing JD Title: {first_jd.get('title')}")

        requirements = jd_agent.extract_key_requirements(first_jd['description'])
        if not requirements:
             # Store minimal info even if extraction fails partially
             job_id = add_job_description(first_jd.get('title', 'Unknown Title'), first_jd.get('description', ''), json.dumps({}))
             result = {"error": "Failed to extract key requirements from job description.", "job_id": job_id}
             save_to_cache(cache_key, result)
             return result

        # Store the extracted requirements and JD in the database
        job_id = add_job_description(
            title=requirements.get('title', first_jd.get('title', 'Unknown Title')), # Use extracted title if available
            description=first_jd['description'],
            requirements_json=json.dumps(requirements)
        )

        result = {"job_requirements": requirements, "job_id": job_id, "error": None} # Clear previous error if successful
        save_to_cache(cache_key, result)
        return result
    except Exception as e:
        print(f"Error in parse_job_description: {e}")
        return {"error": f"Failed to process job description: {str(e)}"}


def parse_candidate_cv(state: GraphState) -> GraphState:
    """Parses the candidate CV PDF and extracts structured information."""
    print("--- PARSING CANDIDATE CV --- \n")
    filepath = state.get("candidate_cv_filepath")
    if not filepath:
        return {"error": "Candidate CV filepath not provided."}

    try:
        # Check if we already have candidate info in state
        existing_candidate_info = state.get("candidate_info", {})
        existing_name = existing_candidate_info.get("name") if existing_candidate_info else None
        if existing_name and existing_name != "Unknown Candidate":
            print(f"Using pre-set candidate name from state: {existing_name}")
        
        # Check cache first
        cache_key = get_cache_key(f"cv_parse_{filepath}")
        cached_result = load_from_cache(cache_key)
        
        if cached_result:
            print("Using cached CV parsing results")
            # If we have a pre-set name, make sure it's preserved
            if existing_name and existing_name != "Unknown Candidate":
                if "candidate_info" in cached_result and isinstance(cached_result["candidate_info"], dict):
                    cached_result["candidate_info"]["name"] = existing_name
                    print(f"Updated cached result with pre-set name: {existing_name}")
            return cached_result
            
        cv_text = candidate_agent.parse_cv(filepath)
        if not cv_text:
             return {"error": "Failed to extract text from CV PDF."}

        # First try the agent's extraction
        print("Attempting to extract candidate info via agent...")
        candidate_info = candidate_agent.extract_candidate_info(cv_text)
        if not candidate_info:
            candidate_info = {}
        
        # If we have a pre-set name, use it
        if existing_name and existing_name != "Unknown Candidate":
            candidate_info["name"] = existing_name
            print(f"Using pre-set name: {existing_name}")
        
        # Continue with the rest of the function...    
        print(f"Initial candidate info: {candidate_info}")
            
        # Extract name from properties if available
        if isinstance(candidate_info, dict) and 'properties' in candidate_info:
            properties = candidate_info.get('properties', {})
            if properties.get('name'):
                candidate_info['name'] = properties['name']
                print(f"Found name in properties: {properties['name']}")
            
        # Try simple regex patterns to enhance extraction
        if not candidate_info.get('name') or candidate_info.get('name') == 'Unknown Candidate':
            print("Candidate name not found or is Unknown - trying to extract name from CV")
            
            # Try multiple patterns to extract names with more flexibility
            name_patterns = [
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # First Last at beginning (allows middle names)
                r'(?:Name:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Name: First Last
                r'(?:Resume of|CV of)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Resume of First Last
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$',  # First Last at end of line
                r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # First Last with leading whitespace
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*(?:Resume|CV))',  # Name followed by Resume/CV
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\n',  # Name on its own line
                r'(?:Contact|Personal)\s*Information.*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Name in contact section
                r'(?:Full Name|Name)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Full Name: First Last
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\|',  # Name followed by |
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*@',  # Name followed by @
            ]
            
            # Try each pattern
            for pattern in name_patterns:
                name_match = re.search(pattern, cv_text, re.MULTILINE)
                if name_match:
                    name = name_match.group(1).strip()
                    if name and len(name.split()) >= 2:  # Ensure we have at least first and last name
                        candidate_info['name'] = name
                        print(f"Extracted name using pattern: {name}")
                        break
            
            # If still no name, try to find any capitalized words that might be a name
            if not candidate_info.get('name') or candidate_info.get('name') == 'Unknown Candidate':
                print("Trying to find capitalized words that might be a name...")
                lines = cv_text.split('\n')
                for line in lines[:20]:  # Check first 20 lines
                    words = line.strip().split()
                    if len(words) >= 2:
                        # Look for consecutive capitalized words
                        potential_name = []
                        for i in range(len(words)-1):
                            if words[i][0].isupper() and words[i+1][0].isupper():
                                potential_name = [words[i], words[i+1]]
                                break
                        if potential_name:
                            name = ' '.join(potential_name)
                            candidate_info['name'] = name
                            print(f"Found potential name from capitalized words: {name}")
                            break
            
            # If still no name, use filename as fallback
            if not candidate_info.get('name') or candidate_info.get('name') == 'Unknown Candidate':
                filename = os.path.basename(filepath)
                candidate_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                if candidate_name and candidate_name != "":
                    candidate_info['name'] = candidate_name
                    print(f"Using filename as name: {candidate_name}")
                else:
                    # Last resort - timestamp-based name
                    candidate_info['name'] = f"Candidate {int(time.time())}"
                    print(f"Using timestamp name: {candidate_info['name']}")
                
        print(f"Final candidate name: {candidate_info.get('name', 'Unknown')}")

        # Ensure we have a valid name before storing in the database
        candidate_name = candidate_info.get('name')
        if not candidate_name or candidate_name == 'Unknown Candidate':
            # Try one last time to get a name from the filename
            filename = os.path.basename(filepath)
            candidate_name = os.path.splitext(filename)[0].replace('_', ' ').title()
            if not candidate_name:
                candidate_name = f"Candidate {int(time.time())}"
            candidate_info['name'] = candidate_name
            print(f"Using final fallback name: {candidate_name}")

        # Store candidate info in the database
        candidate_id = add_candidate(
            name=candidate_name,  # Use the validated name
            email=candidate_info.get('email', 'no-email@example.com'),
            phone=candidate_info.get('phone', ''),
            cv_filepath=filepath,
            extracted_info_json=json.dumps(candidate_info)
        )

        result = {"candidate_cv_text": cv_text, "candidate_info": candidate_info, "error": None}
        save_to_cache(cache_key, result)
        return result
    except Exception as e:
        print(f"Error in parse_candidate_cv: {e}")
        return {"error": f"Failed to process candidate CV: {str(e)}"}


def calculate_match(state: GraphState) -> GraphState:
    """Calculates the match score between the candidate and the job."""
    print("--- CALCULATING MATCH SCORE --- \n")
    candidate_info = state.get("candidate_info")
    job_requirements = state.get("job_requirements")
    job_id = state.get("job_id") # Needed for storing result

    if not candidate_info or not job_requirements or job_id is None:
        return {"error": "Missing candidate info, job requirements, or job_id for matching."}

    try:
        # Check if job requirements has requirements_json field
        if isinstance(job_requirements, dict) and 'requirements_json' in job_requirements:
            try:
                requirements_json = job_requirements.get('requirements_json')
                if requirements_json:
                    parsed_requirements = json.loads(requirements_json)
                    # Merge the parsed requirements with the existing requirements
                    for key, value in parsed_requirements.items():
                        if key not in job_requirements or not job_requirements.get(key):
                            job_requirements[key] = value
                    print(f"Enhanced job requirements from requirements_json")
            except Exception as e:
                print(f"Error parsing requirements_json: {e}")
                
        # Extract data from properties if needed
        if isinstance(candidate_info, dict) and 'properties' in candidate_info:
            print("Extracting candidate info from properties")
            properties = candidate_info.get('properties', {})
            # Create a new dict with the properties
            extracted_candidate = {}
            for key, value in properties.items():
                extracted_candidate[key] = value
            
            # Make sure we extract nested structures as well
            if 'skills' in properties:
                extracted_candidate['skills'] = properties.get('skills', [])
            if 'education' in properties:
                extracted_candidate['education'] = properties.get('education', [])
            if 'experience' in properties:
                extracted_candidate['experience'] = properties.get('experience', [])
            if 'name' in properties:
                extracted_candidate['name'] = properties.get('name', 'Unknown Candidate')
            if 'email' in properties:
                extracted_candidate['email'] = properties.get('email', 'no-email@example.com')
            
            # Add any top-level fields not in properties
            for key, value in candidate_info.items():
                if key != 'properties' and key != 'required' and key not in extracted_candidate:
                    extracted_candidate[key] = value
            
            # Replace original with extracted
            candidate_info = extracted_candidate
            
        if isinstance(job_requirements, dict) and 'properties' in job_requirements:
            print("Extracting job requirements from properties")
            properties = job_requirements.get('properties', {})
            # Create a new dict with the properties
            extracted_job = {}
            for key, value in properties.items():
                extracted_job[key] = value
            
            # Make sure we extract key fields
            if 'title' in properties:
                extracted_job['title'] = properties.get('title', 'Unknown')
            if 'required_skills' in properties:
                extracted_job['required_skills'] = properties.get('required_skills', [])
            if 'company' in properties:
                extracted_job['company'] = properties.get('company', '')
            
            # Add any top-level fields not in properties
            for key, value in job_requirements.items():
                if key != 'properties' and key != 'required' and key not in extracted_job:
                    extracted_job[key] = value
            
            # Replace original with extracted
            job_requirements = extracted_job
        
        # Try to extract required_skills from description if needed
        if (not job_requirements.get('required_skills') or len(job_requirements.get('required_skills', [])) == 0) and job_requirements.get('description'):
            print("Attempting to extract skills from job description")
            description = job_requirements.get('description', '')
            # Use the job agent to extract skills instead of hardcoded list
            extracted_requirements = jd_agent.extract_key_requirements(description)
            if extracted_requirements and 'required_skills' in extracted_requirements:
                job_requirements['required_skills'] = extracted_requirements['required_skills']
                print(f"Extracted skills using agent: {job_requirements['required_skills']}")
            
            # If we have other useful extracted data, add it to job requirements
            if extracted_requirements:
                for key, value in extracted_requirements.items():
                    if key not in job_requirements or not job_requirements.get(key):
                        job_requirements[key] = value
                        print(f"Updated job requirements with {key}")
        
        print(f"Candidate used for matching: {candidate_info.get('name', 'Unknown')} with skills: {candidate_info.get('skills', [])}")
        print(f"Job used for matching: {job_requirements.get('title', 'Unknown')} with skills: {job_requirements.get('required_skills', [])}")
        
        # Check cache with the normalized information
        cache_key = get_cache_key(f"match_{job_id}_{json.dumps(candidate_info)}")
        cached_result = load_from_cache(cache_key)
        
        if cached_result:
            print("Using cached match score results")
            return cached_result
            
        match_score = matching_agent.calculate_match_score(candidate_info, job_requirements)

        # Assume candidate_id is retrievable or not needed here, we use email as identifier for now
        # In a real app, we'd link via candidate_id stored earlier
        candidate_email = candidate_info.get('email', 'unknown')
        candidate_name = candidate_info.get('name', 'Unknown Candidate')

        # Store match result
        add_match_result(
            job_id=job_id,
            candidate_email=candidate_email, # Using email as a simple key for this example
            score=match_score.get("overall_score", 0),
            match_details_json=json.dumps(match_score)
        )

        result = {
            "match_score": match_score, 
            "error": None,
            # Update candidate_info in state with extracted info
            "candidate_info": candidate_info,
            # Update job_requirements in state with extracted info
            "job_requirements": job_requirements
        }
        save_to_cache(cache_key, result)
        return result
    except Exception as e:
        print(f"Error in calculate_match: {e}")
        return {"error": f"Failed to calculate match score: {str(e)}"}


def decide_interview(state: GraphState) -> Dict[str, str]:
    """Determines the next step based on the match score."""
    print("--- DECIDING INTERVIEW --- \n")
    error = state.get("error")
    if error:
        print(f"Error before decision: {error}")
        return {"decision": "handle_error"} # Route to error handling if any previous step failed

    match_score_data = state.get("match_score")
    if not match_score_data:
        print("Error: Match score not found for decision.")
        return {"decision": "handle_error"} # Cannot decide without score

    score = match_score_data.get("overall_score", 0)
    # Get threshold from matching agent instead of hardcoding
    threshold = matching_agent.match_threshold  # Use the threshold from the agent
    
    # Update the threshold_met in the match_score data
    match_score_data["threshold_met"] = score >= threshold
    state["match_score"] = match_score_data
    
    print(f"Overall Score: {score}, Threshold: {threshold}")
    if score >= threshold:
        print("Decision: Generate Interview Request")
        return {"decision": "generate_interview_request"}
    else:
        print("Decision: Reject (Score Too Low)")
        return {"decision": "finalize_reject"}


def generate_interview_email(state: GraphState) -> dict:
    """Generates the interview request email content."""
    print("--- GENERATING INTERVIEW EMAIL --- \n")
    candidate_info = state.get("candidate_info")
    job_requirements = state.get("job_requirements")
    match_score = state.get("match_score")

    if not candidate_info or not job_requirements or not match_score:
         return {"error": "Missing data required for generating interview email."}

    try:
        # Extract candidate name properly
        candidate_name = "Candidate"
        if isinstance(candidate_info, dict):
            if 'properties' in candidate_info and candidate_info['properties'].get('name'):
                candidate_name = candidate_info['properties']['name']
            elif candidate_info.get('name'):
                candidate_name = candidate_info['name']
                
        # Extract job title properly
        job_title = "Position"
        if isinstance(job_requirements, dict):
            if 'properties' in job_requirements and job_requirements['properties'].get('title'):
                job_title = job_requirements['properties']['title']
            elif job_requirements.get('title'):
                job_title = job_requirements['title']
                
        print(f"Generating email for {candidate_name} for {job_title} position")
                
        # Check cache first
        cache_key = get_cache_key(f"interview_{candidate_name}_{job_title}")
        cached_result = load_from_cache(cache_key)
        
        if cached_result:
            print("Using cached interview email results")
            return cached_result
            
        interview_request = matching_agent.generate_interview_request(
            candidate_info, job_requirements, match_score
        )
        
        # Always create an interview request with proper HTML formatting
        if not interview_request:
             # This can happen if score is technically >= threshold but LLM fails or decides not to generate
             print("Interview request generation returned None - creating default email")
             
        # Make sure we have a properly structured interview request
        if not interview_request or not isinstance(interview_request, dict):
            interview_request = {}
             
        # Ensure subject exists
        if not interview_request.get('subject'):
            interview_request['subject'] = f"Interview Request - {job_title} Position"
            
        # Create a default email body if needed
        if not interview_request.get('body'):
            interview_request['body'] = f"""
            <html>
            <body>
            <p>Dear {candidate_name},</p>
            
            <p>We're pleased to inform you that your application for the <b>{job_title}</b>
            position has been successful, and we would like to invite you for an interview.</p>
            
            <p>Your match score was <b>{match_score.get('overall_score', 0)}%</b>, which meets our requirements.</p>
            
            <p>Please reply to this email to confirm your availability for an interview.</p>
            
            <p>Best regards,<br>
            Recruitment Team</p>
            </body>
            </html>
            """
            
        else:
            # Ensure the body is properly formatted as HTML
            body = interview_request['body']
            if not body.strip().startswith("<html") and not body.strip().startswith("<body"):
                interview_request['body'] = f"""
                <html>
                <body>
                {body}
                </body>
                </html>
                """
                
        print(f"Generated interview request with subject: {interview_request.get('subject')}")
        print(f"Email body preview: {interview_request.get('body', '')[:100]}...")
        
        # Store the generated email content in the state
        updated_state = dict(state)  # Make a copy
        updated_state["interview_request_email"] = interview_request
        updated_state["error"] = None
        
        save_to_cache(cache_key, updated_state)  # Cache the whole updated state
        return updated_state
    except Exception as e:
        print(f"Error in generate_interview_email: {e}")
        return {"error": f"Failed to generate interview email content: {str(e)}"}

def send_interview_email(state: GraphState) -> dict:
    """Send the generated interview email to the candidate."""
    try:
        print("--- SENDING INTERVIEW EMAIL --- \n")
        
        # Extract the email data and recipient
        email_data = state.get("interview_request_email", {})
        if not email_data:
            print("No interview email data found in state")
            return {"error": "No interview email data found in state"}
        
        candidate_info = state.get("candidate_info", {})
        recipient_email = candidate_info.get("email")
        
        if not recipient_email:
            print("No recipient email found in candidate info")
            if "properties" in candidate_info:
                recipient_email = candidate_info["properties"].get("email")
                if recipient_email:
                    print(f"Found recipient email in properties: {recipient_email}")
            
            if not recipient_email:
                return {"error": "No recipient email found in candidate info"}
        
        # Get job requirements for the subject
        job_requirements = state.get("job_requirements", {})
        job_title = job_requirements.get("title", "")
        if not job_title and "properties" in job_requirements:
            job_title = job_requirements["properties"].get("title", "")
        
        # Email subject and body
        subject = email_data.get("subject", f"Interview Request - {job_title} Position")
        
        # Get the email body, ensuring it's properly formatted
        body = email_data.get("body", "")
        if not body:
            body = email_data.get("email_body", "")
            
        if not body:
            print(f"Warning: No email body found. Available keys: {list(email_data.keys())}")
            # Create a fallback email body
            body = f"""
            <html>
            <body>
            <p>Dear {candidate_info.get('name', 'Candidate')},</p>
            
            <p>We're pleased to inform you that your application for the {job_title} position
            has been successful, and we would like to invite you for an interview.</p>
            
            <p>Please reply to this email to confirm your availability.</p>
            
            <p>Best regards,<br>
            Recruitment Team</p>
            </body>
            </html>
            """
            print("Created fallback email body")
        
        # Ensure the body is properly formatted as HTML
        if not body.startswith("<html"):
            body = f"""
            <html>
            <body>
            {body}
            </body>
            </html>
            """
        
        # Send the email
        try:
            print(f"Sending email to {recipient_email} with subject: {subject}")
            print(f"Email body preview: {body[:100]}...")
            send_email(
                recipient_email=recipient_email,
                subject=subject,
                body=body,
                html=True
            )
            print(f"Successfully sent interview email to {candidate_info.get('name')} at {recipient_email}")
            
            # Update state
            updated_state = dict(state)  # Make a copy
            updated_state["email_sent"] = True
            updated_state["email_recipient"] = recipient_email
            return updated_state
        except Exception as e:
            print(f"Error in send_interview_email: {e}")
            return {"error": f"Failed to send interview email: {str(e)}"}
            
    except Exception as e:
        print(f"Error in send_interview_email: {e}")
        return {"error": f"Failed to send interview email: {str(e)}"}


def finalize_reject(state: GraphState) -> GraphState:
    """Handles the case where the candidate is rejected (no email sent)."""
    print("--- FINALIZING REJECTION --- \n")
    
    # We could log the rejection reason here
    match_score = state.get("match_score", {})
    job_id = state.get("job_id")
    candidate_info = state.get("candidate_info", {})
    
    # Extract candidate name properly
    candidate_name = "Unknown Candidate"
    if isinstance(candidate_info, dict):
        if 'properties' in candidate_info and isinstance(candidate_info['properties'], dict):
            candidate_name = candidate_info['properties'].get('name', "Unknown Candidate")
        else:
            candidate_name = candidate_info.get('name', "Unknown Candidate")
    
    job_title = "Unknown Job"
    job_requirements = state.get("job_requirements", {})
    if isinstance(job_requirements, dict):
        if 'properties' in job_requirements and isinstance(job_requirements['properties'], dict):
            job_title = job_requirements['properties'].get('title', "Unknown Job")
        else:
            job_title = job_requirements.get('title', "Unknown Job")
    
    print(f"Candidate {candidate_name} rejected for job_id {job_id} ({job_title})")
    print(f"Match score: {match_score.get('overall_score', 0)}")
    print(f"Skills match: {match_score.get('skills_match', 0)}")
    
    # No further action required, match is already stored in the database
    return {"error": None}  # Successful completion of the rejection path

def handle_error(state: GraphState) -> GraphState:
    """Handles the final state when an error occurred."""
    print("--- HANDLING ERROR --- \n")
    error_message = state.get("error", "Unknown error occurred.")
    print(f"Workflow failed with error: {error_message}")
    # Log the error, notify admin, etc.
    return {"error": error_message} # End the graph


# --- Build the Graph ---

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("parse_job_description", parse_job_description)
workflow.add_node("parse_candidate_cv", parse_candidate_cv)
workflow.add_node("calculate_match", calculate_match)
workflow.add_node("generate_interview_email", generate_interview_email)
workflow.add_node("send_interview_email", send_interview_email) # Placeholder for actual sending
workflow.add_node("finalize_reject", finalize_reject)
workflow.add_node("handle_error", handle_error) # Error handling node

# Define edges
workflow.set_entry_point("parse_job_description")
workflow.add_edge("parse_job_description", "parse_candidate_cv")
workflow.add_edge("parse_candidate_cv", "calculate_match")
workflow.add_edge("calculate_match", "decide_interview") # Route to decision node

# Conditional edge based on match score
workflow.add_node("decide_interview", decide_interview)
workflow.add_edge("calculate_match", "decide_interview")  # Add decide_interview as a node first

workflow.add_conditional_edges(
    "decide_interview",
    lambda x: x["decision"],
    {
        "generate_interview_request": "generate_interview_email",
        "finalize_reject": "finalize_reject",
        "handle_error": "handle_error"  # Route to error if decision node itself fails or error occurred before
    }
)

workflow.add_edge("generate_interview_email", "send_interview_email") # If email generated, attempt send
workflow.add_edge("send_interview_email", END) # End after attempting send
workflow.add_edge("finalize_reject", END) # End after rejection
workflow.add_edge("handle_error", END) # End after error handling


# Compile the graph
app = workflow.compile()


# --- Example Usage ---

if __name__ == "__main__":
    # Ensure necessary directories exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    # Example file paths (replace with actual paths from your app)
    # You would get these from your Flask app's upload handling
    example_jd_path = "uploads/example_job_description.csv" # Needs a sample CSV here
    example_cv_path = "uploads/example_candidate_cv.pdf"   # Needs a sample PDF here

    # Create dummy files if they don't exist for basic testing
    if not os.path.exists(example_jd_path):
        with open(example_jd_path, 'w') as f:
            f.write("Job Title,Job Description\n")
            f.write('Software Engineer,"We are looking for a skilled Software Engineer with 5 years of Python and Java experience. Responsibilities include developing web applications and working with cloud platforms like AWS. Bachelor\'s degree required."\n')
            f.write('Data Scientist,"Seeking a Data Scientist with 3 years experience in machine learning, Python, and SQL. Master\'s degree preferred."\n')
        print(f"Created dummy JD file: {example_jd_path}")

    if not os.path.exists(example_cv_path):
        # Creating a dummy PDF requires a library like reportlab or fpdf
        # For now, we'll just note that it needs to exist.
        # You should upload a real PDF CV to 'uploads/example_candidate_cv.pdf'
        print(f"Warning: Dummy CV file not created. Please place a PDF CV at: {example_cv_path}")
        # To allow graph to run partially, we might skip execution or handle file not found
        # For now, let's just exit if the CV isn't there.
        # exit() # Uncomment this to prevent running without a CV

    # Prepare the initial state (assuming CV exists for this run)
    if os.path.exists(example_cv_path):
        initial_state = GraphState(
            job_description_filepath=example_jd_path,
            candidate_cv_filepath=example_cv_path
        )

        print("\n--- RUNNING RECRUITMENT GRAPH --- \n")
        # Run the graph
        final_state = app.invoke(initial_state)

        print("\n--- GRAPH EXECUTION COMPLETE --- \n")
        print("Final State:")
        # Pretty print the final state
        print(json.dumps(final_state, indent=2, default=str)) # Use default=str for non-serializable objects like Pydantic models if needed
    else:
         print(f"Skipping graph execution because CV file not found: {example_cv_path}")

def process_job_application(state: GraphState):
    """Process a job application by matching a CV against job requirements.
    
    Args:
        state: The current graph state containing:
            - job_path: The path to the job description CSV file
            - job_id: The ID of the job in the database
            - candidate_cv_path: The path to the candidate's CV file
        
    Returns:
        A dictionary with the results of processing the application
    """
    # Validate inputs
    if not state.candidate_cv_path:
        return {"error": "Candidate CV path is required"}
    
    if not state.job_description_filepath and not state.job_id:
        return {"error": "Either job file path or job ID is required"}
    
    # Clear the cache to ensure fresh processing
    print("Clearing all cached data to ensure fresh processing...")
    clear_cache()
    
    print(f"Processing application for CV: {state.candidate_cv_path}")
    if state.job_id:
        print(f"Job ID: {state.job_id}")
    else:
        print(f"Job File: {state.job_description_filepath}")
    
    # Extract job requirements if job_id is provided
    if state.job_id:
        try:
            db = Database("recruitment.db")
            job = db.get_job(state.job_id)
            if not job:
                return {"error": f"Job with ID {state.job_id} not found"}
            
            requirements_json = job.get('requirements_json')
            if requirements_json:
                requirements = json.loads(requirements_json)
                # Handle properties wrapping
                if "properties" in requirements:
                    # Create a new dict with the properties plus any top-level fields
                    extracted_job = {**requirements}
                    for key, value in requirements["properties"].items():
                        if key not in extracted_job or not extracted_job[key]:
                            extracted_job[key] = value
                else:
                    extracted_job = requirements
                
                # Set job title
                extracted_job["title"] = job.get('title')
                state.job_requirements = extracted_job
        except Exception as e:
            print(f"Error loading job requirements from DB: {e}")
            return {"error": f"Failed to load job: {str(e)}"}
    
    # Create initial state for the workflow
    # Run the graph
    try:
        final_state = app.invoke(state)
        
        # Check if we have job requirements and candidate info
        job_requirements = final_state.get("job_requirements", {})
        candidate_info = final_state.get("candidate_info", {})
        match_score = final_state.get("match_score", {})
        
        # Debug info
        job_title = job_requirements.get("title", "Unknown")
        if not job_title and "properties" in job_requirements:
            job_title = job_requirements["properties"].get("title", "Unknown")
            
        candidate_name = candidate_info.get("name", "Unknown")
        if not candidate_name and "properties" in candidate_info:
            candidate_name = candidate_info["properties"].get("name", "Unknown")
            
        print(f"Job Title: {job_title}")
        print(f"Candidate: {candidate_name}")
        print(f"Match Score: {match_score.get('overall_score', 0)}")
        
        # Format for API response
        result = {
            "status": "completed",
            "job": {
                "title": job_title,
                "company": job_requirements.get("company", ""),
                "skills_required": job_requirements.get("required_skills", []),
                "experience_required": job_requirements.get("experience_years", 0),
                "qualifications": job_requirements.get("qualifications", [])
            },
            "candidate": {
                "name": candidate_name,
                "email": candidate_info.get("email", "Unknown"),
                "skills": candidate_info.get("skills", []),
                "education": candidate_info.get("education", []),
                "experience": candidate_info.get("experience", [])
            },
            "match": {
                "overall_score": match_score.get("overall_score", 0),
                "skills_match": match_score.get("skills_match", 0),
                "experience_match": match_score.get("experience_match", 0),
                "education_match": match_score.get("education_match", 0),
                "threshold_met": match_score.get("threshold_met", False),
            }
        }
        
        # Add interview data if available
        interview_data = final_state.get("interview_request_email", {})
        if interview_data and match_score.get("threshold_met", False):
            result["interview"] = {
                "generated": True,
                "subject": interview_data.get("subject", ""),
                "email_preview": interview_data.get("email_body", "")[:200] + "...",
                "proposed_dates": interview_data.get("proposed_dates", []),
                "interview_type": interview_data.get("interview_type", ""),
                "email_sent": final_state.get("email_sent", False)
            }
        else:
            result["interview"] = {
                "generated": False
            }
            
        # Return any errors that occurred during processing
        if final_state.get("error"):
            result["error"] = final_state["error"]
        
        return result
    except Exception as e:
        print(f"Error in process_job_application: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)} 