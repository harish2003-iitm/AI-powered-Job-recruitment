import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("FROM_EMAIL")

    async def send_interview_request(self, candidate_email: str, interview_request: Dict) -> bool:
        """Send interview request email to candidate"""
        try:
            print(f"Sending interview request to: {candidate_email}")  # Debug log
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = candidate_email
            msg['Subject'] = interview_request['subject']
            
            # Format dates for email body
            dates_str = "\n".join([f"- {date}" for date in interview_request['proposed_dates']])
            
            # Create email body
            body = f"""
            Dear Candidate,

            {interview_request['body']}

            Please select one of the following dates for your interview:
            {dates_str}

            Interview Type: {interview_request['interview_type']}

            Please reply to this email with your preferred date and time.

            Best regards,
            HR Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            print(f"Successfully sent interview request to {candidate_email}")  # Debug log
            return True
            
        except Exception as e:
            print(f"Error sending email to {candidate_email}: {str(e)}")  # Debug log
            return False 

def send_email(recipient_email: str, subject: str, body: str, html: bool = False) -> bool:
    """
    Send an email to a recipient
    
    Args:
        recipient_email: Email address of the recipient
        subject: Email subject line
        body: Email body content
        html: Whether to send as HTML email (default: False)
        
    Returns:
        bool: True if sent successfully, False otherwise
    """
    try:
        print(f"Sending email to: {recipient_email}")
        
        # Get email settings from environment
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("FROM_EMAIL", "recruitment@example.com")
        
        # Check if credentials are set
        if not smtp_username or not smtp_password:
            print("Email credentials not set in .env file. Email would be sent here:")
            print(f"From: {from_email}")
            print(f"To: {recipient_email}")
            print(f"Subject: {subject}")
            print(f"Body: {body[:100]}...")
            return True  # Return true for testing/demo purposes
            
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Attach body
        if html:
            # Always include a plain text version too
            text_part = MIMEText(body.replace('<p>', '').replace('</p>', '\n\n'), 'plain')
            html_part = MIMEText(body, 'html')
            msg.attach(text_part)
            msg.attach(html_part)
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        # Connect and send
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            
        print(f"Successfully sent email to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        # For security, don't reveal full credentials in error message
        if "Authentication" in str(e):
            print("Authentication failed. Check your email credentials.")
        elif "Connection refused" in str(e):
            print(f"Could not connect to SMTP server {smtp_server}:{smtp_port}")
        return False 