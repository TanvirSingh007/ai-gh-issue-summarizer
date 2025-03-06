import os
import json
import time
import argparse
from datetime import datetime
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
INPUT_DIR = "downloaded_issues/json"
OUTPUT_DIR = "issue_summaries"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api").replace("/generate", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
BATCH_SIZE = None  # Default to process all issues
RESUME_FILE = "summarizer_progress.json"

def setup_directories():
    """Create necessary directories for output"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def format_date(date_string):
    """Format ISO date string to a more readable format"""
    if not date_string:
        return "Not available"
    try:
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y")
    except:
        return date_string

def is_bot_comment(comment):
    """Check if a comment is from a bot"""
    bot_indicators = ["bot", "taro", "[bot]"]
    if not comment.get("author") or not comment["author"].get("login"):
        return True
    
    author = comment["author"]["login"].lower()
    return any(indicator in author for indicator in bot_indicators)

def clear_ollama_context():
    """Clear the Ollama context to ensure each issue is processed independently"""
    try:
        # Instead of trying to clear context (which isn't directly supported),
        # we'll create a new conversation by sending a simple request
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Start a new conversation. Respond with only 'OK'.",
                "stream": False,
                "options": {
                    "num_ctx": 1,  # Minimal context
                    "temperature": 0.0  # No randomness
                }
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("Successfully reset Ollama context")
            return True
        else:
            print(f"Warning: Failed to reset Ollama context. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Warning: Error resetting Ollama context: {str(e)}")
        return False

def generate_summary_with_llama(issue_data):
    """Generate a summary of the issue using Llama 3.2 via Ollama"""
    
    # Clear the Ollama context before processing
    clear_ollama_context()
    
    # Extract basic issue information
    issue_number = issue_data["number"]
    title = issue_data["title"]
    body = issue_data["body"]
    state = issue_data["state"]
    created_at = format_date(issue_data["createdAt"])
    closed_at = format_date(issue_data["closedAt"]) if issue_data.get("closedAt") else "Not closed"
    url = issue_data["url"]
    
    # Extract comments (excluding bot comments)
    comments = [
        {
            "author": comment["author"]["login"],
            "body": comment["body"],
            "created_at": comment["createdAt"]
        }
        for comment in issue_data["comments"]["nodes"]
        if not is_bot_comment(comment)
    ]
    
    # Prepare the prompt for Llama
    prompt = f"""
You are analyzing a GitHub issue. Please provide a detailed summary based on the following information:

ISSUE TITLE: {title}
ISSUE STATE: {state}
ISSUE CREATED: {created_at}
ISSUE CLOSED: {closed_at}

ISSUE DESCRIPTION:
{body}

COMMENTS:
{json.dumps(comments, indent=2)}

Based on the above information, please provide:
1. A concise summary of what the issue is about.
2. A summary of the conversations, mentioning usernames.
3. Whether the ticket was accepted and fixed, or closed and returned to CS along with the reason.
4. If the issue was accepted, what fix was implemented.
5. Whether it was a code change.
6. If it was a code change, what changes were made and to what component.
7. If it was not a code change, what steps were taken to fix the issue.

Make sure to note that the issue state is {state}.
Make sure to ignore any comments that are from bots or automations such as taro or github-actions. Do not mention them in the summary unless they are part of the conversation by another user.

Format your response as a structured analysis, not as a list of answers to these questions.
"""

    # Call Ollama API
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 8192,  # Ensure enough context for large issues
                    "temperature": 0.1  # Lower temperature for more factual responses
                }
            },
            timeout=180  # 3-minute timeout for larger issues
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            return f"Error generating summary: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def process_issue(file_path):
    """Process a single issue JSON file and generate a summary"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            issue_data = json.load(f)
        
        issue_number = issue_data["number"]
        title = issue_data["title"]
        state = issue_data["state"]
        created_at = format_date(issue_data["createdAt"])
        closed_at = format_date(issue_data["closedAt"]) if issue_data.get("closedAt") else "Not closed"
        url = issue_data["url"]
        
        print(f"Processing issue #{issue_number}: {title} (State: {state})")
        
        # Generate summary using Llama
        summary = generate_summary_with_llama(issue_data)
        
        # Create markdown content
        markdown_content = f"""# {title}

## Issue Details
- **Issue Number:** {issue_number}
- **State:** {state}
- **Opened:** {created_at}
- **Closed:** {closed_at}
- **GitHub Link:** [{url}]({url})

## Summary
{summary}
"""
        
        # Save to markdown file
        output_file = os.path.join(OUTPUT_DIR, f"issue_{issue_number}.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return True
    except Exception as e:
        print(f"Error processing issue {file_path}: {str(e)}")
        return False

def save_progress(processed_files):
    """Save progress to a JSON file"""
    with open(RESUME_FILE, 'w', encoding='utf-8') as f:
        json.dump({"processed_files": processed_files}, f)

def load_progress():
    """Load progress from a JSON file"""
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("processed_files", []))
    return set()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process GitHub issues and generate summaries using Llama 3.2')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Number of issues to process in this batch (default: all issues)')
    parser.add_argument('--resume', action='store_true', help='Resume from where the script left off')
    args = parser.parse_args()
    
    setup_directories()
    print("Starting to process issues...")
    
    # Get list of JSON files
    json_files = list(Path(INPUT_DIR).glob("*.json"))
    total_files = len(json_files)
    
    print(f"Found {total_files} issues to process")
    
    # Load progress if resuming
    processed_files = load_progress() if args.resume else set()
    print(f"Already processed {len(processed_files)} issues")
    
    # Filter out already processed files
    remaining_files = [f for f in json_files if f.name not in processed_files]
    
    # Limit to batch size if specified
    if args.batch:
        batch_files = remaining_files[:args.batch]
        print(f"Processing {len(batch_files)} issues in this batch")
    else:
        batch_files = remaining_files
        print(f"Processing all {len(batch_files)} remaining issues")
    
    successful = 0
    try:
        for i, file_path in enumerate(batch_files):
            print(f"Processing {i+1}/{len(batch_files)}: {file_path.name}")
            if process_issue(file_path):
                successful += 1
                processed_files.add(file_path.name)
                save_progress(list(processed_files))  # Save progress after each successful processing
            
            # Add a small delay to avoid overwhelming Ollama
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
    
    print(f"\nCompleted! Successfully processed {successful} out of {len(batch_files)} issues in this batch.")
    print(f"Total processed so far: {len(processed_files)} out of {total_files} issues.")
    print(f"Summaries are saved in the '{OUTPUT_DIR}' directory")

if __name__ == "__main__":
    main() 