#!/usr/bin/env python3
"""
GitHub Issue Metrics Extractor

This script analyzes downloaded GitHub issues using an LLM (via Ollama)
to extract useful metrics and insights, then saves them to a JSON file.

Usage:
    python metrics_extractor.py --model <model_name>
"""

import os
import json
import time
import argparse
import glob
from datetime import datetime
from pathlib import Path
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter, defaultdict
import statistics

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "downloaded_issues" / "json"
OUTPUT_DIR = DATA_DIR / "metrics"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # Default model

def setup_directories():
    """Create necessary directories for output"""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Created output directory: {OUTPUT_DIR}")

def format_date(date_string):
    """Format ISO date string to a more readable format"""
    if not date_string:
        return None
    try:
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt
    except:
        return None

def calculate_time_to_resolution(created_at, closed_at):
    """Calculate time to resolution in days"""
    if not created_at or not closed_at:
        return None
    
    delta = closed_at - created_at
    return delta.days

def extract_basic_metrics(issues_data):
    """Extract basic metrics from issues data without using LLM"""
    metrics = {
        "total_issues": len(issues_data),
        "open_issues": 0,
        "closed_issues": 0,
        "issue_creation_timeline": defaultdict(int),
        "issue_closure_timeline": defaultdict(int),
        "time_to_resolution": [],
        "time_to_resolution_avg": None,
        "time_to_resolution_median": None,
        "labels_distribution": Counter(),
        "authors": Counter(),
        "commenters": Counter(),
        "comments_per_issue": [],
        "comments_per_issue_avg": None,
        "issue_age_distribution": {
            "0-7_days": 0,
            "8-14_days": 0,
            "15-30_days": 0,
            "31-90_days": 0,
            "90+_days": 0
        }
    }
    
    for issue in issues_data:
        # State counts
        if issue["state"] == "OPEN":
            metrics["open_issues"] += 1
        else:
            metrics["closed_issues"] += 1
        
        # Creation timeline
        created_at = format_date(issue["createdAt"])
        if created_at:
            month_year = created_at.strftime("%Y-%m")
            metrics["issue_creation_timeline"][month_year] += 1
        
        # Closure timeline
        closed_at = format_date(issue["closedAt"])
        if closed_at:
            month_year = closed_at.strftime("%Y-%m")
            metrics["issue_closure_timeline"][month_year] += 1
        
        # Time to resolution
        resolution_time = calculate_time_to_resolution(created_at, closed_at)
        if resolution_time is not None:
            metrics["time_to_resolution"].append(resolution_time)
            
            # Age distribution
            if resolution_time <= 7:
                metrics["issue_age_distribution"]["0-7_days"] += 1
            elif resolution_time <= 14:
                metrics["issue_age_distribution"]["8-14_days"] += 1
            elif resolution_time <= 30:
                metrics["issue_age_distribution"]["15-30_days"] += 1
            elif resolution_time <= 90:
                metrics["issue_age_distribution"]["31-90_days"] += 1
            else:
                metrics["issue_age_distribution"]["90+_days"] += 1
        
        # Labels
        for label in issue["labels"]["nodes"]:
            metrics["labels_distribution"][label["name"]] += 1
        
        # Authors
        if issue.get("author") and issue["author"].get("login"):
            metrics["authors"][issue["author"]["login"]] += 1
        
        # Comments
        comment_count = len(issue["comments"]["nodes"])
        metrics["comments_per_issue"].append(comment_count)
        
        # Commenters
        for comment in issue["comments"]["nodes"]:
            if comment.get("author") and comment["author"].get("login"):
                metrics["commenters"][comment["author"]["login"]] += 1
    
    # Calculate averages and medians
    if metrics["time_to_resolution"]:
        metrics["time_to_resolution_avg"] = statistics.mean(metrics["time_to_resolution"])
        metrics["time_to_resolution_median"] = statistics.median(metrics["time_to_resolution"])
    
    if metrics["comments_per_issue"]:
        metrics["comments_per_issue_avg"] = statistics.mean(metrics["comments_per_issue"])
    
    # Convert defaultdicts to regular dicts for JSON serialization
    metrics["issue_creation_timeline"] = dict(sorted(metrics["issue_creation_timeline"].items()))
    metrics["issue_closure_timeline"] = dict(sorted(metrics["issue_closure_timeline"].items()))
    metrics["labels_distribution"] = dict(metrics["labels_distribution"].most_common())
    metrics["authors"] = dict(metrics["authors"].most_common(10))
    metrics["commenters"] = dict(metrics["commenters"].most_common(10))
    
    return metrics

def analyze_issue_with_llm(issue, model):
    """Analyze a single issue using LLM to extract advanced metrics"""
    # Extract basic issue information
    issue_number = issue["number"]
    title = issue["title"]
    body = issue["body"]
    state = issue["state"]
    
    # Extract comments
    comments = [
        {
            "author": comment["author"]["login"] if comment.get("author") and comment["author"].get("login") else "Unknown",
            "body": comment["body"],
            "created_at": comment["createdAt"]
        }
        for comment in issue["comments"]["nodes"]
    ]
    
    # Prepare the prompt for the LLM
    prompt = f"""
You are analyzing a GitHub issue. Extract the following metrics from the issue:

ISSUE TITLE: {title}
ISSUE STATE: {state}
ISSUE DESCRIPTION:
{body}

COMMENTS:
{json.dumps(comments[:5], indent=2)}  # Limiting to first 5 comments to avoid token limits

Please analyze this issue and return a JSON object with the following fields:
1. "issue_type": One of ["bug", "feature_request", "question", "documentation", "other"]
2. "component": The main component or module this issue relates to (extract from context)
3. "resolution_type": One of ["code_change", "configuration_change", "documentation_update", "no_action_needed", "not_resolved", "other"]
4. "complexity": One of ["low", "medium", "high"] based on the issue description and discussion
5. "sentiment": One of ["positive", "neutral", "negative"] based on the tone of the issue and comments

Return ONLY the JSON object without any additional text or explanation.
"""

    # Call Ollama API
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 4096,
                    "temperature": 0.1
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            # Try to parse the LLM's response as JSON
            try:
                # Extract just the JSON part if there's any extra text
                response_text = result["response"]
                # Find JSON-like content (between curly braces)
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    return {
                        "issue_type": "other",
                        "component": "unknown",
                        "resolution_type": "other",
                        "complexity": "medium",
                        "sentiment": "neutral",
                        "error": "Could not extract JSON from response"
                    }
            except json.JSONDecodeError:
                return {
                    "issue_type": "other",
                    "component": "unknown",
                    "resolution_type": "other",
                    "complexity": "medium",
                    "sentiment": "neutral",
                    "error": "JSON decode error"
                }
        else:
            return {
                "issue_type": "other",
                "component": "unknown",
                "resolution_type": "other",
                "complexity": "medium",
                "sentiment": "neutral",
                "error": f"API error: {response.status_code}"
            }
    except Exception as e:
        return {
            "issue_type": "other",
            "component": "unknown",
            "resolution_type": "other",
            "complexity": "medium",
            "sentiment": "neutral",
            "error": f"Exception: {str(e)}"
        }

def extract_advanced_metrics(issues_data, model):
    """Extract advanced metrics from issues data using LLM"""
    advanced_metrics = {
        "issue_types": Counter(),
        "components": Counter(),
        "resolution_types": Counter(),
        "complexity": Counter(),
        "sentiment": Counter(),
        "issues_with_llm_analysis": 0,
        "issues_with_llm_errors": 0
    }
    
    print(f"\nAnalyzing {len(issues_data)} issues with LLM ({model})...")
    
    for issue in tqdm(issues_data, desc="Analyzing issues"):
        llm_analysis = analyze_issue_with_llm(issue, model)
        
        if "error" in llm_analysis and llm_analysis["error"]:
            advanced_metrics["issues_with_llm_errors"] += 1
            continue
        
        advanced_metrics["issues_with_llm_analysis"] += 1
        advanced_metrics["issue_types"][llm_analysis.get("issue_type", "other")] += 1
        advanced_metrics["components"][llm_analysis.get("component", "unknown")] += 1
        advanced_metrics["resolution_types"][llm_analysis.get("resolution_type", "other")] += 1
        advanced_metrics["complexity"][llm_analysis.get("complexity", "medium")] += 1
        advanced_metrics["sentiment"][llm_analysis.get("sentiment", "neutral")] += 1
    
    # Convert Counters to regular dicts for JSON serialization
    advanced_metrics["issue_types"] = dict(advanced_metrics["issue_types"])
    advanced_metrics["components"] = dict(advanced_metrics["components"].most_common(10))
    advanced_metrics["resolution_types"] = dict(advanced_metrics["resolution_types"])
    advanced_metrics["complexity"] = dict(advanced_metrics["complexity"])
    advanced_metrics["sentiment"] = dict(advanced_metrics["sentiment"])
    
    return advanced_metrics

def load_issues(exclude_users=None):
    """Load all downloaded issues from JSON files
    
    Args:
        exclude_users (list): List of usernames to exclude from analysis
    """
    if exclude_users is None:
        exclude_users = []
    
    issues_data = []
    excluded_count = 0
    
    # Find all issue JSON files
    issue_files = glob.glob(str(INPUT_DIR / "issue_*.json"))
    
    if not issue_files:
        print(f"No issue files found in {INPUT_DIR}")
        return []
    
    print(f"Found {len(issue_files)} issue files")
    
    for file_path in issue_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                issue_data = json.load(f)
                
                # Check if the issue creator should be excluded
                # The creator information is in the 'author' field, not 'user'
                creator = issue_data.get('author', {}).get('login', '')
                if creator and creator in exclude_users:
                    excluded_count += 1
                    continue
                
                issues_data.append(issue_data)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if excluded_count > 0:
        print(f"Excluded {excluded_count} issues created by: {', '.join(exclude_users)}")
    
    return issues_data

def main(model=None):
    """Main function to extract metrics from GitHub issues"""
    parser = argparse.ArgumentParser(description="Extract metrics from GitHub issues using LLM")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help=f"Ollama model to use (default: {OLLAMA_MODEL})")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-based analysis (faster)")
    parser.add_argument("--exclude-users", type=str, nargs="*", default=["spec-automation"], 
                        help="List of usernames to exclude from analysis (default: spec-automation)")
    parser.add_argument("--include-all-users", action="store_true", 
                        help="Include all users in analysis (overrides --exclude-users)")
    
    args = parser.parse_args()
    model = args.model if model is None else model
    
    # Determine which users to exclude
    exclude_users = []
    if not args.include_all_users:
        exclude_users = args.exclude_users
    
    print(f"\n{'='*80}")
    print(f"📊 GITHUB ISSUE METRICS EXTRACTOR")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    if exclude_users:
        print(f"Excluding issues created by: {', '.join(exclude_users)}")
    print(f"{'='*80}\n")
    
    # Setup directories
    setup_directories()
    
    # Load issues
    issues_data = load_issues(exclude_users=exclude_users)
    
    if not issues_data:
        print("No issues found. Please run the downloader first.")
        return 0
    
    print(f"Loaded {len(issues_data)} issues")
    
    # Extract basic metrics
    print("Extracting basic metrics...")
    basic_metrics = extract_basic_metrics(issues_data)
    
    # Extract advanced metrics using LLM
    advanced_metrics = {}
    if not args.skip_llm:
        advanced_metrics = extract_advanced_metrics(issues_data, model)
    
    # Combine metrics
    all_metrics = {
        "basic_metrics": basic_metrics,
        "advanced_metrics": advanced_metrics,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model_used": model if not args.skip_llm else None,
            "total_issues_analyzed": len(issues_data),
            "skip_llm": args.skip_llm,
            "excluded_users": exclude_users if exclude_users else []
        }
    }
    
    # Save metrics to JSON file
    output_file = OUTPUT_DIR / "github_issues_metrics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nMetrics saved to {output_file}")
    
    return len(issues_data)

if __name__ == "__main__":
    main()
