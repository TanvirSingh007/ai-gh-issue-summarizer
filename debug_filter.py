#!/usr/bin/env python3
"""
Debug script to check if issues are being properly filtered by user login
"""

import json
import glob
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "downloaded_issues" / "json"

def debug_user_filtering():
    """Check if issues are being properly filtered by user login"""
    # Find all issue JSON files
    issue_files = glob.glob(str(INPUT_DIR / "issue_*.json"))
    
    if not issue_files:
        print(f"No issue files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(issue_files)} issue files")
    
    # Count issues by creator
    creator_counts = {}
    spec_automation_issues = []
    
    for file_path in issue_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                issue_data = json.load(f)
                
                # Get the creator's login
                creator = issue_data.get('user', {}).get('login', 'unknown')
                
                # Count issues by creator
                if creator not in creator_counts:
                    creator_counts[creator] = 0
                creator_counts[creator] += 1
                
                # Save spec-automation issue numbers
                if creator == "spec-automation":
                    issue_number = issue_data.get('number', 'unknown')
                    spec_automation_issues.append(issue_number)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    # Print results
    print("\nIssues by creator:")
    for creator, count in sorted(creator_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {creator}: {count}")
    
    print(f"\nTotal spec-automation issues: {len(spec_automation_issues)}")
    if spec_automation_issues:
        print(f"Sample issue numbers: {spec_automation_issues[:5]}...")

if __name__ == "__main__":
    debug_user_filtering()
