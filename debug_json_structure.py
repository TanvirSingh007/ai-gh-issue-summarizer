#!/usr/bin/env python3
"""
Debug script to examine the structure of issue JSON files
"""

import json
import glob
import os
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "downloaded_issues" / "json"

def examine_json_structure():
    """Examine the structure of issue JSON files"""
    # Find all issue JSON files
    issue_files = glob.glob(str(INPUT_DIR / "issue_*.json"))
    
    if not issue_files:
        print(f"No issue files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(issue_files)} issue files")
    
    # Examine the first file
    sample_file = issue_files[0]
    print(f"\nExamining file: {os.path.basename(sample_file)}")
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            issue_data = json.load(f)
            
            # Print the top-level keys
            print("\nTop-level keys:")
            for key in issue_data.keys():
                print(f"  - {key}")
            
            # Check if 'user' exists and its structure
            if 'user' in issue_data:
                print("\nUser field structure:")
                user_data = issue_data['user']
                if isinstance(user_data, dict):
                    for key, value in user_data.items():
                        print(f"  - {key}: {type(value).__name__}")
                    
                    # Check for login field
                    if 'login' in user_data:
                        print(f"\nSample login value: {user_data['login']}")
                else:
                    print(f"  User field is not a dictionary: {type(user_data).__name__}")
            else:
                print("\nNo 'user' field found in the issue data")
            
            # Check for creator information in other fields
            print("\nSearching for creator information in other fields:")
            for key in issue_data.keys():
                if isinstance(issue_data[key], dict) and 'login' in issue_data[key]:
                    print(f"  - Found 'login' in '{key}' field: {issue_data[key]['login']}")
    
    except Exception as e:
        print(f"Error examining {sample_file}: {str(e)}")

if __name__ == "__main__":
    examine_json_structure()
