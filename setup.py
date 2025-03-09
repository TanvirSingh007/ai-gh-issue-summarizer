#!/usr/bin/env python3
"""
Setup script for GitHub Issue Analysis Tool.

This script sets up the project structure and prepares it for use.
"""

import os
import shutil
from pathlib import Path
import sys

def setup_project():
    """Set up the project structure."""
    base_dir = Path(__file__).resolve().parent
    
    # Create directory structure if it doesn't exist
    directories = [
        "src/downloader",
        "src/summarizer",
        "src/analyzer",
        "data/downloaded_issues",
        "data/issue_summaries",
        "data/embeddings"
    ]
    
    for directory in directories:
        os.makedirs(base_dir / directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Move existing files to their new locations if they exist
    file_mappings = [
        ("github_issues_downloader.py", "src/downloader/downloader.py"),
        ("issue_summarizer.py", "src/summarizer/summarizer.py"),
        ("github_issue_analyzer.py", "src/analyzer/analyzer.py"),
        ("generate_embeddings.py", "src/analyzer/embeddings.py")
    ]
    
    for source, destination in file_mappings:
        source_path = base_dir / source
        dest_path = base_dir / destination
        
        if source_path.exists() and not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            print(f"✅ Copied {source} to {destination}")
    
    # Update README
    readme_path = base_dir / "README.md"
    new_readme_path = base_dir / "README_new.md"
    
    if new_readme_path.exists():
        if readme_path.exists():
            shutil.move(readme_path, base_dir / "README_old.md")
            print("✅ Backed up old README.md to README_old.md")
        
        shutil.move(new_readme_path, readme_path)
        print("✅ Updated README.md")
    
    print("\n" + "="*80)
    print("GitHub Issue Analysis Tool Setup Complete!")
    print("="*80)
    print("\nYou can now use the tool with the following commands:")
    print("  python github_issue_tool.py download    # Download GitHub issues")
    print("  python github_issue_tool.py summarize   # Generate summaries")
    print("  python github_issue_tool.py embed       # Generate embeddings")
    print("  python github_issue_tool.py analyze     # Analyze issues")
    print("\nSee README.md for more details.")

if __name__ == "__main__":
    setup_project()
