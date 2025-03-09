#!/usr/bin/env python3
"""
GitHub Issue Analysis Tool

A comprehensive tool for downloading, summarizing, and analyzing GitHub issues.

This script serves as the main entry point for the GitHub Issue Analysis Tool,
providing a unified interface to access all functionality.

Usage:
    python github_issue_tool.py [command] [options]

Commands:
    download    Download GitHub issues
    summarize   Generate summaries for downloaded issues
    embed       Generate embeddings for issue summaries
    analyze     Analyze issue summaries using LLM
    
Run 'python github_issue_tool.py [command] --help' for more information on a command.
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"

# Ensure the Python path includes our source directories
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

def load_module(module_path, module_name):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def download_command(args):
    """Run the GitHub issue downloader."""
    downloader = load_module(SRC_DIR / "downloader" / "downloader.py", "downloader")
    # Update the output directory to use our data structure
    downloader.OUTPUT_DIR = str(DATA_DIR / "downloaded_issues")
    # Run the downloader
    downloader.main()

def summarize_command(args):
    """Run the issue summarizer."""
    summarizer = load_module(SRC_DIR / "summarizer" / "summarizer.py", "summarizer")
    # Update the input and output directories to use our data structure
    summarizer.INPUT_DIR = str(DATA_DIR / "downloaded_issues" / "json")
    summarizer.OUTPUT_DIR = str(DATA_DIR / "issue_summaries")
    # Run the summarizer with command line arguments
    summarizer.main()

def embed_command(args):
    """Generate embeddings for issue summaries."""
    embeddings = load_module(SRC_DIR / "analyzer" / "embeddings.py", "embeddings")
    
    # Explicitly set the paths to use our data structure
    embeddings.ISSUE_SUMMARIES_DIR = str(DATA_DIR / "issue_summaries")
    embeddings.PERSIST_DIRECTORY = str(DATA_DIR / "embeddings" / args.model)
    
    print(f"Using issue summaries from: {embeddings.ISSUE_SUMMARIES_DIR}")
    print(f"Saving embeddings to: {embeddings.PERSIST_DIRECTORY}")
    
    # Create a custom argv for the embeddings script
    sys.argv = [
        "embeddings.py",
        "--model", args.model,
        "--chunk-size", str(args.chunk_size),
        "--chunk-overlap", str(args.chunk_overlap),
        "--persist-dir", embeddings.PERSIST_DIRECTORY
    ]
    
    # Run the embeddings generator
    embeddings.main()

def analyze_command(args):
    """Analyze issue summaries using LLM."""
    analyzer = load_module(SRC_DIR / "analyzer" / "analyzer.py", "analyzer")
    
    # Explicitly set the paths to use our data structure
    analyzer.ISSUE_SUMMARIES_DIR = str(DATA_DIR / "issue_summaries")
    analyzer.PERSIST_DIRECTORY = str(DATA_DIR / "embeddings" / args.model)
    
    print(f"Using issue summaries from: {analyzer.ISSUE_SUMMARIES_DIR}")
    print(f"Using embeddings from: {analyzer.PERSIST_DIRECTORY}")
    
    # Create a custom argv for the analyzer script
    sys.argv = [
        "analyzer.py",
        "--model", args.model,
        "--persist-dir", analyzer.PERSIST_DIRECTORY
    ]
    
    if args.examples:
        sys.argv.append("--examples")
    
    # Run the analyzer
    analyzer.main()

def setup_parsers():
    """Set up the command-line argument parsers."""
    # Main parser
    parser = argparse.ArgumentParser(
        description="GitHub Issue Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python github_issue_tool.py download
  python github_issue_tool.py summarize
  python github_issue_tool.py embed --model mistral
  python github_issue_tool.py analyze --model mistral
  python github_issue_tool.py analyze --model mistral --examples
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download parser
    download_parser = subparsers.add_parser("download", help="Download GitHub issues")
    
    # Summarize parser
    summarize_parser = subparsers.add_parser("summarize", help="Generate summaries for downloaded issues")
    
    # Embed parser
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings for issue summaries")
    embed_parser.add_argument(
        "--model", 
        type=str, 
        default="mistral",
        help="Ollama model to use for embeddings (default: mistral)"
    )
    embed_parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000,
        help="Chunk size for text splitting (default: 1000)"
    )
    embed_parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=200,
        help="Chunk overlap for text splitting (default: 200)"
    )
    
    # Analyze parser
    analyze_parser = subparsers.add_parser("analyze", help="Analyze issue summaries using LLM")
    analyze_parser.add_argument(
        "--model", 
        type=str, 
        default="mistral",
        help="Ollama model to use for analysis (default: mistral)"
    )
    analyze_parser.add_argument(
        "--examples", 
        action="store_true",
        help="Run example queries instead of interactive mode"
    )
    
    return parser

def main():
    """Main entry point for the GitHub Issue Analysis Tool."""
    parser = setup_parsers()
    args = parser.parse_args()
    
    # Create necessary directories if they don't exist
    os.makedirs(DATA_DIR / "downloaded_issues", exist_ok=True)
    os.makedirs(DATA_DIR / "issue_summaries", exist_ok=True)
    os.makedirs(DATA_DIR / "embeddings", exist_ok=True)
    
    # Execute the appropriate command
    if args.command == "download":
        download_command(args)
    elif args.command == "summarize":
        summarize_command(args)
    elif args.command == "embed":
        embed_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
