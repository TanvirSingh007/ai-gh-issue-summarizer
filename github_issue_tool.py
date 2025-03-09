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
    
    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    # Set up the parameters for the downloader
    owner = args.owner if hasattr(args, 'owner') and args.owner else downloader.REPO_OWNER
    repo = args.repo if hasattr(args, 'repo') and args.repo else downloader.REPO_NAME
    label = args.label if hasattr(args, 'label') and args.label else downloader.LABEL
    output_dir = str(DATA_DIR / "downloaded_issues")
    
    print(f"\nDownloading GitHub issues from {owner}/{repo} with label '{label}'")
    print(f"Output directory: {output_dir}\n")
    
    # Run the downloader with our parameters
    issue_count = downloader.main(owner=owner, repo=repo, label=label, output_dir=output_dir)
    
    if issue_count > 0:
        print(f"\nDownload completed successfully. You can now run the summarizer:")
        print(f"  python github_issue_tool.py summarize")
    else:
        print(f"\nNo issues were downloaded. Please check your parameters and try again.")

def summarize_command(args):
    """Run the issue summarizer."""
    summarizer = load_module(SRC_DIR / "summarizer" / "summarizer.py", "summarizer")
    
    # Create the data directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "issue_summaries").mkdir(exist_ok=True)
    
    # Update the input and output directories to use our data structure
    summarizer.INPUT_DIR = str(DATA_DIR / "downloaded_issues" / "json")
    summarizer.OUTPUT_DIR = str(DATA_DIR / "issue_summaries")
    
    print(f"\nSummarizing issues using model: {args.model}")
    print(f"Reading issues from: {summarizer.INPUT_DIR}")
    print(f"Saving summaries to: {summarizer.OUTPUT_DIR}\n")
    
    # Create a custom argv for the summarizer script
    sys.argv = [
        "summarizer.py",
        "--model", args.model
    ]
    
    # Add optional arguments if provided
    if hasattr(args, 'batch') and args.batch is not None:
        sys.argv.extend(["--batch", str(args.batch)])
    
    if hasattr(args, 'resume') and args.resume:
        sys.argv.append("--resume")
    
    # Run the summarizer with command line arguments
    summarizer.main(model=args.model)

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
  python github_issue_tool.py download --owner "owner" --repo "repo" --label "bug"
  python github_issue_tool.py summarize --model llama3.2
  python github_issue_tool.py summarize --model mistral --resume
  python github_issue_tool.py embed --model mistral
  python github_issue_tool.py analyze --model mistral
  python github_issue_tool.py analyze --model mistral --examples
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download parser
    download_parser = subparsers.add_parser("download", help="Download GitHub issues")
    download_parser.add_argument(
        "--owner",
        type=str,
        help="GitHub repository owner (default: trilogy-group)"
    )
    download_parser.add_argument(
        "--repo",
        type=str,
        help="GitHub repository name (default: eng-maintenance)"
    )
    download_parser.add_argument(
        "--label",
        type=str,
        help="Label to filter issues by (default: Product:AdvocateHub)")
    
    # Summarize parser
    summarize_parser = subparsers.add_parser("summarize", help="Generate summaries for downloaded issues")
    summarize_parser.add_argument(
        "--model", 
        type=str, 
        default="llama3.2",
        help="Ollama model to use for summarization (default: llama3.2)"
    )
    summarize_parser.add_argument(
        "--batch", 
        type=int, 
        help="Number of issues to process in this batch (default: all issues)"
    )
    summarize_parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume from where the script left off"
    )
    
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
