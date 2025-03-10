# GitHub Issue Summarizer

This script processes GitHub issues downloaded by the `github_issues_downloader.py` script and generates detailed summaries using the Llama 3.2 model via Ollama.

## Prerequisites

1. You must have already run the `github_issues_downloader.py` script to download the GitHub issues
2. Ollama must be installed and running with the Llama 3.2 model
   - If you don't have Ollama installed, visit: https://ollama.com/
   - To install the Llama 3.2 model: `ollama pull llama3.2`
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The script uses environment variables from the `.env` file:
- `OLLAMA_API_URL`: URL for the Ollama API (default: http://localhost:11434/api/generate)
- `OLLAMA_MODEL`: Model to use for summarization (default: mistral)

## Features

For each GitHub issue, the script generates a summary that includes:

- Title
- Summary of what the issue is about
- Summary of conversations along with usernames
- Whether the ticket was accepted and fixed or closed and returned to CS
- If the issue was accepted, what fix was implemented
- Whether it was a code change
- If it was a code change, what changes were made and to what component
- If it was not a code change, what steps were taken to fix the issue
- Date when the issue was opened
- Date when it was closed
- GitHub link to the issue

Bot comments (including those from "taro" and other bots) are automatically filtered out.

## Usage

1. Make sure Ollama is running with the appropriate model
2. Run the script with options:
   ```bash
   # Process all issues
   python issue_summarizer.py
   
   # Process a specific number of issues
   python issue_summarizer.py --batch 5
   
   # Resume from where the script left off
   python issue_summarizer.py --resume
   
   # Resume and process a specific number of issues
   python issue_summarizer.py --resume --batch 20
   ```

## Output

The script creates a directory called `issue_summaries` containing a markdown file for each issue. Each markdown file includes:

- Issue details (number, state, dates, link)
- A comprehensive summary generated by the LLM

## Progress Tracking

The script saves progress in a file called `summarizer_progress.json`, which allows you to resume processing from where you left off. This is especially useful when processing a large number of issues.

You can safely interrupt the script with Ctrl+C, and your progress will be saved.

## Notes

- The script processes issues in batches with a fresh context for each issue
- Each issue is processed one at a time with a small delay
- The LLM runs locally on your machine, so no data is sent to external services
- Processing a large number of issues may take significant time depending on your hardware 