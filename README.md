# GitHub Issues Downloader

This script downloads GitHub issues and their comments from a specified repository, saving them in both JSON and Markdown formats. It's specifically configured to download closed issues with the label "Product:AdvocateHub" from the trilogy-group/eng-maintenance repository.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your GitHub Personal Access Token:
   - Create a Personal Access Token with `repo` scope at https://github.com/settings/tokens
   - Set it as an environment variable:
     ```bash
     export GITHUB_TOKEN='your_token_here'
     ```

## Usage

Simply run the script:
```bash
python github_issues_downloader.py
```

The script will:
1. Create a `downloaded_issues` directory
2. Download all matching issues
3. Save each issue as individual JSON and Markdown files
4. Create a consolidated `all_issues.json` file with all issues

## Output Structure

```
downloaded_issues/
├── json/
│   ├── issue_1.json
│   ├── issue_2.json
│   └── ...
├── markdown/
│   ├── issue_1.md
│   ├── issue_2.md
│   └── ...
└── all_issues.json
```

## Features

- Uses GitHub's GraphQL API for efficient bulk downloading
- Handles pagination automatically
- Respects GitHub's rate limits
- Saves both JSON (for programmatic analysis) and Markdown (for human reading) formats
- Includes all issue metadata, comments, and labels 