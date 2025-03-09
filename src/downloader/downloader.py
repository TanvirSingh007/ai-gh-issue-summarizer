import os
import json
import time
from datetime import datetime
import requests
import markdown

# Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # You'll need to set this environment variable
REPO_OWNER = "trilogy-group"
REPO_NAME = "eng-maintenance"
LABEL = "Product:AdvocateHub"
OUTPUT_DIR = "downloaded_issues"
ISSUES_PER_PAGE = 100  # Maximum allowed by GitHub API

# GraphQL query for fetching issues
ISSUES_QUERY = """
query ($owner: String!, $name: String!, $cursor: String, $label: String!, $perPage: Int!) {
  repository(owner: $owner, name: $name) {
    issues(first: $perPage, after: $cursor, labels: [$label], states: [CLOSED]) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        body
        state
        createdAt
        closedAt
        url
        labels(first: 10) {
          nodes {
            name
          }
        }
        author {
          login
        }
        comments(first: 100) {
          nodes {
            author {
              login
            }
            body
            createdAt
          }
        }
      }
    }
  }
}
"""

def setup_directories():
    """Create necessary directories for output"""
    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create subdirectories for JSON and Markdown files
    json_dir = os.path.join(OUTPUT_DIR, 'json')
    md_dir = os.path.join(OUTPUT_DIR, 'markdown')
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)
    
    print(f"Created directories:\n- {OUTPUT_DIR}\n- {json_dir}\n- {md_dir}")

def run_query(query, variables):
    """Execute a GraphQL query"""
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Content-Type': 'application/json',
    }
    
    response = requests.post(
        'https://api.github.com/graphql',
        json={'query': query, 'variables': variables},
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Query failed with status code: {response.status_code}")

def convert_to_markdown(issue):
    """Convert issue data to markdown format"""
    md = f"# Issue #{issue['number']}: {issue['title']}\n\n"
    
    # Add issue state prominently at the top
    state = issue['state']
    md += f"**State:** {state}\n"
    md += f"**Created by:** {issue['author']['login']} on {issue['createdAt']}\n"
    
    if state == "CLOSED":
        md += f"**Closed on:** {issue['closedAt']}\n"
    
    md += f"**URL:** {issue['url']}\n\n"
    
    # Add labels
    labels = [label['name'] for label in issue['labels']['nodes']]
    md += f"**Labels:** {', '.join(labels)}\n\n"
    
    # Add issue body
    md += "## Description\n\n"
    md += issue['body'] + "\n\n"
    
    # Add comments
    if issue['comments']['nodes']:
        md += "## Comments\n\n"
        for comment in issue['comments']['nodes']:
            md += f"### Comment by {comment['author']['login']} on {comment['createdAt']}\n\n"
            md += comment['body'] + "\n\n"
            md += "---\n\n"
    
    return md

def fetch_all_issues():
    """Fetch all issues and save them in both JSON and Markdown formats"""
    cursor = None
    all_issues = []
    page_count = 0
    total_issues = 0
    
    print(f"\n{'='*80}")
    print(f"📥 DOWNLOADING GITHUB ISSUES")
    print(f"{'='*80}")
    print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
    print(f"Label: {LABEL}")
    print(f"Issues per page: {ISSUES_PER_PAGE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    # Ensure directories exist
    setup_directories()
    
    while True:
        page_count += 1
        print(f"Fetching page {page_count}...")
        
        variables = {
            "owner": REPO_OWNER,
            "name": REPO_NAME,
            "cursor": cursor,
            "label": LABEL,
            "perPage": ISSUES_PER_PAGE
        }
        
        try:
            result = run_query(ISSUES_QUERY, variables)
            if "errors" in result:
                print(f"Error in response: {result['errors']}")
                break
                
            issues = result['data']['repository']['issues']
            current_batch = issues['nodes']
            batch_size = len(current_batch)
            all_issues.extend(current_batch)
            total_issues += batch_size
            
            print(f"Retrieved {batch_size} issues on page {page_count} (total: {total_issues})")
            
            # Save each issue individually
            for i, issue in enumerate(current_batch, 1):
                issue_number = issue['number']
                
                # Save JSON
                json_path = os.path.join(OUTPUT_DIR, 'json', f'issue_{issue_number}.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(issue, f, indent=2, ensure_ascii=False)
                
                # Save Markdown
                md_path = os.path.join(OUTPUT_DIR, 'markdown', f'issue_{issue_number}.md')
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(convert_to_markdown(issue))
                
                # Print progress every 10 issues or for the last one
                if i % 10 == 0 or i == batch_size:
                    print(f"Saved {i}/{batch_size} issues from page {page_count}")
            
            # Check if we've reached the end of pagination
            if not issues['pageInfo']['hasNextPage']:
                print(f"\nNo more pages to fetch. All issues downloaded.")
                break
                
            cursor = issues['pageInfo']['endCursor']
            
            # Respect rate limits - pause between requests to avoid hitting GitHub's rate limit
            print(f"Waiting 2 seconds before fetching next page...")
            time.sleep(2)
            
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}")
            print("Saving issues collected so far...")
            break
    
    # Save all issues in a single JSON file
    all_issues_path = os.path.join(OUTPUT_DIR, 'all_issues.json')
    with open(all_issues_path, 'w', encoding='utf-8') as f:
        json.dump(all_issues, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"Total issues downloaded: {total_issues}")
    print(f"Pages processed: {page_count}")
    print(f"JSON files saved to: {os.path.join(OUTPUT_DIR, 'json')}")
    print(f"Markdown files saved to: {os.path.join(OUTPUT_DIR, 'markdown')}")
    print(f"All issues saved to: {all_issues_path}")
    print(f"{'='*80}")
    
    return total_issues

def main(owner=None, repo=None, label=None, output_dir=None):
    """Main function to run the GitHub issues downloader.
    
    Args:
        owner (str, optional): GitHub repository owner. Defaults to None.
        repo (str, optional): GitHub repository name. Defaults to None.
        label (str, optional): Label to filter issues by. Defaults to None.
        output_dir (str, optional): Directory to save downloaded issues. Defaults to None.
    """
    global REPO_OWNER, REPO_NAME, LABEL, OUTPUT_DIR
    
    # Override default values if provided
    if owner:
        REPO_OWNER = owner
    if repo:
        REPO_NAME = repo
    if label:
        LABEL = label
    if output_dir:
        OUTPUT_DIR = output_dir
    
    if not GITHUB_TOKEN:
        print("\n❌ Error: GitHub token not found. Please set the GITHUB_TOKEN environment variable.")
        print("You can create a token at https://github.com/settings/tokens")
        return 0
    
    # Setup directories is now called inside fetch_all_issues
    total_issues = fetch_all_issues()
    return total_issues

if __name__ == "__main__":
    main() 