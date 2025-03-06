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

# GraphQL query for fetching issues
ISSUES_QUERY = """
query ($owner: String!, $name: String!, $cursor: String, $label: String!) {
  repository(owner: $owner, name: $name) {
    issues(first: 100, after: $cursor, labels: [$label], states: [CLOSED]) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        body
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
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        os.makedirs(os.path.join(OUTPUT_DIR, 'json'))
        os.makedirs(os.path.join(OUTPUT_DIR, 'markdown'))

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
    md += f"**Created by:** {issue['author']['login']} on {issue['createdAt']}\n"
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
    
    while True:
        variables = {
            "owner": REPO_OWNER,
            "name": REPO_NAME,
            "cursor": cursor,
            "label": LABEL
        }
        
        try:
            result = run_query(ISSUES_QUERY, variables)
            if "errors" in result:
                print(f"Error in response: {result['errors']}")
                break
                
            issues = result['data']['repository']['issues']
            current_batch = issues['nodes']
            all_issues.extend(current_batch)
            
            # Save each issue individually
            for issue in current_batch:
                issue_number = issue['number']
                
                # Save JSON
                json_path = os.path.join(OUTPUT_DIR, 'json', f'issue_{issue_number}.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(issue, f, indent=2, ensure_ascii=False)
                
                # Save Markdown
                md_path = os.path.join(OUTPUT_DIR, 'markdown', f'issue_{issue_number}.md')
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(convert_to_markdown(issue))
                
                print(f"Saved issue #{issue_number}")
            
            if not issues['pageInfo']['hasNextPage']:
                break
                
            cursor = issues['pageInfo']['endCursor']
            
            # Respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            break
    
    # Save all issues in a single JSON file
    with open(os.path.join(OUTPUT_DIR, 'all_issues.json'), 'w', encoding='utf-8') as f:
        json.dump(all_issues, f, indent=2, ensure_ascii=False)
    
    return len(all_issues)

def main():
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable is not set")
        return
    
    setup_directories()
    print("Starting to fetch issues...")
    
    total_issues = fetch_all_issues()
    print(f"\nCompleted! Downloaded {total_issues} issues.")
    print(f"Files are saved in the '{OUTPUT_DIR}' directory")

if __name__ == "__main__":
    main() 