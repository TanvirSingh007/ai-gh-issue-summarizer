# GitHub Issue Analysis Tool

A comprehensive tool for downloading, summarizing, and analyzing GitHub issues using Large Language Models.

## Project Overview

This project provides an end-to-end solution for working with GitHub issues:

1. **Download** - Fetch issues from GitHub repositories
2. **Summarize** - Generate concise summaries of issues using LLMs
3. **Analyze** - Create a searchable knowledge base to query issues using natural language
4. **Metrics** - Extract useful metrics from issues using LLMs
5. **Visualize** - View interactive dashboards of issue metrics

## Project Structure

```
github-issue-tool/
├── github_issue_tool.py      # Main entry point script
├── requirements.txt          # Project dependencies
├── src/                      # Source code
│   ├── downloader/           # Issue downloading module
│   │   └── downloader.py     # GitHub issue downloader script
│   ├── summarizer/           # Issue summarizing module
│   │   └── summarizer.py     # Issue summarizer script
│   ├── analyzer/             # Issue analysis module
│   │   ├── analyzer.py       # Issue analyzer script
│   │   ├── embeddings.py     # Embeddings generator script
│   │   └── metrics_extractor.py # Metrics extraction script
│   └── visualizer/           # Metrics visualization module
│       ├── app.py            # Flask web application
│       └── templates/        # HTML templates for visualization
│           └── index.html    # Dashboard template
└── data/                     # Data storage
    ├── downloaded_issues/    # Raw downloaded issues
    ├── issue_summaries/      # Generated issue summaries
    ├── metrics/              # Extracted metrics data
    └── embeddings/           # Vector embeddings for analysis
        ├── mistral/          # Embeddings using Mistral model
        └── ...               # Other model embeddings
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd github-issue-tool
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your GitHub Personal Access Token:
   - Create a Personal Access Token with `repo` scope at https://github.com/settings/tokens
   - Set it as an environment variable:
     ```bash
     export GITHUB_TOKEN='your_token_here'
     ```

4. Install Ollama and the required models:
   - Follow the installation instructions at [Ollama's website](https://ollama.ai/)
   - Pull the models you want to use:
     ```bash
     ollama pull mistral
     ollama pull llama3.2
     ```

## Usage

The project provides a unified command-line interface through the `github_issue_tool.py` script:

### 1. Download GitHub Issues

```bash
python github_issue_tool.py download
```

This will download issues from the configured GitHub repository and save them in both JSON and Markdown formats in the `data/downloaded_issues` directory.

### 2. Generate Issue Summaries

```bash
python github_issue_tool.py summarize
```

This will process the downloaded issues and generate concise summaries using the configured LLM. Summaries are saved in the `data/issue_summaries` directory.

### 3. Generate Embeddings

```bash
python github_issue_tool.py embed --model mistral
```

Options:
- `--model`: Ollama model to use for embeddings (default: mistral)
- `--chunk-size`: Chunk size for text splitting (default: 1000)
- `--chunk-overlap`: Chunk overlap for text splitting (default: 200)

This will process the issue summaries and generate vector embeddings for efficient searching. Embeddings are saved in the `data/embeddings/<model>` directory.

### 4. Analyze Issues

```bash
python github_issue_tool.py analyze --model mistral
```

Options:
- `--model`: Ollama model to use for analysis (default: mistral)
- `--examples`: Run example queries instead of interactive mode

This will start an interactive session where you can ask questions about the GitHub issues in natural language.

### 5. Extract Metrics

```bash
python github_issue_tool.py metrics --model llama3.2
```

Options:
- `--model`: Ollama model to use for metrics extraction (default: llama3.2)
- `--skip-llm`: Skip LLM-based analysis for faster processing (only extract basic metrics)

This will analyze the downloaded issues and extract useful metrics such as issue volume, resolution time, label distribution, and more. The metrics are saved as a JSON file in the `data/metrics` directory.

### 6. Visualize Metrics

```bash
python github_issue_tool.py visualize
```

Options:
- `--port`: Port to run the visualization server on (default: 5000)

This will start a web server that provides an interactive dashboard to visualize the extracted metrics. Open your browser and navigate to `http://localhost:5000` to view the dashboard.

## Configuration

Each module can be configured by modifying the respective script in the `src` directory:

- **Downloader**: Edit `src/downloader/downloader.py` to change the repository, labels, or other GitHub API parameters.
- **Summarizer**: Edit `src/summarizer/summarizer.py` to modify the summarization prompt or LLM parameters.
- **Analyzer**: Edit `src/analyzer/analyzer.py` or `src/analyzer/embeddings.py` to adjust the analysis parameters.

## Example Workflow

```bash
# 1. Download GitHub issues
python github_issue_tool.py download

# 2. Generate summaries
python github_issue_tool.py summarize

# 3. Generate embeddings with Mistral
python github_issue_tool.py embed --model mistral

# 4. Analyze issues with Mistral
python github_issue_tool.py analyze --model mistral

# 5. Extract metrics from issues
python github_issue_tool.py metrics --model llama3.2

# 6. Visualize metrics in a web dashboard
python github_issue_tool.py visualize
```

## Advanced Usage

### Using Different Models

You can use different models for embeddings and analysis:

```bash
# Generate embeddings with llama3.2
python github_issue_tool.py embed --model llama3.2

# Analyze using llama3.2
python github_issue_tool.py analyze --model llama3.2
```

### Running Example Queries

To run a set of predefined example queries:

```bash
python github_issue_tool.py analyze --model mistral --examples
```

## Troubleshooting

- **GitHub API Rate Limiting**: If you encounter rate limiting issues, the downloader will automatically wait and retry.
- **Ollama Connection Issues**: Make sure Ollama is running and the model is available. Run `ollama list` to check available models.
- **Embedding Dimension Mismatch**: If you get a dimension mismatch error, make sure you're using the same model for both generating and using embeddings.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
