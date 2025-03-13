#!/usr/bin/env python3
"""
GitHub Issue Metrics Visualizer

A Flask web application to visualize metrics extracted from GitHub issues.
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory

# Set the correct template folder path
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Default configuration
DEFAULT_PORT = 5000
DEFAULT_METRICS_FILE = str(Path(__file__).resolve().parent.parent.parent / "data" / "metrics" / "github_issues_metrics.json")

# Global variable to store metrics data
metrics_data = None

def load_metrics(metrics_file):
    """Load metrics data from JSON file"""
    global metrics_data
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        print(f"Loaded metrics data from {metrics_file}")
        return True
    except Exception as e:
        print(f"Error loading metrics data: {str(e)}")
        return False

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/metrics')
def get_metrics():
    """API endpoint to get all metrics data"""
    if metrics_data:
        return jsonify(metrics_data)
    else:
        return jsonify({"error": "No metrics data loaded"}), 404

@app.route('/api/metrics/basic')
def get_basic_metrics():
    """API endpoint to get basic metrics data"""
    if metrics_data and 'basic_metrics' in metrics_data:
        return jsonify(metrics_data['basic_metrics'])
    else:
        return jsonify({"error": "No basic metrics data available"}), 404

@app.route('/api/metrics/advanced')
def get_advanced_metrics():
    """API endpoint to get advanced metrics data"""
    if metrics_data and 'advanced_metrics' in metrics_data:
        return jsonify(metrics_data['advanced_metrics'])
    else:
        return jsonify({"error": "No advanced metrics data available"}), 404

@app.route('/api/metrics/metadata')
def get_metadata():
    """API endpoint to get metrics metadata"""
    if metrics_data and 'metadata' in metrics_data:
        return jsonify(metrics_data['metadata'])
    else:
        return jsonify({"error": "No metadata available"}), 404

def run_server(port=None, metrics_file=None):
    """Run the Flask server"""
    # Set default values if not provided
    port = port or DEFAULT_PORT
    metrics_file = metrics_file or DEFAULT_METRICS_FILE
    
    # Create templates and static directories if they don't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    # Load metrics data
    if not load_metrics(metrics_file):
        print(f"Warning: Could not load metrics data from {metrics_file}")
        print("The visualization will not show any data until metrics are extracted.")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Issue Metrics Visualizer")
    parser.add_argument(
        "--port", 
        type=int, 
        default=DEFAULT_PORT,
        help=f"Port to run the server on (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--metrics-file", 
        type=str, 
        default=DEFAULT_METRICS_FILE,
        help="Path to the metrics JSON file"
    )
    
    args = parser.parse_args()
    run_server(port=args.port, metrics_file=args.metrics_file)
