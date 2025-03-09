#!/usr/bin/env python3
"""
GitHub Issue Analyzer

This script provides a chat interface for analyzing GitHub issue summaries using LangChain,
Mistral via Ollama, and a pre-generated Chroma vector store.

Usage:
    python github_issue_analyzer.py [--model MODEL]

Requirements:
    - Pre-generated embeddings in the Chroma database (run generate_embeddings.py first)
    - Mistral model running in Ollama (run: ollama run mistral)
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configuration
ISSUE_SUMMARIES_DIR = "data/issue_summaries"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
PERSIST_DIRECTORY = "data/embeddings/mistral"

# Example queries to analyze the GitHub issues
EXAMPLE_QUERIES = [
    "What are the most common themes in these issues?",
    "What is the sentiment of the issue summaries?",
    "Summarize all issues related to bugs.",
    "What features are most frequently requested?",
    "Identify any security concerns mentioned in the issues."
]

class GitHubIssueAnalyzer:
    """Provides a chat interface for analyzing GitHub issue summaries using LangChain and Ollama."""

    def __init__(self, 
                 model_name: str = OLLAMA_MODEL,
                 persist_directory: str = PERSIST_DIRECTORY):
        """Initialize the analyzer with the model name and persist directory."""
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        
        # Check if Ollama is running and embeddings exist
        self._check_ollama_connection()
        self._check_embeddings_exist()

    def _check_ollama_connection(self) -> None:
        """Check if Ollama is running with the required model."""
        try:
            # Create a test embedding to check connection
            embeddings = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=self.model_name
            )
            _ = embeddings.embed_query("Test connection")
            print(f"✅ Successfully connected to Ollama with model: {self.model_name}")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print(f"Make sure Ollama is running with the {self.model_name} model.")
            print(f"Run: ollama run {self.model_name}")
            sys.exit(1)
            
    def _check_embeddings_exist(self) -> None:
        """Check if embeddings exist in the persist directory."""
        if not os.path.exists(self.persist_directory):
            print(f"❌ Error: Embeddings directory not found at {self.persist_directory}")
            print("Please run generate_embeddings.py first to create embeddings.")
            print("Example: python generate_embeddings.py")
            sys.exit(1)
        print(f"✅ Found embeddings at {self.persist_directory}")

    def load_vectorstore(self) -> None:
        """Load the existing vector store from the persist directory."""
        print(f"🔍 Loading vector store from {self.persist_directory}...")
        try:
            embeddings = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=self.model_name
            )
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=embeddings
            )
            print(f"✅ Successfully loaded vector store with {self.vectorstore._collection.count()} documents")
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            sys.exit(1)

    def setup_qa_chain(self) -> None:
        """Set up the question-answering chain."""
        if not self.vectorstore:
            print("❌ No vector store available. Call create_vectorstore() first.")
            return
        
        print("🤖 Setting up QA chain with Mistral...")
        
        # Create the LLM
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=self.model_name,
            temperature=0.2
        )
        
        # Create a custom prompt template for better analysis
        template = """
        You are an AI assistant specialized in analyzing GitHub issues. 
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Your analysis:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 10}),
            prompt=prompt,
            return_source_documents=True
        )
        print("✅ QA chain setup complete")

    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze the GitHub issues based on the provided query."""
        if not self.qa_chain:
            print("❌ QA chain not set up. Call setup_qa_chain() first.")
            return {"error": "QA chain not initialized"}
        
        print(f"🔍 Analyzing: '{query}'")
        try:
            result = self.qa_chain.invoke({"query": query})
            return result
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return {"error": str(e)}

    def run_example_queries(self) -> None:
        """Run the example queries and print the results."""
        if not self.qa_chain:
            print("❌ QA chain not set up. Call setup_qa_chain() first.")
            return
        
        print("\n" + "="*80)
        print("📊 RUNNING EXAMPLE QUERIES")
        print("="*80)
        
        for i, query in enumerate(EXAMPLE_QUERIES, 1):
            print(f"\n📌 QUERY {i}: {query}")
            print("-"*80)
            
            result = self.analyze(query)
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                # Handle different response formats
                if isinstance(result, dict) and 'result' in result:
                    print(f"🔍 ANALYSIS:\n{result['result']}")
                elif isinstance(result, dict) and 'answer' in result:
                    print(f"🔍 ANALYSIS:\n{result['answer']}")
                else:
                    print(f"🔍 ANALYSIS:\n{result}")
            
            print("-"*80)

    def process_all(self) -> None:
        """Load the vector store and set up the QA chain."""
        self.load_vectorstore()
        self.setup_qa_chain()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GitHub Issue Analyzer using LangChain and Ollama")
    parser.add_argument(
        "--model", 
        type=str, 
        default=OLLAMA_MODEL,
        help=f"Ollama model to use (default: {OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--persist-dir", 
        type=str, 
        default=PERSIST_DIRECTORY,
        help=f"Directory containing the embeddings (default: {PERSIST_DIRECTORY})"
    )
    parser.add_argument(
        "--examples", 
        action="store_true",
        help="Run example queries instead of interactive mode"
    )
    return parser.parse_args()

def main():
    """Main function to run the GitHub issue analyzer."""
    # Parse command-line arguments
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("🚀 GITHUB ISSUE ANALYZER")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Embeddings directory: {args.persist_dir}")
    print("="*80)
    
    # Initialize analyzer with command-line arguments
    analyzer = GitHubIssueAnalyzer(
        model_name=args.model,
        persist_directory=args.persist_dir
    )
    analyzer.process_all()
    
    # Run example queries if requested
    if args.examples:
        analyzer.run_example_queries()
        return
    
    # Interactive mode
    while True:
        print("\n" + "-"*80)
        print("Enter a query to analyze the GitHub issues (or 'quit' to exit):")
        query = input("> ")
        
        if query.lower() in ('quit', 'exit', 'q'):
            break
        
        result = analyzer.analyze(query)
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            # Handle different response formats
            if isinstance(result, dict) and 'result' in result:
                print(f"\n🔍 ANALYSIS:\n{result['result']}")
            elif isinstance(result, dict) and 'answer' in result:
                print(f"\n🔍 ANALYSIS:\n{result['answer']}")
            else:
                print(f"\n🔍 ANALYSIS:\n{result}")


if __name__ == "__main__":
    main()
