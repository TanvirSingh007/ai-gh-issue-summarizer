#!/usr/bin/env python3
"""
GitHub Issue Analyzer

This script analyzes GitHub issue summaries stored as Markdown files using LangChain,
DeepSeek R1 via Ollama, and Chroma vector store.

Usage:
    python github_issue_analyzer.py

Requirements:
    - "issue_summaries" folder with Markdown files in the same directory
    - DeepSeek R1 model running in Ollama (run: ollama run deepseek-r1)
"""

import os
import sys
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configuration
ISSUE_SUMMARIES_DIR = "issue_summaries"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OLLAMA_MODEL = "deepseek-r1"
OLLAMA_BASE_URL = "http://localhost:11434"
PERSIST_DIRECTORY = "chroma_db"

# Example queries to analyze the GitHub issues
EXAMPLE_QUERIES = [
    "What are the most common themes in these issues?",
    "What is the sentiment of the issue summaries?",
    "Summarize all issues related to bugs.",
    "What features are most frequently requested?",
    "Identify any security concerns mentioned in the issues."
]

class GitHubIssueAnalyzer:
    """Analyzes GitHub issue summaries using LangChain and Ollama."""

    def __init__(self, issue_dir: str = ISSUE_SUMMARIES_DIR):
        """Initialize the analyzer with the directory containing issue summaries."""
        self.issue_dir = issue_dir
        self.documents = None
        self.chunks = None
        self.vectorstore = None
        self.qa_chain = None
        
        # Check if Ollama is running
        self._check_ollama_connection()

    def _check_ollama_connection(self) -> None:
        """Check if Ollama is running with the required model."""
        try:
            # Create a test embedding to check connection
            embeddings = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL
            )
            _ = embeddings.embed_query("Test connection")
            print(f"✅ Successfully connected to Ollama with model: {OLLAMA_MODEL}")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print(f"Make sure Ollama is running with the {OLLAMA_MODEL} model.")
            print(f"Run: ollama run {OLLAMA_MODEL}")
            sys.exit(1)

    def load_documents(self) -> None:
        """Load all Markdown files from the issue summaries directory."""
        try:
            if not os.path.exists(self.issue_dir):
                raise FileNotFoundError(f"Directory not found: {self.issue_dir}")
            
            # Count the number of Markdown files
            md_files = [f for f in os.listdir(self.issue_dir) if f.endswith('.md')]
            if not md_files:
                raise FileNotFoundError(f"No Markdown files found in {self.issue_dir}")
            
            print(f"📂 Loading {len(md_files)} Markdown files from {self.issue_dir}...")
            
            # Use TextLoader to load each Markdown file
            self.documents = []
            for i, filename in enumerate(md_files):
                if i % 50 == 0:
                    print(f"  Progress: {i}/{len(md_files)} files loaded")
                try:
                    file_path = os.path.join(self.issue_dir, filename)
                    loader = TextLoader(file_path)
                    self.documents.extend(loader.load())
                except Exception as file_error:
                    print(f"  Warning: Could not load {filename}: {file_error}")
            
            print(f"✅ Loaded {len(self.documents)} documents")
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            sys.exit(1)

    def split_documents(self) -> None:
        """Split documents into chunks for processing."""
        if not self.documents:
            print("❌ No documents loaded. Call load_documents() first.")
            return
        
        print(f"📄 Splitting documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"✅ Created {len(self.chunks)} chunks")

    def create_vectorstore(self) -> None:
        """Create a vector store from the document chunks."""
        if not self.chunks:
            print("❌ No document chunks available. Call split_documents() first.")
            return
        
        print("🔍 Creating vector store with DeepSeek R1 embeddings...")
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL
        )
        
        # Create a custom wrapper for Chroma.from_documents to show progress
        print("Preparing to generate embeddings...")
        total_chunks = len(self.chunks)
        print(f"Total chunks to process: {total_chunks}")
        
        # Process in smaller batches with a progress bar
        batch_size = 10
        batches = [self.chunks[i:i+batch_size] for i in range(0, total_chunks, batch_size)]
        
        # Create an empty vector store first
        if os.path.exists(PERSIST_DIRECTORY):
            print(f"Using existing vector store at {PERSIST_DIRECTORY}")
            self.vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        else:
            print(f"Creating new vector store at {PERSIST_DIRECTORY}")
            self.vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            
        # Add documents in batches with progress bar
        print("Adding documents to vector store with progress tracking:")
        for i, batch in enumerate(tqdm(batches, desc="Embedding documents")):
            self.vectorstore.add_documents(documents=batch)
            if (i + 1) % 5 == 0 or (i + 1) == len(batches):
                # Persist every 5 batches or on the last batch
                self.vectorstore.persist()
                print(f"Progress: {min((i+1)*batch_size, total_chunks)}/{total_chunks} chunks processed")
        
        print(f"✅ Vector store created and persisted to {PERSIST_DIRECTORY}")

    def setup_qa_chain(self) -> None:
        """Set up the question-answering chain."""
        if not self.vectorstore:
            print("❌ No vector store available. Call create_vectorstore() first.")
            return
        
        print("🤖 Setting up QA chain with DeepSeek R1...")
        
        # Create the LLM
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
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
        """Run the complete analysis pipeline."""
        self.load_documents()
        self.split_documents()
        self.create_vectorstore()
        self.setup_qa_chain()
        self.run_example_queries()


def main():
    """Main function to run the GitHub issue analyzer."""
    print("\n" + "="*80)
    print("🚀 GITHUB ISSUE ANALYZER")
    print("="*80)
    
    analyzer = GitHubIssueAnalyzer()
    analyzer.process_all()
    
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
