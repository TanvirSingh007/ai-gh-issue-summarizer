#!/usr/bin/env python3
"""
GitHub Issue Embeddings Generator

This script processes GitHub issue summaries stored as Markdown files and generates
embeddings using Ollama models. The embeddings are stored in a Chroma database
for later use by the github_issue_analyzer.py script.

Usage:
    python generate_embeddings.py [--model MODEL] [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP]

Requirements:
    - "issue_summaries" folder with Markdown files in the same directory
    - Ollama running with the specified model
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration
ISSUE_SUMMARIES_DIR = "data/issue_summaries"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
PERSIST_DIRECTORY = "data/embeddings/mistral"

class EmbeddingsGenerator:
    """Generates embeddings for GitHub issue summaries using LangChain and Ollama."""

    def __init__(self, 
                 issue_dir: str = ISSUE_SUMMARIES_DIR,
                 model_name: str = OLLAMA_MODEL,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 persist_directory: str = PERSIST_DIRECTORY):
        """Initialize the embeddings generator."""
        self.issue_dir = issue_dir
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.documents = None
        self.chunks = None
        self.vectorstore = None
        
        # Check if Ollama is running
        self._check_ollama_connection()

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
        
        print(f"📄 Splitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"✅ Created {len(self.chunks)} chunks")

    def create_vectorstore(self) -> None:
        """Create a vector store from the document chunks."""
        if not self.chunks:
            print("❌ No document chunks available. Call split_documents() first.")
            return
        
        print(f"🔍 Creating vector store with {self.model_name} embeddings...")
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=self.model_name
        )
        
        # Create a custom wrapper for Chroma.from_documents to show progress
        print("Preparing to generate embeddings...")
        total_chunks = len(self.chunks)
        print(f"Total chunks to process: {total_chunks}")
        
        # Process in smaller batches with a progress bar
        batch_size = 10
        batches = [self.chunks[i:i+batch_size] for i in range(0, total_chunks, batch_size)]
        
        # Create an empty vector store first
        if os.path.exists(self.persist_directory):
            print(f"Using existing vector store at {self.persist_directory}")
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        else:
            print(f"Creating new vector store at {self.persist_directory}")
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
            
        # Add documents in batches with progress bar
        print("Adding documents to vector store with progress tracking:")
        for i, batch in enumerate(tqdm(batches, desc="Embedding documents")):
            self.vectorstore.add_documents(documents=batch)
            if (i + 1) % 5 == 0 or (i + 1) == len(batches):
                # Persist every 5 batches or on the last batch
                self.vectorstore.persist()
                print(f"Progress: {min((i+1)*batch_size, total_chunks)}/{total_chunks} chunks processed")
        
        print(f"✅ Vector store created and persisted to {self.persist_directory}")
        print(f"You can now use github_issue_analyzer.py to chat with the data without reprocessing.")

    def generate_all(self) -> None:
        """Run the full embedding generation pipeline."""
        self.load_documents()
        self.split_documents()
        self.create_vectorstore()
        print("✅ Embedding generation complete!")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GitHub Issue Embeddings Generator")
    parser.add_argument(
        "--model", 
        type=str, 
        default=OLLAMA_MODEL,
        help=f"Ollama model to use for embeddings (default: {OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=CHUNK_SIZE,
        help=f"Chunk size for text splitting (default: {CHUNK_SIZE})"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=CHUNK_OVERLAP,
        help=f"Chunk overlap for text splitting (default: {CHUNK_OVERLAP})"
    )
    parser.add_argument(
        "--persist-dir", 
        type=str, 
        default=PERSIST_DIRECTORY,
        help=f"Directory to persist embeddings (default: {PERSIST_DIRECTORY})"
    )
    return parser.parse_args()


def main():
    """Main function to run the embeddings generator."""
    # Parse command-line arguments
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("🚀 GITHUB ISSUE EMBEDDINGS GENERATOR")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Persist directory: {args.persist_dir}")
    print("="*80)
    
    # Initialize generator with command-line arguments
    generator = EmbeddingsGenerator(
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_directory=args.persist_dir
    )
    generator.generate_all()


if __name__ == "__main__":
    main()
