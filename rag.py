import json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAI

import os

class RAG:
    def __init__(self, json_path: str = 'data/total.json'):
        self.data = self.load_json(json_path)
        # Using a smaller, faster embedding model for demonstration
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.llm = OpenAI(temperature=0)
        
    def load_json(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_documents(self):
        docs = []
        for raw in self.data:
            filepath = raw.get("content_file")
            loader = TextLoader(filepath)
            res = loader.load()
            if len(res) < 1:
                print(f"Unable to load doc: {filepath}")
                continue
            doc = res[0]
            # Add the metadata to the Document
            doc.metadata.update(raw)
            docs.append(doc)
        return docs
        

    def prepare_documents(self) -> List[Dict[str, str]]:
        """Prepare documents for vectorization"""
        documents = []
        
        for article in self.raw_data:
            # Create a rich context combining all relevant information
            content = f"""
Title: {article['title']}
Date: {article['date']}
Author: {article['author']}
Categories: {', '.join(article['categories'])}

Content:
{article['content']}

Summary:
{article['excerpt']}

Source URL: {article['url']}
            """
            
            # Add metadata for better retrieval and citation
            metadata = {
                'title': article['title'],
                'date': article['date'],
                'author': article['author'],
                'url': article['url'],
                'categories': article['categories']
            }
            
            documents.append({
                'content': content,
                'metadata': metadata
            })
        
        return documents
    
    def split_documents(self, documents):
        """Split documents into smaller chunks for better retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

    def create_vector_store(self) -> Chroma:
        """Create and populate the vector store"""
        persist_dir = "./vectorstore"
        
        # Check if vector store already exists
        if os.path.exists(persist_dir):
            print("Loading existing vector store...")
            return Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        
        print("Creating new vector store...")
        documents = self.load_documents()
        # Split documents into chunks 
        chunks = self.split_documents(documents)

        print(f"Creating embeddings for {len(chunks)} chunks...")
        # Create and persist the vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )

        print(f"Vector database created and saved to {persist_dir}")
        return vector_store
    
    def format_docs(self, docs):
        """Format documents into a clean context string"""
        return "\n\n".join([
            f"Article: {doc.metadata['title']}\n"
            f"Date: {doc.metadata['date']}\n"
            f"URL: {doc.metadata['url']}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        ])

    def setup_qa_chain(self, vector_store: Chroma) -> RetrievalQA:
        """Set up the question-answering chain"""
        # Create a custom prompt template
        prompt_template = """You are an expert on African technology companies and startups.
Use the following articles to answer the question. If you don't know or aren't sure, say so.
Always cite your sources using the provided URLs.

Articles:
{context}

Question: {question}

Answer with facts from the provided articles, citing sources using URLs when possible:"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True
            },
            return_source_documents=True
        )
        
        return qa_chain
    
    def initialize_system(self):
        """Initialize the complete RAG system"""
        print("Creating vector store...")
        vector_store = self.create_vector_store()
        
        print("Setting up QA chain...")
        qa_chain = self.setup_qa_chain(vector_store)
        
        return qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the system with a question"""
        if not hasattr(self, 'qa_chain'):
            self.qa_chain = self.initialize_system()
        
        result = self.qa_chain({"query": question})
        
        # Format the response with sources
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "title": doc.metadata["title"],
                "date": doc.metadata["date"],
                "url": doc.metadata["url"]
            })
        
        return {
            "answer": result["result"],
            "sources": sources
        }

def main():
    # Initialize the RAG system
    rag = RAG('data/total.json')
    # rag.load_documents()
    # Example queries
    questions = [
        # "What happened with Marasoft fraud in 2025?",
        "What is an example of a fraud case in Kenya?",
        "What is CIG Motors?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source['title']} ({source['date']})")
            print(f"  URL: {source['url']}")

if __name__ == "__main__":
    main()
