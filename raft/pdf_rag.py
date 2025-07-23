import argparse
import os
from typing import List
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline
)


class PDFRAGSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", qa_model_name="deepset/roberta-base-squad2"):
        """Initialize the RAG system with embedding and QA models"""
        print("Loading models...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)
        self.chunks = []
        self.embeddings = None
        self.index = None
        print("Models loaded successfully!")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        print(f"Extracting text from {pdf_path}...")
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} text chunks")
        return chunks
    
    def build_index(self, pdf_path: str):
        """Build FAISS index from PDF content"""
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split into chunks
        self.chunks = self.chunk_text(text)
        
        if not self.chunks:
            raise ValueError("No text chunks found in PDF")
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.embedding_model.encode(self.chunks)
        
        # Build FAISS index
        print("Building search index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Index built with {len(self.chunks)} chunks")
    
    def search_relevant_chunks(self, question: str, top_k: int = 3) -> List[str]:
        """Search for relevant text chunks given a question"""
        if self.index is None:
            raise ValueError("Index not built. Please call build_index() first.")
        
        # Encode the question
        question_embedding = self.embedding_model.encode([question])
        
        # Search for similar chunks
        scores, indices = self.index.search(question_embedding.astype('float32'), top_k)
        
        # Return relevant chunks
        relevant_chunks = [self.chunks[idx] for idx in indices[0]]
        return relevant_chunks
    
    def answer_question(self, question: str, top_k: int = 3) -> dict:
        """Answer a question using RAG"""
        # Find relevant chunks
        relevant_chunks = self.search_relevant_chunks(question, top_k)
        
        # Combine chunks as context
        context = " ".join(relevant_chunks)
        
        # Use QA model to get answer
        result = self.qa_pipeline(question=question, context=context)
        
        return {
            "question": question,
            "answer": result["answer"],
            "confidence": result["score"],
            "context_used": relevant_chunks
        }


def main():
    parser = argparse.ArgumentParser(description="PDF RAG System - Ask questions about any PDF")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to PDF file")
    parser.add_argument("--question", type=str, help="Question to ask (optional, will start interactive mode if not provided)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file {args.pdf_path} not found!")
        return
    
    # Initialize RAG system
    rag_system = PDFRAGSystem()
    
    # Build index from PDF
    try:
        rag_system.build_index(args.pdf_path)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return
    
    if args.question:
        # Single question mode
        result = rag_system.answer_question(args.question)
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"\nContext used:")
        for i, chunk in enumerate(result['context_used']):
            print(f"{i+1}. {chunk[:200]}...")
    else:
        # Interactive mode
        print(f"\nPDF loaded: {args.pdf_path}")
        print("Interactive Q&A mode. Type 'quit' to exit.")
        print("=" * 50)
        
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            try:
                result = rag_system.answer_question(question)
                print(f"\nAnswer: {result['answer']}")
                print(f"Confidence: {result['confidence']:.3f}")
                
                # Optionally show context
                show_context = input("Show context? (y/n): ").lower().startswith('y')
                if show_context:
                    print("\nRelevant context:")
                    for i, chunk in enumerate(result['context_used']):
                        print(f"{i+1}. {chunk[:300]}...")
                        print("-" * 40)
                        
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
