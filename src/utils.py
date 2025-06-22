
import json
import os
import re
import logging
from typing import List

def clean_text(text):
    """Clean and normalize text for better LLM processing"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Clean up common web artifacts
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'\t+', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', '---', text)
    
    return text.strip()

def save_docs_to_json(docs, filename):
    """Save documents to JSON with cleaned content"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    cleaned_docs = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content and len(cleaned_content) > 50:  # Only save meaningful content
            cleaned_docs.append({
                "page_content": cleaned_content,
                "metadata": doc.metadata,
                "content_length": len(cleaned_content)
            })
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cleaned_docs, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved {len(cleaned_docs)} cleaned chunks to {filename}")
        return len(cleaned_docs)
    except Exception as e:
        logging.error(f"Failed to save data to {filename}: {e}")
        return 0

def sanitize_filename(url):
    """Create a safe filename from URL"""
    sanitized = re.sub(r'^https?:\/\/', '', url)
    sanitized = re.sub(r'[\/:*?"<>|]', '_', sanitized)
    return sanitized[:100]

def create_rag_summary(all_chunks):
    """Create a comprehensive summary for RAG use"""
    if not all_chunks:
        return {"error": "No chunks available"}
    
    total_chunks = len(all_chunks)
    sources = list(set([chunk.metadata.get('source', 'Unknown') for chunk in all_chunks]))
    
    # Calculate content statistics
    content_lengths = [len(chunk.page_content) for chunk in all_chunks]
    avg_length = sum(content_lengths) / len(content_lengths)
    
    summary = {
        "total_documents": total_chunks,
        "unique_sources": len(sources),
        "sources": sources,
        "content_stats": {
            "avg_chunk_length": round(avg_length, 2),
            "min_chunk_length": min(content_lengths),
            "max_chunk_length": max(content_lengths),
            "total_characters": sum(content_lengths)
        },
        "ready_for_rag": True,
        "embedding_compatible": True
    }
    
    return summary
