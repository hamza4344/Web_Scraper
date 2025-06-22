Web Scraper + RAG System
Intelligent web scraping solution that extracts clean data from websites and prepares it for AI/LLM applications. Built with production-grade error handling, robots.txt compliance, and vector search capabilities.

Overview
Scrapes multiple websites, processes content into clean chunks, and creates searchable embeddings. The system respects robots.txt files and handles various content types automatically.

Features
Scrapes 10+ configurable websites with full content extraction
Robots.txt compliance built-in
Smart content cleaning (removes nav, ads, irrelevant sections)
Vector embeddings for similarity search
Ready for LLM integration (OpenAI, Claude, local models)
Quick Start
Install dependencies:

pip install -r requirements.txt
playwright install --with-deps
Run the scraper:

python main.py
Test search functionality:

python main.py demo
Architecture
Root/
├── config.py           # URLs and settings
├── main.py            # Main script
├── requirements.txt   # Dependencies
├── src/
│   ├── scraper.py     # Web scraping logic
│   ├── utils.py       # Data processing
│   └── rag.py         # Vector embeddings
└── data/
    ├── processed/     # JSON outputs
    └── vector_store/  # FAISS index
Configuration
Edit URLs in config.py:

URLS_TO_SCRAPE = [
    "https://www.promptingguide.ai/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    # Add more URLs...
]
Adjust processing settings:

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
🤖 LLM Integration Status
Ready to Use
Vector embeddings and similarity search
Structured JSON datasets
Enriched metadata for context
FAISS vector store for fast retrieval
Optional LLM Enhancement
For demonstration purposes, a simple LLM integration can be added. The core RAG infrastructure is complete - connecting to any LLM provider is straightforward:

"""Optional: Simple LLM integration demo"""
    print("Note: Add your preferred LLM here (OpenAI, Claude, local model)")

3 api integration
def query_with_context(question):
    # Retrieve relevant documents
    # your llm logice comes here
    return response

# Local Model Example  
from transformers import pipeline
llm = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def local_query(question):
    docs = rag_processor.similarity_search(question, k=2)
    context = docs[0].page_content[:500] if docs else ""
    return llm(f"Context: {context}\nQ: {question}\nA:")
Output Files
The scraper generates several data formats:

data/processed/{url}_chunks.json - Individual website data
data/processed/all_chunks_combined.json - Complete dataset
data/processed/rag_summary.json - Processing statistics
data/vector_store/ - FAISS embeddings for search
Search Demo
Test the RAG functionality interactively:

python main.py demo
Try queries like:

"What is web scraping?"
The system will find relevant content from scraped pages and display matches with source attribution.

Implementation Details
Web Scraping Process
The scraper uses Playwright for reliable content extraction. It checks robots.txt compliance, tries multiple CSS selectors to find main content, and converts HTML to clean markdown format.

Data Processing Pipeline
Content goes through several cleaning stages: removing navigation elements, normalizing whitespace, filtering short chunks, and adding metadata. The system uses header-aware chunking to preserve document structure.

Vector Search Setup
Embeddings are created using sentence-transformers and stored in FAISS for fast similarity search. The vector store can handle thousands of documents efficiently.

Dependencies
Core libraries used:

playwright - Web scraping engine
langchain - Text processing and chunking
sentence-transformers - Text embeddings
faiss-cpu - Vector similarity search
`html2text' - HTML to markdown conversion
