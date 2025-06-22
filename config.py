
URLS_TO_SCRAPE = [
    "https://www.promptingguide.ai/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Web_scraping",
    "https://www.wired.com/category/science/",
    "https://docs.docker.com/get-started/",
    "https://www.freecodecamp.org/news/what-is-web-scraping/",
    "https://github.blog/2023-11-08-universe-2023-copilot-transforms-github-into-the-ai-powered-developer-platform/",
    "https://www.theverge.com/tech",
    "https://www.joelonsoftware.com/2000/08/09/the-joel-test-12-steps-to-better-code/"
]

# RAG Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "data/vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
