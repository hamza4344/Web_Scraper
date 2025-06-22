
import os
import logging
import json
from config import URLS_TO_SCRAPE
from src.scraper import scrape_and_process_url
from src.utils import save_docs_to_json, sanitize_filename, create_rag_summary
from src.rag import RAGProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
OUTPUT_DIR = "data/processed"

def run_enhanced_scraper():
    """Main function to run scraping with RAG integration"""
    logging.info("="*60)
    logging.info("STARTING ENHANCED WEB SCRAPER WITH RAG INTEGRATION")
    logging.info("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_chunks = []
    successful_scrapes = 0
    failed_scrapes = 0
    total_chunks = 0

    # Phase 1: Scrape and process all URLs
    logging.info(f"Phase 1: Scraping {len(URLS_TO_SCRAPE)} URLs with robots.txt compliance")
    
    for i, url in enumerate(URLS_TO_SCRAPE, 1):
        logging.info(f"\n[{i}/{len(URLS_TO_SCRAPE)}] Processing: {url}")
        
        try:
            processed_chunks = scrape_and_process_url(url)

            if processed_chunks and len(processed_chunks) > 0:
                # Save individual URL chunks
                safe_filename = sanitize_filename(url)
                output_path = os.path.join(OUTPUT_DIR, f"{safe_filename}_chunks.json")
                chunks_saved = save_docs_to_json(processed_chunks, output_path)
                
                if chunks_saved > 0:
                    all_chunks.extend(processed_chunks)
                    total_chunks += len(processed_chunks)
                    successful_scrapes += 1
                    logging.info(f"✓ Success: {len(processed_chunks)} chunks created")
                else:
                    failed_scrapes += 1
                    logging.warning(f"✗ Failed: No valid chunks after cleaning")
            else:
                failed_scrapes += 1
                logging.warning(f"✗ Failed: Could not process {url}")
                
        except Exception as e:
            failed_scrapes += 1
            logging.error(f"✗ Error processing {url}: {e}")

    # Phase 1 Summary
    logging.info("\n" + "="*50)
    logging.info("PHASE 1 COMPLETE: SCRAPING SUMMARY")
    logging.info("="*50)
    logging.info(f"Successful scrapes: {successful_scrapes}")
    logging.info(f"Failed scrapes: {failed_scrapes}")
    logging.info(f"Total chunks created: {total_chunks}")
    logging.info(f"Average chunks per successful URL: {total_chunks/successful_scrapes if successful_scrapes > 0 else 0:.1f}")

    if not all_chunks:
        logging.error("No content was successfully scraped. Cannot proceed with RAG setup.")
        return False

    # Phase 2: RAG Integration
    logging.info("\n" + "="*50)
    logging.info("PHASE 2: RAG INTEGRATION")
    logging.info("="*50)
    
    try:
        rag_processor = RAGProcessor()
        
        logging.info("Creating vector embeddings...")
        vector_store = rag_processor.create_vector_store(all_chunks)
        
        if vector_store:
            # Save vector store
            if rag_processor.save_vector_store():
                logging.info("✓ Vector store saved successfully")
            
            # Create and save comprehensive summary
            rag_summary = create_rag_summary(all_chunks)
            summary_path = os.path.join(OUTPUT_DIR, "rag_summary.json")
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(rag_summary, f, indent=2, ensure_ascii=False)
            
            logging.info(f"✓ RAG summary saved to {summary_path}")
            
            # Save combined dataset
            combined_path = os.path.join(OUTPUT_DIR, "all_chunks_combined.json")
            save_docs_to_json(all_chunks, combined_path)
            logging.info(f"✓ Combined dataset saved to {combined_path}")
            
            # Phase 3: Test RAG functionality
            logging.info("\n" + "="*50)
            logging.info("PHASE 3: TESTING RAG SYSTEM")
            logging.info("="*50)
            
            test_queries = [
                "What is web scraping?",
                "How do large language models work?",
                "What are coding best practices?",
                "Tell me about AI agents"
            ]
            
            for query in test_queries:
                logging.info(f"\nTesting query: '{query}'")
                results = rag_processor.similarity_search(query, k=2)
                
                if results:
                    for j, doc in enumerate(results, 1):
                        source = doc.metadata.get('source', 'Unknown')
                        title = doc.metadata.get('title', 'No title')
                        preview = doc.page_content[:150].replace('\n', ' ')
                        logging.info(f"  [{j}] {title}")
                        logging.info(f"      Source: {source}")
                        logging.info(f"      Preview: {preview}...")
                else:
                    logging.warning(f"  No results found for: {query}")
            
            # Final success message
            logging.info("\n" + "="*60)
            logging.info("✓ SUCCESS: RAG SYSTEM READY FOR LLM INTEGRATION")
            logging.info("="*60)
            logging.info(f"📊 Total documents in vector store: {len(all_chunks)}")
            logging.info(f"📁 Data saved in: {OUTPUT_DIR}")
            logging.info(f"🔍 Vector store ready for similarity search")
            logging.info(f"🤖 Compatible with any LLM for RAG applications")
            logging.info("="*60)
            
            return True
            
        else:
            logging.error("Failed to create vector store")
            return False
            
    except Exception as e:
        logging.error(f"RAG integration failed: {e}")
        return False

def demo_rag_search():
    """Demonstrate RAG search capabilities"""
    logging.info("DEMO: RAG Search Functionality")
    
    try:
        rag_processor = RAGProcessor()
        
        if rag_processor.load_vector_store():
            while True:
                query = input("\nEnter search query (or 'quit' to exit): ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query:
                    results = rag_processor.similarity_search(query, k=3)
                    print(f"\nFound {len(results)} results for: '{query}'")
                    
                    for i, doc in enumerate(results, 1):
                        source = doc.metadata.get('source', 'Unknown')
                        title = doc.metadata.get('title', 'No title')
                        print(f"\n[{i}] {title}")
                        print(f"Source: {source}")
                        print(f"Content: {doc.page_content[:300]}...")
                        print("-" * 50)
        else:
            print("Could not load vector store. Run the scraper first.")
            
    except Exception as e:
        logging.error(f"Demo failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_rag_search()
    else:
        success = run_enhanced_scraper()
        if success:
            print("\n🎉 Scraping and RAG setup completed successfully!")
            print("Run 'python main.py demo' to test search functionality")
        else:
            print("\n❌ Scraping failed. Check logs for details.")
