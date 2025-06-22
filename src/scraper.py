
import logging
import html2text
import re
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_robots_txt(url):
    """Check robots.txt compliance"""
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        logging.info(f"Checking robots.txt at: {robots_url}")
        parser.read()
        is_allowed = parser.can_fetch("*", url)
        if not is_allowed:
            logging.warning(f"Scraping DISALLOWED by robots.txt for: {url}")
        return is_allowed
    except Exception as e:
        logging.warning(f"Could not fetch robots.txt from {robots_url}. Assuming allowed. Error: {e}")
        return True

def clean_markdown_content(markdown_text):
    """Clean and improve markdown content"""
    if not markdown_text:
        return ""
    
    # Remove excessive blank lines
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    
    # Clean up malformed markdown
    markdown_text = re.sub(r'\*{3,}', '**', markdown_text)
    markdown_text = re.sub(r'_{3,}', '__', markdown_text)
    
    # Remove navigation and menu items
    lines = markdown_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip common navigation patterns
        if (len(line) < 3 or 
            line.lower().startswith(('menu', 'nav', 'skip to', 'home |', '| home')) or
            re.match(r'^[\s\|\-]+$', line) or
            line.count('|') > 5):  # Likely navigation menu
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def scrape_and_process_url(url):
    """Scrape URL and return cleaned, processed chunks"""
    if not check_robots_txt(url):
        return None

    logging.info(f"Scraping allowed. Loading content from: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Set user agent to avoid blocking
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            page.goto(url, timeout=60000, wait_until='domcontentloaded')

            # Enhanced content extraction with better selectors
            content_selectors = [
                'article', 'main', '[role="main"]', '#main', '#content', 
                '.main', '.content', '.post', '.entry', '.article-content',
                '.post-content', '.entry-content', '.page-content'
            ]
            
            html_content = ""
            selector_used = None
            
            for selector in content_selectors:
                elements = page.locator(selector)
                if elements.count() > 0:
                    html_content = elements.first.inner_html()
                    selector_used = selector
                    logging.info(f"Extracted content using selector: '{selector}'")
                    break

            if not html_content:
                logging.warning(f"No main content found. Using body content for {url}")
                html_content = page.locator('body').inner_html()
                selector_used = 'body'

            # Get additional metadata
            page_title = page.title() or "No Title"
            page_description = ""
            try:
                desc_element = page.locator('meta[name="description"]')
                if desc_element.count() > 0:
                    page_description = desc_element.get_attribute('content') or ""
            except:
                pass

            browser.close()

        if not html_content:
            logging.warning(f"Could not extract HTML content from {url}")
            return None

        # Convert HTML to clean markdown
        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_links = False
        h.ignore_images = True  # Skip images for cleaner text
        h.ignore_emphasis = False
        h.skip_internal_links = True
        
        markdown_content = h.handle(html_content)
        
        # Clean the markdown content
        markdown_content = clean_markdown_content(markdown_content)

        if not markdown_content.strip() or len(markdown_content.strip()) < 100:
            logging.warning(f"Insufficient content after cleaning for {url}")
            return None

        # Smart chunking with header awareness
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )
        
        try:
            md_header_splits = markdown_splitter.split_text(markdown_content)
        except:
            md_header_splits = []

        # Enhanced metadata
        base_metadata = {
            "source": url,
            "title": page_title,
            "description": page_description,
            "selector_used": selector_used,
            "content_type": "markdown"
        }

        if md_header_splits:
            final_chunks = []
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            for split in md_header_splits:
                split.metadata.update(base_metadata)
                # Add header context if available
                header_info = split.metadata.get('Header 1', '') or split.metadata.get('Header 2', '')
                if header_info:
                    split.metadata['section'] = header_info
                
                further_splits = splitter.split_documents([split])
                final_chunks.extend(further_splits)
            
            chunks = final_chunks
        else:
            # Fallback chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.create_documents([markdown_content], metadatas=[base_metadata])

        # Filter out very short or empty chunks
        valid_chunks = []
        for chunk in chunks:
            clean_content = chunk.page_content.strip()
            if len(clean_content) > 50 and not re.match(r'^[\s\W]*$', clean_content):
                chunk.page_content = clean_content
                valid_chunks.append(chunk)

        if not valid_chunks:
            logging.warning(f"No valid chunks after filtering for {url}")
            return None

        logging.info(f"Successfully processed {url}: {len(valid_chunks)} clean chunks created")
        return valid_chunks

    except PlaywrightTimeoutError:
        logging.error(f"Timeout error for {url}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error scraping {url}: {e}")
        return None
