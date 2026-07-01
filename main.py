import os
import re
import json
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from markdownify import markdownify as md

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

load_dotenv()

API_URL = "https://support.optisigns.com/api/v2/help_center/en-us/articles.json?sort_by=updated_at&sort_order=desc&per_page=50"

# Directories for local state
DATA_DIR = "data"
ARTICLES_DIR = os.path.join(DATA_DIR, "articles")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
SYNC_STATE_FILE = os.path.join(DATA_DIR, "sync_state.json")

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def load_sync_state():
    if os.path.exists(SYNC_STATE_FILE):
        try:
            with open(SYNC_STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Could not decode sync_state.json. Starting fresh.")
    return {}

def save_sync_state(state):
    with open(SYNC_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=4)

def setup_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTICLES_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

def scrape_and_format_article(article):
    title = article.get('title', 'Untitled')
    body = article.get('body', '')
    html_url = article.get('html_url', '')
    article_id = article.get('id')
    
    if not body:
        return None
        
    slug = slugify(title)
    if not slug:
        slug = f"article-{article_id}"
        
    # Add a header to the markdown file
    md_content = f"# {title}\n\n"
    md_content += f"**Source:** [{html_url}]({html_url})\n\n"
    md_content += "---\n\n"
    
    # Convert HTML body to Markdown
    body_md = md(body, heading_style="ATX", strip=['script', 'style'])
    md_content += body_md
    
    filepath = os.path.join(ARTICLES_DIR, f"{slug}.md")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    return filepath

def chunk_and_embed_documents(file_paths, vectorstore):
    print(f"Loading and chunking {len(file_paths)} documents...")
    documents = []
    for file_path in file_paths:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    if not documents:
        return
        
    # Strategy: chunk_size=800, chunk_overlap=200 for good retrieval accuracy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Upload to Vector Store in batches to avoid rate limit
    batch_size = 80
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} (size: {len(batch)} chunks)...")
        
        while True:
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                print(f"Rate limit or error hit, sleeping for 65s and retrying... Error: {e}")
                time.sleep(65)
                
        if i + batch_size < len(chunks):
            print("Sleeping for 65 seconds to avoid exceeding rate limits...")
            time.sleep(65)

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Error: GEMINI_API_KEY is not set in .env")
        return

    print("=== OptiBot Scraper Cron Job Started ===")
    setup_directories()
    sync_state = load_sync_state()
    
    print(f"Fetching articles from {API_URL}...")
    response = requests.get(API_URL)
    response.raise_for_status()
    
    data = response.json()
    articles = data.get('articles', [])
    print(f"Found {len(articles)} articles from main API.")
    
    # Tính năng mở rộng: Lấy thêm các bài viết lẻ dựa trên ID (từ file .env)
    extra_ids = os.getenv("EXTRA_ARTICLE_IDS", "")
    if extra_ids:
        print(f"Fetching extra articles: {extra_ids}")
        for a_id in extra_ids.split(','):
            a_id = a_id.strip()
            if not a_id: continue
            try:
                # Zendesk API lấy bài viết đơn lẻ
                r = requests.get(f"https://support.optisigns.com/api/v2/help_center/en-us/articles/{a_id}.json")
                if r.status_code == 200:
                    art_data = r.json().get('article')
                    if art_data:
                        # Tránh trùng lặp nếu bài viết đã nằm trong danh sách cào chính
                        if not any(a.get('id') == art_data.get('id') for a in articles):
                            articles.append(art_data)
                            print(f" - Fetched extra article: {a_id}")
            except Exception as e:
                print(f" - Failed to fetch extra article {a_id}: {e}")
                
    print(f"Total articles to process: {len(articles)}")
    
    stats = {"added": 0, "updated": 0, "skipped": 0, "failed": 0}
    processed_files = []
    
    # Lấy giới hạn số bài viết từ file .env (mặc định 30)
    max_articles = int(os.getenv("MAX_ARTICLES", 30))
    
    # Analyze delta
    for i, article in enumerate(articles):
        # Dừng lại nếu đã đạt giới hạn max_articles
        if i >= max_articles:
            break
            
        article_id = str(article.get('id'))
        updated_at = article.get('updated_at')
        
        if article_id in sync_state:
            if sync_state[article_id] == updated_at:
                stats["skipped"] += 1
                continue
            else:
                stats["updated"] += 1
        else:
            stats["added"] += 1
            
        # If we reach here, it's added or updated -> scrape
        filepath = scrape_and_format_article(article)
        if filepath:
            processed_files.append(filepath)
            # Update state with new updated_at
            sync_state[article_id] = updated_at
        else:
            stats["failed"] += 1
            
    print(f"Delta Analysis: {stats['added']} added, {stats['updated']} updated, {stats['skipped']} skipped, {stats['failed']} failed.")
    
    if processed_files:
        print("Initializing Vector Store for new/updated content...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
        
        chunk_and_embed_documents(processed_files, vectorstore)
        print("Finished embedding documents.")
        
    save_sync_state(sync_state)
    print("=== Cron Job Finished Successfully ===")

if __name__ == "__main__":
    main()
