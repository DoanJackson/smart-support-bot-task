# Bot Scraper

A web scraper and document indexer for OptiSigns Knowledge Base. It extracts support articles via the Zendesk API, converts them into Markdown, and uploads them to a Chroma Vector Database for AI Assistants.

## Features
- **Scrapes Zendesk API**: Efficiently extracts knowledge base articles.
- **Markdown Conversion**: Uses `markdownify` to convert HTML to clean markdown, preserving headings, links, and formatting.
- **Delta Update Syncing**: Uses `sync_state.json` to keep track of updated articles and only fetches the newly modified data (`added` or `updated`).
- **Vector DB Integration**: Utilizes LangChain and Google Gemini Embeddings (`models/gemini-embedding-2`) to chunk and store data in ChromaDB.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Copy `.env.sample` to `.env` and fill in your Gemini API Key.
   ```bash
   cp .env.sample .env
   ```

## Running Locally

To run the job manually on your machine:
```bash
python main.py
```
*Note: All scraped articles and ChromaDB data will be stored inside the local `data/` directory.*

## Running as a Daily Cron Job (Docker)

To run the script as a daily task, it has been dockerized. It uses a Volume Mount to persist the `data/` directory (which contains `sync_state.json` and `chroma_db`).

1. **Build the image:**
   ```bash
   docker build -t optibot-scraper .
   ```

2. **Run the container:**
   ```bash
   docker run --env-file .env -v $(pwd)/data:/app/data optibot-scraper
   ```
   *Note: This container runs once and exits with code 0 (exit 0). You can schedule this command using `crontab` on AWS/DigitalOcean or set up a daily task via cloud services.*

## Vector Store & Chunking Strategy

**Strategy:** The text is chunked using LangChain's `RecursiveCharacterTextSplitter` with `chunk_size=800` and `chunk_overlap=200`.
- **Why 800?**: It allows chunks to hold about a paragraph or two of context which is ideal for accurate retrieval by the AI without exceeding reasonable token limits per snippet.
- **Why overlap 200?**: Overlap prevents the splitter from cutting off sentences abruptly, preserving the context at boundaries between adjacent chunks.
- **Embeddings:** `models/gemini-embedding-2` was used.

## Logs & Proof of Concept

- **Job Logs**: [Insert Log URL / Artifact URL here]
- **Sanity Check**: Assistant correctly answered "How do I add a YouTube video?".
  
  **OptiBot's Response:**
  > To add a YouTube video to your digital signs:
  > - Go to Files/Assets, and click on "App".
  > - Click "YouTube" (or YouTube Live).
  > - Enter your video's Name and actual URL (not the Share link).
  > - Click Save.
  > - Assign the newly created YouTube asset to your screen.
  > 
  > Article URL: https://support.optisigns.com/hc/en-us/articles/360051014713-How-to-use-YouTube-with-OptiSigns

- **Screenshots**: (See provided screenshots in `picture/Sanity_check_response.png` for evidence of Assistant response and citations).
