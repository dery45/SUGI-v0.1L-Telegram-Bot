# 🌾 SUGI v0.1L – Intelligent Agricultural Assistant (Indonesia)

**S**ystem **U**tama **G**enerative **I**ntelijen  
An AI-powered RAG assistant specifically designed for farmers, growers, government officials, and agribusiness players in Indonesia & Southeast Asia.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green?logo=ollama)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB%20Server-purple)](https://www.trychroma.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Daily%20Insights-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
![Score](https://img.shields.io/badge/RAG%20Score-93%2F100-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 🔥 Main Features

- **Hybrid RAG** → BM25 + Dense Vector (mxbai-embed-large) + Cross-Encoder Reranker  
- **Multi-Platform** → CLI interface & **Telegram Bot Integration**  
- **Dynamic Retriever** → Automatic weighting based on query type (plant/weather/history)  
- **3-Layer Query Rewriting** → rule-based (0ms) → Qwen2.5 fallback (~800ms) → original query  
- **Definitional Query Detection** → Phrases like "apa itu X" / "what is X" skip rewriting to prevent false topic injection.  
- **Rewrite-Type Scope Gating** → Only suffix-based rewrites (e.g., "menanamnya" → "menanam semangka") can bypass scope; word replacements cannot inject agriculture keywords into unrelated queries.  
- **High Performance (Try 2 Times)** → History limited to 2 turns & Retrieval k=2 for near-instant responses.
- **Real-time Agricultural Data**  
  - Daily weather + agronomic alerts (drought, flood, heat stress, disease) from Open-Meteo  
  - Complete plant information (species, pests, diseases, care guides) via Perenual API  
  - **Fast Failure API** → Fails in <1s if rate limited (429), immediately falling back to local RAG data.
  - Automatic indexing for CSV/XLSX/PDF (commodity prices, cultivation guides, etc.)  
- **Stability Guard** → 8,000 character truncation & 4,096 context window to prevent overflow errors.
- **Embedding Safety** → Automatically truncates long documents (>2,000 chars) before storage to fit embedding model limits. Failed embeddings are caught gracefully without crashing.
- **Long-term Memory** → Session summaries stored in ChromaDB for multi-turn context  
- **Daily Insight Engine** → Sends daily insights to MongoDB every 12 hours (regional prices, weather, planting tips, policies).
- **INI-based Configuration** → All keywords & plant maps in `config/settings/`, no code changes needed.
- **Scope Guard** → Only answers agriculture & plantation related topics.  

## Tech Stack

| Component | Detail |
|---|---|
| **Primary LLM** | Llama 3.2 fine-tuned → `sugi-v0.1L` (via Ollama) |
| **Utility Model** | `qwen2.5:1.5b` — query rewriting fallback, plant extraction, eval loop, insights |
| **Embedding** | `mxbai-embed-large` |
| **Vector Store** | ChromaDB Server mode — 4 collections: `langchain`, `weather_data`, `plant_data`, `conversation_memory` |
| **Retriever** | Ensemble (BM25 + Vector) + Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`, top_n=4, k=2) |
| **Insight DB** | MongoDB — 5 collections in `sugi_insights` database |
| **External APIs** | Open-Meteo (Free weather), Perenual (Plants & Pests) |
| **Framework** | LangChain, LangChain-Classic |

## Installation (Local Development)

### 1. Prerequisites

- Python 3.10+
- Ollama installed with the following models:
  ```bash
  ollama pull llama3.2
  ollama pull qwen2.5:1.5b
  ollama create sugi-v0.1L -f Modelfile
  ollama pull mxbai-embed-large
  ```

### 2. Clone & Setup

```bash
git clone https://github.com/dery45/SUGI-v0.1L-Telegram-Bot.git
cd SUGI-v0.1L-Telegram-Bot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configuration (.env)

Create a `.env` file in the project root based on `.env.example`. Key settings include:

| Variable | Description | Default |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | API Token from @BotFather | - |
| `LLM_MODEL` | Main response model | `sugi-v0.1L` |
| `UTILITY_MODEL` | Utility tasks (rewriting, extraction) | `qwen2.5:1.5b` |
| `CHROMA_HOST` / `PORT` | ChromaDB server connection | `localhost:8000` |
| `DEBUG_ALLOWED_USERS` | Restrict `!debug` commands (comma-separated IDs) | *(All)* |
| `MEMORY_TTL_DAYS` | Days to keep session summaries | `14` |
| `BM25_CACHE_PATH` | Path to BM25 index file | `bm25_cache.pkl` |

### 4. Database Server
ChromaDB must be running in the background before any AI queries or indexing.

**Run server (Terminal 1):**
```bash
chroma run --path data/db --port 8000
```

### 5. Running the Application Pipeline
You can now start all RAG services and the Telegram bot with a single command:

**Run pipeline (Terminal 2):**
```bash
python start_all.py
```
*(Alternatively, you can run `python interfaces/cli/main.py` to run the terminal client).*

## Project Architecture

```
SUGI-v0.1L/
├── config/                        # Configuration
│   ├── .env                       # Environment variables
│   ├── .env.example               # Template for .env
│   ├── Modelfile                  # Ollama model definition
│   └── settings/                  # INI-based keyword configs
│       ├── scope_config.ini       # Allowed/blocked topic keywords
│       ├── rewriter_config.ini    # Referential words & topic mapping
│       └── plant_keywords.ini     # Plant name map & detection keywords
├── core/                          # Shared Engine Logic
│   ├── sugi_core.py               # Main RAG pipeline & query processing
│   ├── plant_api.py               # Perenual API client & caching
│   ├── eval_loop.py               # Faithfulness & relevance scoring
│   ├── query_logger.py            # Query tracing & debug logs
│   └── user_store.py              # User profile management
├── services/                      # Background Services
│   ├── vectorCSV.py               # CSV/XLSX watcher & indexer
│   ├── vectorpdf.py               # PDF watcher & indexer
│   ├── vectorWeather.py           # Open-Meteo weather crawler
│   └── daily_insight.py           # MongoDB insight generator (12h cron)
├── interfaces/                    # Entry Points
│   ├── cli/
│   │   └── main.py                # Terminal chat client
│   └── telegram/
│       ├── telegram_bot.py        # Telegram bot entry point
│       ├── requirements_telegram.txt
│       └── DEPLOYMENT_GUIDE.md    # Telegram-specific setup guide
├── data/                          # Runtime Data (git-ignored)
│   ├── db/                        # ChromaDB persistent storage
│   ├── logs/                      # Query logs
│   ├── raw_dataset/               # CSV/XLSX source files
│   ├── raw_pdfs/                  # PDF source files
│   └── users.json                 # User profiles
├── tests/                         # Testing utilities
├── start_all.py                   # Master deployment script
├── README.md
└── requirements.txt
```

## License
MIT License.

Last updated: March 2026 · v0.1L (Scope-Hardened) · RAG Score 93/100
