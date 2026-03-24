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
- **High Performance (Try 2 Times)** → History limited to 2 turns & Retrieval k=2 for near-instant responses.
- **Real-time Agricultural Data**  
  - Daily weather + agronomic alerts (drought, flood, heat stress, disease) from Open-Meteo  
  - Complete plant information (species, pests, diseases, care guides) via Perenual API  
  - **Fast Failure API** → Fails in <1s if rate limited (429), immediately falling back to local RAG data.
  - Automatic indexing for CSV/XLSX/PDF (commodity prices, cultivation guides, etc.)  
- **Stability Guard** → 8,000 character truncation & 4,096 context window to prevent overflow errors.
- **Embedding Safety** → Automatically truncates long documents (>3000 chars) before storage to fit embedding model limits.
- **Long-term Memory** → Session summaries stored in ChromaDB for multi-turn context  
- **Daily Insight Engine** → Sends daily insights to MongoDB every 12 hours (regional prices, weather, planting tips, policies).
- **INI-based Configuration** → All keywords & plant maps in `word_config/`, no code changes needed.
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

### 4. ChromaDB Server ⚠️ Required before running other scripts

SUGI uses ChromaDB in **server mode** to allow simultaneous indexing and serving.

**Run server (Terminal 1):**
```bash
chroma run --path ./chrome_longchain_db --port 8000
```

### 5. Running the Pipeline

```bash
python vectorCSV.py      # Terminal 2 — Index CSV/XLSX
python vectorpdf.py      # Terminal 3 — Index PDF
python vectorWeather.py  # Terminal 4 — Refresh weather (every 5 mins)
python telegram_connection/telegram_bot.py  # Terminal 5 — Start Bot
python main.py           # Terminal 6 — Start CLI (Optional)
python daily_insight.py  # Terminal 7 — Start Insight Engine (Optional)
```

## Folder Structure

```
sugi-v0.1L/
├── dataset/                  # CSV/XLSX (Excluded by Git)
├── pdfsource/                # PDF Documents (Excluded by Git)
├── chrome_longchain_db/      # ChromaDB data (Excluded by Git)
├── word_config/              # Domain & keyword rules (.ini)
├── telegram_connection/      # Telegram Bot modules
├── sugi_core.py              # Shared Core Logic
├── user_store.py             # Shared User Management
├── main.py                   # CLI Entry point
├── .env                      # Unified Configuration
└── requirements.txt
```

## License
MIT License.

Last updated: March 2026 · v0.1L (Context Optimized) · RAG Score 93/100
