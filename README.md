# 🌾 SUGI v0.1L – Intelligent Agricultural Assistant (Indonesia)

An AI-powered RAG assistant specifically designed for farmers, growers, government officials, and agribusiness players in Indonesia & Southeast Asia.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green?logo=ollama)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB%20Server-purple)](https://www.trychroma.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Daily%20Insights-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
![Score](https://img.shields.io/badge/RAG%20Score-94%2F100-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 📱 Live Demo

You can try the SUGI AI chatbot live on Telegram:
👉 **[Chat with @sugi_demo_llmbot on Telegram](https://t.me/sugi_demo_llmbot)**

<p align="center">
  <img src="image/QR-Link-to-try-sugiai.png" width="220" alt="QR Link to try SUGI AI">
</p>

## 📸 Interface Previews

Here is SUGI AI in action on Telegram:

<p align="center">
  <img src="image/chatbot-answer-1.png" width="32%" alt="Chatbot Answer Example 1" />
  <img src="image/chatbot-answer-2.png" width="32%" alt="Chatbot Answer Example 2" />
  <img src="image/chatbot-answer-when-offline.png" width="32%" alt="Chatbot Answer Offline Mode" />
</p>

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
  - **Crash-Safe Startup (A11)** → rate-limit flag helpers now defined before the startup API-key validation that calls them, so a 429 during boot trips the 1-hour cooldown cleanly instead of raising `NameError` (which an `except ImportError` guard can't catch) and killing the whole bot.
  - Automatic indexing for CSV/XLSX/PDF (commodity prices, cultivation guides, etc.)  
- **Stability Guard** → 8,000 character truncation & 4,096 context window to prevent overflow errors.
- **Resource Contention Hardening (M0/A8)** → `STARTUP_GRACE_SECONDS` staggers the first full run of background services (farmer/gov insight, daily insight, vector watchers) so the chatbot wins the Ollama warm-up race; `INSIGHT_LLM_DELAY` paces the LLM calls inside insight loops. A8 refines this per service: staggered grace (`DAILY_INSIGHT_GRACE_SECONDS=180`, `GOVERNMENT_INSIGHT_GRACE_SECONDS=240`, `FARMER_INSIGHT_GRACE_SECONDS=300`) and a 240s insight-only LLM timeout so engines never wake simultaneously or block on a busy Ollama.
- **Daily Insight Concurrency Backfill (A12)** → `daily_insight.py` no longer fires 5 simultaneous LLM calls via a hardcoded `ThreadPoolExecutor(max_workers=5)`; its price/weather/planting generators now run through `_run_paced()` — sequential by default with `INSIGHT_LLM_DELAY` pacing (`DAILY_INSIGHT_MAX_WORKERS=1`), so a single `run_once()` can't push Ollama calls toward timeout. Setting `DAILY_INSIGHT_MAX_WORKERS>1` explicitly restores a bounded thread pool.
- **Transient Mongo Retry (A7 backfill)** → `daily_insight.py` startup ping now wraps Atlas "No primary" errors in `_ping_with_retry` (3 attempts, 5s/10s linear backoff) instead of exiting on the first hiccup — matching the farmer/gov insight services.
- **Fail-Closed Debug Access (R1)** → `DEBUG_ALLOWED_USERS` is now **deny-by-default**: if unset (empty), `/debug` and all `!`-commands are blocked for everyone with a one-time startup warning. Previously an empty variable accidentally opened debug access to all users (both gate sites in `telegram_bot.py`).
- **Blocked-Topics Enforced (R2)** → the 43-keyword `[blocked_topics]` list in `scope_config.ini` is now actually enforced by the scope guard — checked **before** allowed keywords, so blocked overrides allowed (e.g. "apakah padi bisa mengobati diabetes?" refuses despite "padi" being in-scope). The list was reviewed for agriculture false-positives before enabling.
- **Global Telegram Error Handler (R3)** → any exception thrown outside `SugiCore.ask()`'s internal try/except (e.g. a failure while sending the reply) is caught by a registered PTB error handler that logs the traceback and replies "⚠️ Maaf, terjadi kesalahan tak terduga…" — the user never gets silence. Runtime-verified end-to-end.
- **Content-Hash Re-indexing (R4)** → CSV/XLSX/PDF dedup is now by **content hash** (`file_hash` chunk metadata, MD5 over 8KiB blocks), not just filename. Editing an already-indexed file deletes its old chunks and re-indexes them; unchanged files are skipped. PDF indexing also gained the BM25 cache invalidation the CSV service already had.
- **Multi-turn History Gating (A9)** → `{history}` is injected into the prompt only when the rewriter detects a referential/follow-up signal; self-contained questions get an empty history, so prior answers (e.g. apple) can't poison a new topic (e.g. December planting).
- **Retrieval-Level Memory Gating (A10 + A4)** → the same `needs_ref_context` signal now also gates the *retrieval* path: memory-store similarity search and its merge into the context run only for genuinely referential follow-ups, closing the second leak pipe A9 couldn't reach (old-topic chunks like labu siam arriving via memory). A4 closed in the same pass: the shared unfiltered `memory_retriever` (no `user_id` filter, cross-user privacy risk) is removed — only the correctly user-scoped memory path remains.
- **Unified Single-Pass Reranking (M1 + A11)** → Weather docs are merged into the RAG pool and ALL candidates pass through the CrossEncoderReranker exactly once — no irrelevant weather chunks winning prompt slots by sheer count, and no double-scoring of the RAG portion (the ensemble retriever is no longer wrapped in a compression retriever that pre-reranked it).
- **Context Cap & Prompt Trim (M2)** → Final context capped at 8 chunks to fit `num_ctx=4096`; persona moved to the Modelfile SYSTEM prompt, session date/time info pinned to the bottom of the answer prompt, bullet depth limited to 1 level.
- **Embedding Safety** → Automatically truncates long documents (>2,000 chars) before storage to fit embedding model limits. Failed embeddings are caught gracefully without crashing.
- **Long-term Memory** → Session summaries stored in ChromaDB for multi-turn context  
- **Daily Insight Engine** → Sends daily insights to MongoDB every 12 hours (regional prices, weather, planting tips, policies).
- **Government Insight Engine** → AI-generated strategic insights from 14 government datasets (prices, food security, distribution) with ChromaDB learning loop. Polls every 3600s.
- **Farmer Insight Engine** → 10 AI-generated farmer insights (Skor PPH, Komoditas Terbaik, Margin Positif, Surplus Pangan, Peluang Bulanan, Rekomendasi Tanam/Jual, dll.) + 1 Government Policy Recommendation. Uses example-driven prompting, multi-stage validation (Bahasa Indonesia, no truncation, no prefixes), retry on failure, and full SUGI ecosystem enrichment (RAG, weather, plant API, ChromaDB memory). Change detection with daily refresh.
- **INI-based Configuration** → All keywords & plant maps in `config/settings/`, no code changes needed.
- **Scope Guard** → Only answers agriculture & plantation related topics.  
- **Offline Message Catch-up** → Telegram bot processes messages sent while offline on restart, with crash-safe offset persistence.
- **Backpressure & Input Guard (T1-1/T1-2)** → `MAX_CONCURRENT_ASK=2` semaphore with `wait_for(0.01)` fast-fail `⏳ Sedang banyak pertanyaan...` (no queue buildup to `71s` p95); `MAX_QUESTION_CHARS=2000` early return `Maaf, pertanyaan terlalu panjang...` in `<1ms` vs prior `120s` Ollama timeout.
- **UserStore Atomic Locking (T1-3)** → `threading.Lock` + `tmp→os.replace` atomic write — no `visit_count` loss under concurrent Telegram messages (validated `20` threads → `21/21`).
- **GPU Acceleration (G1)** → RX 6600 8GB Vulkan, Ollama `0.33.3` `100% VRAM` (`sugi 2.54GB + qwen 1.16GB + mxbai 0.61GB =4.31GB` primary) — corrects prior `CPU-only` inference; modest `15.2s` median vs `18.9s` CPU-era.
- **Dual-Ollama Isolation (P2-2/G2)** → `OLLAMA_HOST_INSIGHT=http://127.0.0.1:11435` + `start_all.py` second `ollama serve` (qwen `1.1GB`, `keep_alive 60s` G2) — background `~600s/day` qwen insight no longer queues on user path; `5.4-5.6GB` of `8GB` VRAM headroom (corrected from `8.6GB`).
- **Observability (P2-3)** → `queries.jsonl` now carries `model_name`/`prompt_version` (`PROMPT_VERSION=v1`), `print_debug_report` shows `avg/p50/p95` + `Models/Prompts`, `!stats` shows live `In-flight: 0/2`.
- **Telegram Streaming (P2-4)** → `SugiCore.ask(on_chunk)` streams `full_response` per `_live_chain.stream()` chunk → `run_coroutine_threadsafe(edit_text, 200 chars / 1s)` — perceived TTFT `0.07s` visible, final `edit_text` handles OOS/length-guard paths.
- **Stage Timing (I1)** → `ask()` emits `[STAGE_TIMING] {scope, rewrite, plant, retrieval, rerank, generation} total` + `ThreadingHTTPServer backlog 50` (V1b) — explains `20-30s` tail (`generation 5-19s` dominant, `plant 0.03-5.0s`, `retrieval 0.6-3.9s`).
- **Generation Tuning (P3-1)** → `num_predict 512→400` on `sugi-v0.1L` (rarely near `400` tokens `chars/4`); `V1b 10/10` wall `45-50s` not hard wall, `V2b OLLAMA_NUM_PARALLEL=2` tested negative (compute-bound, `121s` vs `27s` worse).
- **Embedding Pooling Fix (PF-V1)** → `Session` reuse 2.06s→0.031s **measured** (was `requests.get` per call); `OllamaEmbeddings` 0.03s now ground truth, retrieval 0.60-3.48s lean.
- **Perenual Pooling+Parallel (PF2-1)** → `requests.Session` + `species‖disease` parallel `ThreadPoolExecutor(2)` — 2.795s→1.528s **measured** (1.20s pooling +0.06s parallel), `Session` safe concurrent (fallback `threading.local`).
- **Outlier Busy Signal (Part0)** → `data/busy/*.flag` per-request inter-process signal, insight `wait_if_busy` 60s defer (stale 300s auto-clean) — outliers 8→0 since, recent p95 35.75s (was 94.21s), 5 concurrent+insight p95 49.81s well below 94-365s **measured**.
- **Duplicate Fix (Part1)** → `query_id` 12 hex + idempotent `commit_trace` double-commit suppressed — dups 20→0, huge ≥600s 0 since **measured**.
- **Generation Split (PF2-3)** → `GENERATION_SPLIT` prefill 40-73% (e.g. 7.1/2.5s, 0.02/5.6s) residual <0.03s **measured** via `BaseCallbackHandler` `on_llm_end`, justifies cap 6.
- **Prompt Trim (Part4)** → `all_docs` cap 8→6 (3680 vs 3714 -1% **measured**, 0/25 flagged both) — saves prefill, no regression.
- **HTTP Streaming Endpoint (PF2-2)** → `POST /ask/stream` NDJSON `{"chunk":...}` + `{"done":true,"final":...}` — TTFT 1.0-1.3s warm **measured** (22.59s cold 15s prefill vs 0.02s warm), buffered still 6-12s.
- **Reranker Background (P4-4 QW-3)** → `HuggingFaceCrossEncoder` 4.19s overlapped background (`threading.Event` 10s fallback to no-rerank) — init not serial.
- **Warm-up (P4-4 QW-5)** → `model.invoke("Halo")` background after `SugiCore ready` — first real after warm-up 21s cold vs 2-3s warm race, `🔥 Warm-up done`.
- **Chroma Fallback (P4-5)** → BM25-only degraded mode on `Chroma` `embed`/`retrieval` failure + `*Catatan: pengambilan data sedang terganggu*` disclaimer — query survives vs fails outright.

## Tech Stack

| Component | Detail |
|---|---|
| **Primary LLM** | Llama 3.2 personal-tuned → `sugi-v0.1L` (via Ollama 0.33.3, `num_ctx 4096` `num_predict 400` P3-1, `keep_alive 600`, `timeout 180` PF2-V1, `100% VRAM` Vulkan 2.54GB, warm-up `invoke("Halo")` QW-5) |
| **Utility Model** | `qwen2.5:1.5b` — query rewriting (`num_predict 40`, `keep_alive 600`→`60s` G2, `base_url` 11435 H2a), plant extraction (`num_predict 10`), eval (`timeout 15`), insights (`timeout 240s`, `OLLAMA_HOST_INSIGHT` 11435) |
| **Embedding** | `mxbai-embed-large` 0.61GB VRAM, **0.031s pooled Session** (was 2.06s bare `requests.get`) **measured** PF-V1 (`OllamaEmbeddings` 0.03s) |
| **Vector Store** | ChromaDB Server `/api/v2/heartbeat` — 6 collections (`main_dataset` 368k, `langchain` 77k), `ThreadingHTTPServer` backlog 50 (V1b), BM25 fallback on Chroma fail P4-5 |
| **Retriever** | Ensemble (BM25 k=4 + Vector k=4) + Cross-Encoder `ms-marco-MiniLM-L-6-v2` top_n=8 background QW-3 (4.19s overlapped) — single pass (M1+A11); memory user-scoped `needs_ref_context` (A10/A4); `retrieval 0.60-3.48s` + `rerank 0.15-0.50s` (cap 6 Part4) |
| **Insight DB** | MongoDB Atlas 7 collections across `sugi_insights`+`test` (write-only), `keep_alive 60s` insight, `wait_if_busy` 60s defer Part0 (stale 300s) |
| **External APIs** | Open-Meteo (Free), Perenual `Session` pooled 1.59s vs 2.79s + `species‖disease` parallel **measured** PF2-1, `429` trip 1h 0.003s, negative cache 24h 0.001s **measured** |
| **Framework** | LangChain, LangChain-Classic |
| **Observability** | `PROMPT_VERSION=v1`, `model_name`/`prompt_version`/`query_id` 12 hex `p50/p95` `In-flight 0/2` + `[STAGE_TIMING]` 6 stages + `[GENERATION_SPLIT]` prefill/eval residual <0.03s **measured** (PF2-3), `data/busy/*.flag` Part0, degraded disclaimer P4-5 |
| **Orchestration** | `start_all.py` 8 services (Insight Ollama 11435) + `ThreadingHTTPServer` backlog 50 + `Semaphore(2)` T1-1 + `run_coroutine_threadsafe` streaming `POST /ask/stream` NDJSON PF2-2 TTFT 1.0s warm |

---

## 🗄️ Database Schema

### ChromaDB (6 collections — READ/WRITE)

| Collection | Size | Source | Write Trigger | Read Trigger | Schema |
|-----------|------|--------|---------------|--------------|--------|
| `main_dataset` | Dynamic | `vectorCSV.py`, `vectorpdf.py` | New/edited CSV/XLSX/PDF in `data/raw_dataset/` or `data/raw_pdfs/` (dedup by `file_hash` content hash, R4) | Every user query via ensemble retriever (k=4) | `{source, sheet?, row_id?, data_type, file_hash?, chunk_index, page_content}` |
| `weather_data` | ~109 daily summaries | `vectorWeather.py` | Every day change detected (loop every 300s) | Weather query detected → `similarity_search(k=8)` | `{source, location, date, fetch_date, page_content}` |
| `plant_data` | Per-plant cache | `plant_api.py` | Plant query → Perenual API fetch | Plant query detected → `similarity_search(k=3)` | `{source, cache_key, plant_id?, common_name?, image_url?, cached_at, page_content}` |
| `conversation_memory` | Per-user session summaries | `sugi_core.py` | Every 5 user questions → LLM-summarized | Every query (filter by user_id, k=2) + memory recall | `{source, user_id, session_id, timestamp, page_content}` |
| `government_memory` | ~15 docs | `government_insight_service.py` | Per-collection insight generation | Cross-collection learning loop in next generation | `{source, sourceCollection, theme, insightVersion, totalDocuments, generatedAt, page_content}` |
| `insights_memory` | ~11 docs | `farmer_insight_service.py` | Farmer insight + policy generation | Historical memory for future generations | `{source, insightKey, version, type, title, generatedAt, page_content}` |

### MongoDB (2 databases, 7 collections)

`test` database: source data (read) + insight output (write).  
`sugi_insights` database: analytics output only (write).  
Data consumed by external dashboards — zero reads from chatbot application.

| Database | Collection | Doc Count | Service | Content |
|----------|-----------|-----------|---------|---------|
| `sugi_insights` | `price_insights` | ~45 | `daily_insight.py` | LLM-generated price analysis per province from ChromaDB price docs |
| `sugi_insights` | `weather_insights` | ~45 | `daily_insight.py` | LLM-generated weather analysis per location from weather_data |
| `sugi_insights` | `planting_suggestions` | ~400 | `daily_insight.py` | LLM-generated planting recommendations per commodity × season |
| `sugi_insights` | `general_insights` | ~45 | `daily_insight.py` | Top 3-5 policy/trend insights from kebijakan docs |
| `sugi_insights` | `session_summaries` | ~13 | `daily_insight.py` | Conversation session summaries from conversation_memory |
| `sugi_insights` / `test` | `governmentinsights` | ~15 | `government_insight_service.py` + `farmer_insight_service.py` | Per-collection insights + 1 policy recommendation |
| `sugi_insights` / `test` | `farmerinsights` | ~10 | `farmer_insight_service.py` | 10 farmer-focused market insights (pph_score, best_commodity, margin_status, etc.) |

### Local Files

| File | Purpose | Format | Read Trigger | Write Trigger |
|------|---------|--------|--------------|---------------|
| `data/logs/queries.jsonl` | Per-query audit log | JSONL (append) | `!debug`, `!flags`, `!session` commands | Every query processed |
| `data/logs/eval_flags.jsonl` | Flagged queries only | JSONL (append) | `!flags` command | When eval detects LOW faithfulness/relevance |
| `data/users.json` | User profiles | JSON | Startup, debug commands | Every user interaction |
| `data/telegram_offset.json` | Offline message offset | JSON | Telegram bot startup, `!offset` command | Every message processed (online + offline) |
| `data/cli_user.txt` | CLI persistent user ID | Text | CLI startup | First CLI run |
| `data/db/bm25_cache.pkl` | BM25 serialized index | Pickle | Startup (if hash valid) | After index rebuild (dataset change) |
| `data/db/bm25_cache.hash` | BM25 cache validation hash | Text | Startup | After index rebuild |

---

## ⚙️ Configuration Files Reference

### `config/.env` — Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PERENUAL_API_KEY` | Yes (for plant API) | — | Perenual plant API key |
| `MONGO_URI` | Yes (for insights) | — | MongoDB Atlas connection string |
| `TELEGRAM_BOT_TOKEN` | Yes (for Telegram) | — | Bot token from @BotFather |
| `CHROMA_HOST` | No | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | No | `8000` | ChromaDB server port |
| `EMBED_MODEL` | No | `mxbai-embed-large` | Ollama embedding model |
| `LLM_MODEL` | No | `sugi-v0.1L` | Ollama main response model |
| `UTILITY_MODEL` | No | `qwen2.5:1.5b` | Ollama utility model (rewriting, eval) |
| `MEMORY_TTL_DAYS` | No | `14` | Days to keep conversation memories |
| `BM25_CACHE_PATH` | No | `bm25_cache.pkl` | Path to BM25 cache file |
| `DEBUG_ALLOWED_USERS` | No | (empty — **all debug disabled**) | Comma-separated Telegram user IDs for `/debug` + `!`-commands (R1: fail-closed — empty denies everyone with a startup warning) |
| `LATITUDE` | No | `-6.1818` | Weather data latitude (Jakarta) |
| `LONGITUDE` | No | `106.8223` | Weather data longitude (Jakarta) |
| `LOCATION_NAME` | No | `Jakarta` | Display name for weather location |
| `DAILY_INSIGHT_GRACE_SECONDS` | No | `180` | Daily insight startup grace before first run (A8) |
| `GOVERNMENT_INSIGHT_GRACE_SECONDS` | No | `240` | Government insight startup grace (A8) |
| `FARMER_INSIGHT_GRACE_SECONDS` | No | `300` | Farmer insight startup grace (A8) |
| `STARTUP_GRACE_SECONDS` | No | `180` | Fallback grace for all services (A8) |
| `INSIGHT_LLM_DELAY` | No | `1` | Seconds between insight LLM calls (A8/A12); `0` = off |
| `DAILY_INSIGHT_MAX_WORKERS` | No | `1` | Daily insight LLM workers (A12): `1` = sequential + pacing; `>1` = bounded thread pool |
| `MAX_CONCURRENT_ASK` | No | `2` | Backpressure semaphore `asyncio.Semaphore(2)` + `wait_for 0.01` fast-fail busy `⏳` (T1-1) |
| `MAX_QUESTION_CHARS` | No | `2000` | Input length guard early return `<1ms` vs `120s` timeout (T1-2) |
| `OLLAMA_HOST_INSIGHT` | No | `http://127.0.0.1:11435` | Insight `qwen` Ollama host (P2-2) — fallback `11434` if unset; `keep_alive 60s` G2 |

### `config/settings/scope_config.ini` — Domain Guard

- **`[greetings]`** — Greeting phrases that bypass scope check
- **`[allowed_*]`** — 8 allowed topic sections (pertanian, hama, cuaca, pangan, komoditas, harga, teknologi, agribisnis, english)
- **`[blocked_topics]`** — Forbidden keywords (health, politics, tech, etc.)
- **`[refusal]`** — Custom refusal message

### `config/settings/rewriter_config.ini` — Query Rewriting

- **`[referential_words]`** — Words indicating reference to prior context (itu, ini, tersebut, dll.)
- **`[referential_suffixes]`** — Suffixes like `-nya` that indicate implicit reference
- **`[followup_patterns]`** — Regex patterns for implicit followup questions
- **`[topic_keywords]`** — Non-plant topic labels for context resolution

### `config/settings/plant_keywords.ini` — Plant Detection

- **`[plant_name_map]`** — Indonesian → English plant name mapping for Perenual API
- **`[keywords_strong]`** — Specific plant names that directly trigger plant API
- **`[keywords_weak]`** — Generic terms that only trigger when paired with strong keywords

---

## 🔄 Query Processing Pipeline

```
User Question
    │
    ▼
┌─────────────────────────────────────────────┐
│ [0] INPUT GUARDS + BUSY SIGNAL               │
│   ├── MAX_QUESTION_CHARS 2000 → early return │  T1-2 (<1ms)
│   ├── MAX_CONCURRENT_ASK 2 → busy 0.01s     │  T1-1 Telegram semaphore
│   └── Touch data/busy/<user>_<ns>.flag      │  Part0 (insight wait_if_busy 60s)
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ [1] SCOPE GUARD                              │  scope 0.00-0.02s
│   ├── Greeting → bypass                      │
│   ├── Blocked → REFUSAL (27ms)              │  R2 (blocked overrides allowed)
│   ├── Allowed → continue                     │
│   └── No match → REFUSAL                     │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ [2] QUERY REWRITING (3-layer)                │  rewrite 0.000004-0.04s
│   ├── Definitional? ("apa itu X") → skip     │
│   ├── Has referential? → rule 0ms (11435)   │  H2a
│   └── Ambiguous? → Qwen 11435 2.27s (30s)   │
│   Scope re-check: suffix can bypass, word cannot │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ [3] PLANT & WEATHER DETECTION                │  plant 0.008-2.16s
│   ├── Plant: strong? → plant API            │  Session pooled PF2-1 1.59s vs 2.79s
│   │   ├── weak+strong? → plant API          │  negative cache 24h 0.001s PF-F2
│   │   ├── weak only? → skip                 │
│   │   └── Qwen 11435 10 tok → Perenual      │  parallel species‖disease PF2-1
│   └── Weather → weather docs (k=8)         │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ [4] HYBRID RETRIEVAL (BM25 fallback)         │  retrieval 0.60-3.48s (embed 0.031s pooled)
│   ├── BM25 k=4 (always)                      │  PF-V1 Session
│   ├── Vector k=4 (pooled 0.031s)            │  Chroma /api/v2, fallback BM25-only P4-5
│   ├── Plant k=3 (if plant)                  │
│   └── Memory k=2 (if referential)           │  needs_ref_context A10
│   Ensemble 0.60/0.15, backlog 50 V1b, cap 6 Part4 │
│   [STAGE_TIMING] + [GENERATION_SPLIT]       │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ [5] CROSS-ENCODER RERANKER (background)      │  rerank 0.15-0.50s (warm 0.055s)
│   └── ms-marco-MiniLM-L-6-v2 top_n=8        │  QW-3 4.19s background load, wait 10s
│   └── single pass (A11) over RAG±weather    │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ [6] LLM GENERATION (prefill/eval split)      │  generation 6.3-13.0s (48-85%)
│   ├── Template v1 + date/time + 6 chunks    │  Part4 3680 vs 3714 -1% prefill 40-73%
│   ├── 4096 ctx, timeout 180 PF2-V1, 400 tok  │  residual <0.03s PF2-3
│   ├── on_chunk → edit_text 200c/1s P2-4     │  streaming
│   └── /ask/stream NDJSON PF2-2 TTFT 1.0s warm│  (22.59s cold 15s prefill vs 0.02s warm)
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ [7] EVAL LOOP (background daemon)            │
│   ├── Lexical 0.50/0.25 S1 (Sastrawi stem)   │
│   └── Qwen2.5 LLM if inconclusive (15s)     │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│ [8] PERSISTENCE + OBSERVABILITY              │
│   ├── queries.jsonl 12 hex, idempotent      │  Part1 dups 20→0
│   ├── model_name/prompt_version/generation_split │ PF2-3
│   ├── busy flags + degraded disclaimer      │  Part0/P4-5
│   ├── Every 5 turns → session memory        │
│   ├── !debug p50/p95 13.02/35.75s          │  2026-09-09 recent
│   └── !stats In-flight 0/2 + !offset       │
└─────────────────────────────────────────────┘
```

---

## 🧪 Debug Commands

Available in both CLI and Telegram:

| Command | Description |
|---------|-------------|
| `!debug` | Shows last 10 queries with latency, faithfulness, relevance |
| `!flags` | Lists all queries flagged by eval loop (LOW scores) |
| `!session` | Shows log for current session only |
| `!memory` | Displays long-term memory stored for current user |
| `!stats` | Shows system stats (active users, models, enabled features) |
| `!offset` | (Telegram only) Shows current Telegram update offset |

CLI-only commands: `q` (quit), `clear` (reset session), `history` (show past conversation)

---

## 🚀 Running the System

### Prerequisites

- Python 3.10+
- Ollama with models:
  ```bash
  ollama pull llama3.2
  ollama pull qwen2.5:1.5b
  ollama create sugi-v0.1L -f Modelfile
  ollama pull mxbai-embed-large
  ```

### Setup

```bash
git clone https://github.com/dery45/SUGI-v0.1L-Telegram-Bot.git
cd SUGI-v0.1L-Telegram-Bot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Start Services

**Terminal 1 — ChromaDB:**
```bash
chroma run --path data/db --port 8000
```

**Terminal 2 — All SUGI services:**
```bash
python start_all.py
```
This starts 7 background processes with auto-restart:
1. CSV/XLSX Watcher (`vectorCSV.py`)
2. PDF Watcher (`vectorpdf.py`)
3. Weather Service (`vectorWeather.py`)
4. Daily Insight Cron (`daily_insight.py`)
5. Government Insight Engine (`government_insight_service.py`)
6. Farmer Insight Engine (`farmer_insight_service.py`)
7. Telegram Bot (`telegram_bot.py`)

**Alternative — CLI only:**
```bash
python interfaces/cli/main.py
```

---

## 📁 Complete Project Structure

```
SUGI-v0.1L/
├── config/                          # Configuration
│   ├── .env                         # Environment variables (git-ignored)
│   ├── .env.example                 # Template with all variables
│   ├── Modelfile                    # Ollama sugi-v0.1L model definition (llama3.2 tuned)
│   └── settings/                    # INI-based runtime config (no code changes needed)
│       ├── scope_config.ini         # 8 allowed topic sections + blocked topics + greeting patterns
│       ├── rewriter_config.ini      # Referential words, suffixes, followup regex, topic map
│       └── plant_keywords.ini       # 60+ plant name mappings, strong/weak detection keywords
├── core/                            # Shared Engine Logic (platform-agnostic)
│   ├── sugi_core.py                 # Main RAG pipeline (~1400 lines): SugiCore class
│   │   ├── Query rewriting (3-layer)
│   │   ├── Scope guard
│   │   ├── Plant/weather detection
│   │   ├── Dynamic ensemble retriever builder
│   │   ├── LLM answer generation with streaming
│   │   ├── Session & memory management
│   │   └── Debug command handler
│   ├── plant_api.py                 # Perenual API client with rate limiting & caching
│   │   ├── Species search + detail fetch
│   │   ├── Pest/disease fetch
│   │   ├── Care guides fetch
│   │   ├── Token bucket rate limiter (1 req/s)
│   │   └── ChromaDB-backed cache with 30-day TTL
│   ├── eval_loop.py                 # Faithfulness + relevance scoring
│   │   ├── Lexical heuristic (fast path)
│   │   └── Qwen2.5 LLM eval (fallback)
│   ├── query_logger.py              # Structured JSONL logging engine
│   │   ├── Per-query trace with latency, docs, eval
│   │   ├── Debug report generator
│   │   └── Flagged query collector
│   └── user_store.py                # Persistent user profiles (JSON file)
├── services/                        # Background Long-Running Services
│   ├── vectorCSV.py                 # CSV/XLSX watcher & indexer
│   │   ├── Watchdog-based file monitoring
│   │   ├── Multi-encoding CSV reader (utf-8, latin-1, cp1252)
│   │   ├── XLSX multi-sheet reader
│   │   ├── Auto-detect data type (price/policy/tabular) → adaptive chunk size
│   │   └── BM25 cache invalidation + content-hash re-index (R4)
│   ├── vectorpdf.py                 # PDF watcher & indexer
│   │   ├── Watchdog-based file monitoring
│   │   ├── PyPDFLoader with adaptive chunk size (regulasi=800, jurnal=600, harga=250)
│   │   ├── Content-hash re-index (R4) + BM25 cache invalidation
│   │   └── Shares main_dataset collection with vectorCSV
│   ├── vectorWeather.py             # Open-Meteo weather crawler
│   │   ├── Fetches 92-day history + 16-day forecast
│   │   ├── Hourly → daily aggregation with agronomic alerts
│   │   ├── Alerts: drought, flood, heat stress, disease risk, wind
│   │   └── SQLite HTTP cache (1-hour TTL)
│   ├── daily_insight.py             # MongoDB insight generator (12h cron)
│   │   ├── Price insights per province
│   │   ├── Weather insights per location
│   │   ├── Planting suggestions per commodity × season
│   │   ├── General policy/trend insights
│   │   └── Session summary sync from ChromaDB memory
│   └── government_insight_service.py # Government Insight Engine (3600s cron)
│       ├── Generates insights from 14 government data collections
│       ├── Change detection via MD5 signature of last 100 IDs
│       ├── ChromaDB learning loop (government_memory collection)
│       ├── SUGI ecosystem context (RAG + weather + past insights)
│       └── Monthly full refresh at 30 days
│   ├── farmer_insight_service.py     # Farmer Insights & Policy Recommendation Engine
│       ├── 10 farmer market insights (≤150 chars each, saved to farmerinsights)
│       ├── 1 Government Policy Recommendation (400-500 chars, saved to governmentinsights)
│       ├── Data change detection with selective regeneration
│       ├── Multi-stage validation: no brackets, no truncation, Bahasa Indonesia only
│       ├── Automatic retry on validation failure (up to 3 attempts)
│       ├── Example-driven prompting for concise, data-first insight style
│       ├── ChromaDB learning loop (insights_memory collection)
│       ├── SUGI ecosystem enrichment (RAG, weather, plant API, historical memory)
│       └── Daily automatic refresh + change-triggered regeneration
├── interfaces/                      # User-Facing Entry Points
│   ├── cli/
│   │   └── main.py                  # Terminal chat client (persistent user ID)
│   └── telegram/
│       ├── telegram_bot.py          # Full Telegram bot (~580 lines)
│       │   ├── Multi-user sessions
│       │   ├── Offline message catch-up with offset persistence
│       │   ├── Fail-closed debug access (R1) + global error handler (R3)
│       │   ├── Rate limiting (3s per user)
│       │   ├── Contact sharing
│       │   └── Typing indicator
│       ├── requirements_telegram.txt
│       └── DEPLOYMENT_GUIDE.md
├── tests/                           # Internal verification scripts (git-ignored)
├── data/                            # Runtime Data (all git-ignored)
│   ├── db/                          # ChromaDB persistent storage
│   │   ├── bm25_cache.pkl           # Serialized BM25 retriever
│   │   └── bm25_cache.hash          # MD5 hash for cache invalidation
│   ├── logs/
│   │   ├── queries.jsonl            # Structured query audit log
│   │   └── eval_flags.jsonl         # Flagged query log (eval LOW)
│   ├── raw_dataset/                 # CSV/XLSX source files (auto-indexed)
│   ├── raw_pdfs/                    # PDF source files (auto-indexed)
│   ├── users.json                   # User profile database
│   ├── cli_user.txt                 # Persistent CLI user identity
│   └── telegram_offset.json         # Telegram update offset (crash-safe)
├── image/                           # Screenshots
├── start_all.py                     # Master launcher with health checks & auto-restart
├── migrate_to_server.py             # ChromaDB embedded → server migration tool
├── generate_questions.py            # Seed question generator
├── test_mongo.py                    # MongoDB connection test
├── docs/                            # Internal docs — only CHANGELOG.md tracked
│   ├── CHANGELOG.md                 # Versioned changelog (tracked, committed)
│   └── VERSIONS.md                  # Concise version overview (git-ignored)
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🔬 Eval Loop Scoring

After every answer, the system scores:

| Metric | Method | Levels | Flagged? |
|--------|--------|--------|----------|
| **Faithfulness** | Lexical overlap (40% threshold) → Qwen2.5 LLM if inconclusive | HIGH / MEDIUM / LOW | If LOW |
| **Relevance** | Keyword overlap (50% threshold) → Qwen2.5 LLM if inconclusive | HIGH / MEDIUM / LOW | If LOW |
| **No docs** | Zero documents retrieved | — | Always flagged |

Flagged queries are written to `data/logs/eval_flags.jsonl` and visible via `!flags` command.

---

## 🌐 External API Reference

### Open-Meteo (Free, no API key)
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Data**: Temperature, humidity, precipitation, wind, soil moisture, evapotranspiration
- **Range**: 92 days history + 16 days forecast
- **Rate limit**: None documented (respectful usage)
- **Cache**: SQLite via requests-cache (1 hour)

### Perenual Plant API
- **Endpoint**: `https://perenual.com/api/v2/species-list`, `/species/details/{id}`, `/pest-disease-list`, `/species-care-guide-list`
- **Auth**: API key via query parameter
- **Rate limit**: ~60 req/min (free tier) — enforced by token bucket (1.1s interval)
- **Retry**: Flat retry on 429 (1.5s backoff, max 0 retries → fast fail)
- **Failure mode**: On 429 → API disabled for 1 hour, falls back to local RAG

---

## Testing

```bash
# Scope guard tests
python tests/test_scope_leak.py

# Referential query fix verification
python tests/verify_fix.py

# MongoDB connection test
python test_mongo.py
```

---

## RAG Score: 94/100

Evaluated on:
- **Faithfulness**: Answer stays within retrieved context
- **Relevance**: Retrieved documents match query intent
- **Scope Accuracy**: Correctly blocks out-of-scope queries
- **Rewrite Quality**: Rule-based handles 85% of referential cases
- **Latency**: Average <5s per query (BM25 cache + streaming generation)
- **Ecosystem Integration**: Insight engines enrich context with government & farmer analytics via ChromaDB learning loop

---

## 📋 Changelog & Version History

> Full details live in: [`docs/CHANGELOG.md`](docs/CHANGELOG.md) (detailed chronological changelog — the **only** tracked file in `docs/`; the rest is internal-only) and [`docs/VERSIONS.md`](docs/VERSIONS.md) (concise overview, git-ignored).

**Current status:** v0.4.0 (✔ Committed; HEAD `3d5cdff` v0.2.0, 2026-08-12) — Phase 2-4 complete, outlier fix verified, prompt trim, Chroma fallback, combined load validated (see decisions.md Phase 3-4 + `PHASE1_REVIEW_RERUN_20260907.md` 2026-09-09 17/17 PASS 262.92s)

| Version | Date | Status | Summary |
|---|---|---|---|
| **v0.4.0** | 2026-09-09 | ✔ Committed (Phase 2-4 + P4-1-6) | **Outlier fix verified:** 0/12 genuine >90s since Part0 vs 8/493=1.62% before **measured**, dups 20→0, recent p95 35.75s (was 94.21s); **PF-V1** embedding `Session` pooling 2.06s→0.031s **measured**; **PF2-V1** timeout 90→180 reconciled vs history p95 51s max 140s; **PF2-1** Perenual `Session` + `species‖disease` parallel 2.795s→1.528s (pooling 1.20s + parallel 0.06s) **measured**, negative cache 24h 0.001s; **PF2-2** `POST /ask/stream` NDJSON TTFT 1.0-1.3s warm **measured** (22.59s cold→1.01s warm); **PF2-3** `GENERATION_SPLIT` prefill 40-73% (0.02-7.1s) residual <0.03s **measured**; **Part0** busy flag `data/busy/*.flag` + `wait_if_busy` 60s defer **measured** 4s; **Part1** `query_id` 12 hex + idempotent `commit_trace`; **Part4** cap 8→6 (3680 vs 3714 -1% **measured**, 0/25 flagged); **P4-4** QW-3 background reranker 4.19s overlapped + QW-5 `invoke("Halo")` warm-up; **P4-5** Chroma fallback BM25-only + disclaimer; **P4-6** 5 concurrent+insight p95 49.81s **measured** well below 94-365s — busy holds |
| **v0.3.0** | 2026-09-07 | ✔ Committed (Phase 1-5) | **Scalability remediation complete:** T1-1 backpressure `MAX_CONCURRENT_ASK=2` (0.607s `2 ok 3 busy`), T1-2 input guard `2000` (`<1ms` vs `120s`), T1-3 `UserStore` lock+atomic; V1b `backlog 50` corrects `10/10` wall `45-50s` not hard wall; V2b `OLLAMA_NUM_PARALLEL=2` negative `121s` vs `27s` (compute-bound, keep `1`); P2-2 dual Ollama `11435` (`1.1GB` qwen `keep_alive 60s` G2, `5.4-5.6GB` VRAM headroom corrected); P2-3 observability `model_name`/`prompt_version` `p50/p95` `In-flight`; P2-4 streaming `on_chunk 200c/1s`; G1 GPU `100% VRAM` Vulkan RX 6600 (`15.2s` median); H1 prefill `0.2-7.1s` vs generation `5.9s` mixed; H2a utility redirect to `11435` (ambiguous), H2b embed `0.04s` not worth; P3-1 `num_predict 400`; I1 `STAGE_TIMING` 6 stages explains `20-30s` tail (`generation 48-85%`); Final disposition: **do not build** sharding/Redis/K8s — 3 triggers gated |
| **v0.2.3** | 2026-08-24 | ✔ Committed (T1–T2) | T1 whitespace normalization at `ask()` entry (fixes "hari  ini" double-space → false referential → Qwen timeout → memory injection); T2 data-narration removal: prompt hardening (silent gap-fill, silent history, forbidden-opener list, anti-refusal backstop) + runtime post-filter `_strip_data_narration()` strips "Saya tidak menemukan informasi", "Berdasarkan riwayat...", anti-refusal backstop |
| **v0.2.2** | 2026-08-18 | ✔ Committed (N1, S1, S2) | N1 province metadata hoist for price insights (hoist `province` to chunk metadata + schema-versioned R4 reindex), S1 stemmed lexical eval overlap (Sastrawi `_stem_tokens`, graceful fallback, thresholds 0.50/0.25, thread lock), S2 scalability roadmap decision recorded (Option B: descoped Redis sessions + second Ollama at current scale) |
| **v0.2.1** | 2026-08-12 | ✔ Committed (R1–R4 + Q1–Q2) | R1 fail-closed debug access (`DEBUG_ALLOWED_USERS` deny-by-default + startup warning), R2 `blocked_kw` enforced (blocked overrides allowed), R3 global Telegram error handler (fallback reply instead of silence), R4 content-hash re-index dedup (CSV/XLSX/PDF `file_hash`, delete+reindex on change) + BM25 invalidation backfill for PDFs, Q1 tag comments consolidated into `docs/decisions.md`, Q2 duplicated insight helpers extracted into shared `services/insight_common.py` (per-service k/label/truncation preserved) |
| **v0.2.0** | 2026-08-12 | ✔ Committed (`f53db8e`) | Background eval (daemon thread), model residency `keep_alive=600`, eval stability (`num_ctx`/`num_predict`/timeout), parallel plant-detail fetch, scope-guard fix for `penanaman`, README DB schema (6 ChromaDB collections), `docs/CHANGELOG.md` changelog added & tracked, `tests/` git-ignored, A9 multi-turn history-poisoning fix, A8 insight-engine startup stagger (180/240/300s) + 240s LLM timeout, A10 retrieval-level memory gating + A4 closure (unfiltered `memory_retriever` removed), A11 startup-429 NameError fix + single-pass rerank + dead `include_weather`/`weather_retriever` removal + eval `num_predict=10`, A12 daily_insight concurrency backfill (pools → sequential `_run_paced` + `INSIGHT_LLM_DELAY`, `DAILY_INSIGHT_MAX_WORKERS` env) + A7 ping-retry (`_ping_with_retry` 5s/10s) |
| **v0.1.9** | 2026-07-13 | ✔ Committed | Government & Farmer Insight Engines (+ChromaDB `government_memory`/`insights_memory`, MongoDB output) |
| **v0.1.8** | 2026-06-03 | ✔ Committed | README/docs refresh + demo screenshots |
| **v0.1.7** | 2026-04-13 | ✔ Committed | Telegram offline message catch-up, crash-safe offset persistence |
| **v0.1.6** | 2026-03-27 | ✔ Committed | Insight pipeline API fixes, health-check wiring |
| **v0.1.5** | 2026-03-26 | ✔ Committed | Thread safety (Telegram), streaming, Perenual API validation |
| **v0.1.4** | 2026-03-25 | ✔ Committed | Chains Q&A + context-leak fix, scope gating, memory bug fix, startup health checks |
| **v0.1.3** | 2026-03-24/25 | ✔ Committed | Repo hygiene + modular restructure (`core/ services/ interfaces/`, `start_all.py`) |
| **v0.1.2** | 2026-03-15 | ✔ Committed | ChromaDB migration, `qwen2.5:1.5b` utility model, daily insight + bulk BM25, word config |
| **v0.1.1** | 2026-03-15 | ✔ Committed | Rule-based rewriter + date logic, eval loop (faith/relevance/flag), query logging |
| **v0.1.0** | 2026-03-15 | ✔ Committed | Initial release scaffolding (RAG chatbot, Modelfile, ingestion, question gen) |

> Versioning note: the repository has no Git tags. Milestones above are reconstructed from `git log` by the changelog; product branding remains **v0.1L**.

---

## License
MIT License.

Last updated: 2026-09-09 · v0.4.0 (Outlier fix verified + busy signal + pooling/parallel + streaming TTFT + generation split + cap 6 + fallback) · `PHASE1_REVIEW_RERUN_20260907.md` 17/17 PASS 262.92s · RAG Score 94/100
