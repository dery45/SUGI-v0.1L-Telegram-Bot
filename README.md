# 🌾 SUGI v0.1L – Asisten Pertanian Cerdas Indonesia

**S**istem **U**tama **G**enerative **I**ntelijen  
Asisten AI berbasis RAG khusus untuk petani, pekebun, pemerintah, dan pelaku agribisnis di Indonesia & Asia Tenggara.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green?logo=ollama)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB%20Server-purple)](https://www.trychroma.com/)
![License](https://img.shields.io/badge/License-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/ux/sugi-v0.1L?color=green)

## 🔥 Fitur Utama

- **RAG Hybrid** → BM25 + Dense Vector (mxbai-embed-large) + Cross-Encoder Reranker  
- **Dynamic Retriever** → bobot otomatis berdasarkan jenis query (tanaman/cuaca/history)  
- **Query Rewriting 3 lapis** → rule-based (0ms) → Qwen2.5 fallback (~800ms) → original  
- **Data Real-time Pertanian**  
  - Cuaca harian + alert agronomi (kekeringan, banjir, heat stress, penyakit) dari Open-Meteo  
  - Informasi tanaman lengkap (spesies, hama, penyakit, panduan perawatan) via Perenual API + cache  
  - Indexing otomatis CSV/XLSX/PDF (harga komoditas, panduan budidaya, dll.)  
- **Long-term Memory** → ringkasan sesi disimpan di ChromaDB untuk konteks multi-turn  
- **Scope Guard Ketat** → hanya jawab topik pertanian, tolak hal di luar domain dengan ramah  
- **Personality** → ramah & mudah dipahami petani, profesional untuk pemerintah/investor  
- **Bahasa** → default Indonesia, switch ke English jika user pakai English  


## Tech Stack

- **LLM** : Llama 3.2 fine-tune → sugi-v0.1L (via Ollama)  
- **Utility model** : Qwen2.5-1.5B — query rewriting fallback, plant name extraction, eval loop  
- **Embedding** : mxbai-embed-large  
- **Vector Store** : ChromaDB Server (collections: langchain, weather_data, plant_data, conversation_memory)  
- **Retriever** : Ensemble (BM25 + Vector) + Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2)  
- **API Eksternal** : Open-Meteo (cuaca), Perenual (tanaman & hama)  
- **Lainnya** : LangChain, Watchdog (auto-index)  


## Instalasi (Local Development)

### 1. Prasyarat

- Python 3.10+
- Ollama terinstall & jalankan model:
  ```bash
  ollama pull llama3.2
  ollama pull qwen2.5:1.5b   # utility model: rewriting + plant extraction + eval
  ollama create sugi-v0.1L -f Modelfile
  ollama pull mxbai-embed-large
  ```

  > Kalau sebelumnya pakai phi3, bisa dihapus setelah Qwen berjalan normal:
  > ```bash
  > ollama rm phi3
  > ```

### 2. Clone & Setup

```bash
git clone https://github.com/ux/sugi-v0.1L.git
cd sugi-v0.1L

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Konfigurasi

Buat file `.env` di root:
```env
PERENUAL_API_KEY=sk-your-perenual-key-here
```

Sesuaikan lokasi di `vectorWeather.py` jika bukan Jakarta:
```python
LATITUDE      = -7.7956   # contoh: Yogyakarta
LONGITUDE     = 110.3695
LOCATION_NAME = "Yogyakarta"
```

### 4. ChromaDB Server ⚠️ Wajib dijalankan sebelum semua script lain

SUGI menggunakan ChromaDB dalam **server mode** agar semua proses (indexing + serving) bisa berjalan bersamaan tanpa konflik file.

**Jalankan server di Terminal 1 — biarkan tetap terbuka:**

```bash
# Linux / Mac
chroma run --path ./chrome_longchain_db --port 8000

# Windows (kalau chroma sudah di PATH)
chroma run --path ./chrome_longchain_db --port 8000

# Windows (kalau chroma belum di PATH)
python -m chromadb.cli.cli run --path ./chrome_longchain_db --port 8000
```

**Verifikasi server berjalan:**
```bash
curl http://localhost:8000/api/v1/heartbeat
# Output: {"nanosecond heartbeat": ...}
```

> **Windows — tambah chroma ke PATH (sekali saja):**  
> Buka PowerShell sebagai Administrator, jalankan:
> ```powershell
> [Environment]::SetEnvironmentVariable(
>   "PATH",
>   [Environment]::GetEnvironmentVariable("PATH","User") + ";" +
>   (python -c "import sys; print(sys.prefix + '\\Scripts')"),
>   "User"
> )
> ```
> Tutup dan buka kembali terminal, lalu test dengan `chroma --version`.

**Migrasi data lama (hanya perlu dilakukan sekali jika sudah punya `chrome_longchain_db/`):**
```bash
# Pastikan server sudah jalan di terminal lain dulu
python migrate_to_server.py
```

### 5. Jalankan Server Vector (terminal terpisah)

```bash
python vectorCSV.py      # Terminal 2 — index CSV/XLSX dataset
python vectorpdf.py      # Terminal 3 — index PDF
python vectorWeather.py  # Terminal 4 — auto-refresh cuaca setiap 5 menit
```

> Semua terminal bisa berjalan bersamaan karena terhubung ke ChromaDB server yang sama.

### 6. Jalankan SUGI

```bash
python main.py           # Terminal 5
```

Ketik pertanyaan pertanian, ketik `q` untuk keluar & simpan memory sesi.

**Debug commands (ketik langsung di prompt SUGI):**
```
!debug    → ringkasan 10 query terakhir (latency, eval score, doc count)
!flags    → semua query yang di-flag eval (faithfulness/relevance rendah)
!session  → log sesi yang sedang berjalan
```

### Urutan startup lengkap

```
Terminal 1:  chroma run --path ./chrome_longchain_db --port 8000
Terminal 2:  python vectorCSV.py
Terminal 3:  python vectorpdf.py
Terminal 4:  python vectorWeather.py
Terminal 5:  python main.py
```


## Struktur Folder

```
sugi-v0.1L/
├── dataset/                  # taruh CSV/XLSX harga, panduan budidaya
├── pdfsource/                # taruh PDF regulasi, jurnal pertanian
├── chrome_longchain_db/      # ChromaDB data dir (jangan commit — ada di .gitignore)
├── logs/                     # query logs JSONL (jangan commit — ada di .gitignore)
│   ├── queries.jsonl         # semua query + eval result
│   └── eval_flags.jsonl      # query yang di-flag (faithfulness/relevance rendah)
├── word_config/              # konfigurasi keyword (edit tanpa ubah kode)
│   ├── scope_config.ini      # allowed/blocked topics + greeting patterns
│   ├── rewriter_config.ini   # referential words, followup patterns, topic keywords
│   └── plant_keywords.ini    # plant name map + strong/weak keywords
├── main.py                   # program utama
├── vectorCSV.py              # indexer CSV/XLSX
├── vectorpdf.py              # indexer PDF
├── vectorWeather.py          # fetcher + indexer cuaca Open-Meteo
├── plant_api.py              # Perenual API client + cache
├── eval_loop.py              # RAG eval (faithfulness + relevance)
├── query_logger.py           # structured query logging ke JSONL
├── migrate_to_server.py      # migrasi data dari embedded → server mode
├── Modelfile                 # definisi model sugi-v0.1L (Llama 3.2 base)
├── .env                      # API keys (jangan commit)
├── .env.example
├── .gitignore
└── requirements.txt
```


## Konfigurasi Lanjutan

### Ubah utility model (Qwen2.5-1.5B)
Di `main.py`, ubah konstanta `UTILITY_MODEL`:
```python
UTILITY_MODEL = "qwen2.5:1.5b"   # ganti dengan model Ollama lain jika perlu
```
Konstanta ini dipakai oleh tiga komponen sekaligus: query rewriting fallback, plant name extraction, dan eval loop — sehingga cukup ubah satu baris.

### Ubah keyword scope tanpa restart
Edit `word_config/scope_config.ini`, tambahkan keyword baru di section yang sesuai, lalu restart `main.py`.

### Tambah tanaman baru ke plant map
Edit `word_config/plant_keywords.ini`, tambahkan baris `nama_indonesia = english_name` di section `[plant_name_map]`. Frasa lebih panjang harus di atas yang lebih pendek.

### Ubah TTL memory sesi
Di `main.py`, ubah nilai `MEMORY_TTL_DAYS` (default 14 hari). Set ke `0` untuk mematikan TTL.

### Ubah lokasi ChromaDB server
Di semua file `*.py`, ubah:
```python
CHROMA_HOST = "localhost"   # ganti dengan IP/hostname server
CHROMA_PORT = 8000          # ganti port jika perlu
```


## Lisensi

MIT License – bebas digunakan, dimodifikasi, dan dikembangkan lebih lanjut.

Last updated: Maret 2026 · v0.1L (Qwen2.5-1.5B upgrade)