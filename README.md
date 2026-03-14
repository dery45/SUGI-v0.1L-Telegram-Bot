# 🌾 SUGI v0.1L – Asisten Pertanian Cerdas Indonesia

**S**istem **U**tama **G**enerative **I**ntelijen  
Asisten AI berbasis RAG khusus untuk petani, pekebun, pemerintah, dan pelaku agribisnis di Indonesia & Asia Tenggara.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green?logo=ollama)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)](https://www.trychroma.com/)
![License](https://img.shields.io/badge/License-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/ux/sugi-v0.1L?color=green)

## 🔥 Fitur Utama

- **RAG Hybrid** → BM25 + Dense Vector (mxbai-embed-large) + Cross-Encoder Reranker  
- **Dynamic Retriever** → bobot otomatis berdasarkan jenis query (tanaman/cuaca/history)  
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
- **Embedding** : mxbai-embed-large  
- **Vector Store** : ChromaDB (separate collections: default, weather, plant, memory)  
- **Retriever** : Ensemble (BM25 + Vector) + Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2)  
- **API Eksternal** : Open-Meteo (cuaca), Perenual (tanaman & hama)  
- **Lainnya** : LangChain, Watchdog (auto-index), phi3 (query rewrite & plant extraction)  

## Instalasi (Local Development)

1. **Prasyarat**
   - Python 3.10+
   - Ollama terinstall & jalankan model:
     ```bash
     ollama pull llama3.2
     ollama pull phi3     # untuk rewrite & plant extraction
     ollama create sugi-v0.1L -f Modelfile  # gunakan Modelfile kamu
     ollama pull mxbai-embed-large
     ```

2. **Clone & Setup**
   ```bash
   git clone https://github.com/ux/sugi-v0.1L.git
   cd sugi-v0.1L

   # Buat virtualenv
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Konfigurasi**
   - Buat file `.env` di root:
     ```env
     PERENUAL_API_KEY=sk-your-perenual-key-here
     ```
   - Sesuaikan `vectorweather.py`: LATITUDE & LONGITUDE (default Jakarta, ubah ke Yogyakarta: -7.7956, 110.3695)

4. **Jalankan Server Vector (background)**
   ```bash
   python vectorcsv.py    # terminal 1
   python vectorpdf.py    # terminal 2
   python vectorweather.py # terminal 3 (auto-refresh cuaca)
   ```

5. **Jalankan SUGI**
   ```bash
   python main.py
   ```

   Ketik pertanyaan pertanian, ketik `q` untuk keluar & simpan memory.

## Struktur Folder

```
sugi-v0.1L/
├── dataset/              # taruh CSV/XLSX harga, panduan budidaya
├── pdfsource/            # taruh PDF regulasi, jurnal pertanian
├── chrome_longchain_db/  # ChromaDB persist (jangan commit)
├── *.py                  # vectorcsv, vectorpdf, vectorweather, plant_api, main
├── Modelfile             # definisi sugi-v0.1L
├── scope_config.ini      # allowed/forbidden keywords
├── .env.example
└── requirements.txt
```

## Cara Kontribusi

1. Fork repo ini  
2. Buat branch `feature/nama-fitur`  
3. Commit & push  
4. Buat Pull Request dengan deskripsi jelas  

Kami sangat terbuka untuk:  
- Integrasi harga komoditas real-time  
- UI web (Streamlit/Gradio)  
- Dukungan bahasa daerah (Jawa/Sunda)  
- Tambah sumber data (satelit NDVI, BMKG API, dll.)

## Lisensi

MIT License – bebas digunakan, dimodifikasi, dan dikembangkan lebih lanjut.  

Last updated: Maret 2026
