"""
daily_insight.py — SUGI Daily Insight Engine
=============================================
Menghasilkan insight harian dari data RAG lokal (ChromaDB) dan mengirimkannya
ke MongoDB dalam koleksi terpisah per kategori.

Koleksi MongoDB:
  sugi_insights.price_insights        — harga komoditas per provinsi
  sugi_insights.weather_insights      — cuaca & prakiraan tanam
  sugi_insights.planting_suggestions  — rekomendasi tanam per komoditas/musim
  sugi_insights.general_insights      — insight umum pertanian / kebijakan / tren
  sugi_insights.session_summaries     — ringkasan sesi percakapan dari main.py

Jadwal:
  - Selalu kirim saat pertama kali dijalankan
  - Setelah itu kirim otomatis setiap 12 jam (bisa ubah INTERVAL_HOURS)

Env (.env):
  MONGO_URI   — MongoDB connection string (wajib)
  PORT        — tidak dipakai di sini, disimpan untuk konsistensi env

Cara pakai:
  python daily_insight.py               # jalankan & biarkan loop
  python daily_insight.py --once        # kirim sekali lalu exit
"""

import argparse
import os
import sys
import time
import hashlib
import traceback
from datetime import datetime, timezone
from typing import Any
import concurrent.futures

# ── Env loading ───────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_ROOT / "config" / ".env")
except ImportError:
    # dotenv opsional — bisa set env variable manual
    pass

MONGO_URI      = os.getenv("MONGO_URI", "")
INTERVAL_HOURS = 12          # interval antar pengiriman (ubah sesuai kebutuhan)
BATCH_SIZE     = 500         # batch size saat fetch dari ChromaDB (hindari SQL limit)

if not MONGO_URI:
    print("❌  MONGO_URI tidak di-set di .env atau environment variable.")
    print("    Isi MONGO_URI di file .env dan coba lagi.")
    sys.exit(1)

# ── MongoDB ───────────────────────────────────────────────────────────────────
try:
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import PyMongoError
    import certifi
except ImportError:
    print("❌  pymongo / certifi belum terinstall. Jalankan: pip install pymongo certifi")
    sys.exit(1)

_mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000, tlsCAFile=certifi.where())
_db           = _mongo_client["sugi_insights"]

# Koleksi per kategori
COL_PRICE      = _db["price_insights"]
COL_WEATHER    = _db["weather_insights"]
COL_PLANTING   = _db["planting_suggestions"]
COL_GENERAL    = _db["general_insights"]
COL_SESSION    = _db["session_summaries"]

# ── ChromaDB ──────────────────────────────────────────────────────────────────
try:
    import chromadb as _chromadb
except ImportError:
    print("❌  chromadb belum terinstall.")
    sys.exit(1)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

try:
    _chroma = _chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    _chroma.heartbeat()
    print(f"✅  ChromaDB terhubung di {CHROMA_HOST}:{CHROMA_PORT}")
except Exception as e:
    print(f"❌  Tidak bisa konek ke ChromaDB: {e}")
    sys.exit(1)

# ── Ollama LLM (Qwen2.5-1.5B untuk insight generation) ───────────────────────
try:
    from langchain_ollama.llms import OllamaLLM
    _llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0.4, repeat_penalty=1.1)
    print("✅  LLM (qwen2.5:1.5b) siap.")
except Exception as e:
    _llm = None
    print(f"⚠️   LLM tidak tersedia ({e}) — insight akan berupa ringkasan data mentah.")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers: ChromaDB batch fetch
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_all_docs(collection_name: str) -> list[dict]:
    """
    Fetch semua dokumen dari koleksi ChromaDB dengan batching.
    Return list of {id, text, metadata}.
    """
    try:
        col = _chroma.get_collection(collection_name)
    except Exception:
        return []

    id_result = col.get(include=[])
    all_ids   = id_result.get("ids", [])
    if not all_ids:
        return []

    docs = []
    for offset in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[offset: offset + BATCH_SIZE]
        batch     = col.get(ids=batch_ids, include=["documents", "metadatas"])
        for text, meta in zip(batch.get("documents", []), batch.get("metadatas", [])):
            docs.append({"text": text or "", "meta": meta or {}})
    return docs


def _ask_llm(prompt: str, fallback: str = "") -> str:
    """
    Tanya LLM. Kalau LLM tidak tersedia atau error, return fallback.
    """
    if _llm is None:
        return fallback
    try:
        result = _llm.invoke(prompt)
        return result.strip()
    except Exception as e:
        print(f"   ⚠️  LLM error: {e}")
        return fallback


def _doc_hash(text: str, category: str, date_str: str) -> str:
    """Buat ID unik dari isi + kategori + tanggal pengiriman."""
    return hashlib.md5(f"{category}|{date_str}|{text[:200]}".encode()).hexdigest()


def _upsert_many(collection, docs: list[dict]):
    """
    Upsert ke MongoDB berdasarkan field 'insight_id'.
    Tidak duplikasi kalau insight_id sama.
    """
    if not docs:
        return
    ops = [
        UpdateOne(
            {"insight_id": d["insight_id"]},
            {"$setOnInsert": d},
            upsert=True,
        )
        for d in docs
    ]
    try:
        result = collection.bulk_write(ops, ordered=False)
        inserted = result.upserted_count
        print(f"   ✅  {inserted} baru / {len(docs) - inserted} sudah ada")
    except PyMongoError as e:
        print(f"   ❌  MongoDB error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# Insight Generators
# ═════════════════════════════════════════════════════════════════════════════

NOW_UTC  = datetime.now(timezone.utc)
DATE_STR = NOW_UTC.strftime("%Y-%m-%d")


def _base_doc(category: str, text: str, **extra) -> dict:
    """Buat dokumen insight dengan field standar."""
    return {
        "insight_id":  _doc_hash(text, category, DATE_STR),
        "category":    category,
        "generated_at": NOW_UTC,
        "date":        DATE_STR,
        "raw_text":    text[:2000],    # simpan raw sebagian untuk audit
        **extra,
    }


# ─── 1. Harga Komoditas per Provinsi ─────────────────────────────────────────

def generate_price_insights() -> list[dict]:
    """
    Ambil dokumen harga dari ChromaDB, kelompokkan per provinsi/komoditas,
    minta LLM buat ringkasan insight per grup.
    """
    print("\n💰  Generating price insights...")
    docs = _fetch_all_docs("main_dataset")    # koleksi utama hasil vectorCSV

    # Filter dokumen bertipe harga
    price_keywords = {"harga", "price", "komoditas", "pasar", "rp", "rupiah", "kg", "ton"}
    price_docs = [
        d for d in docs
        if any(kw in d["text"].lower() for kw in price_keywords)
        and d["meta"].get("type") in ("price", "tabular", None)
    ]

    if not price_docs:
        print("   ℹ️  Tidak ada dokumen harga ditemukan.")
        return []

    # Kelompokkan per provinsi (dari metadata, fallback ke 'Nasional')
    groups: dict[str, list[str]] = {}
    for d in price_docs:
        prov = d["meta"].get("province") or d["meta"].get("region") or "Nasional"
        groups.setdefault(prov, []).append(d["text"])

    results = []

    def _process_group(province, texts):
        combined = "\n".join(texts[:10])    # ambil 10 chunk per grup
        prompt = (
            f"Kamu adalah analis pertanian Indonesia. "
            f"Berdasarkan data harga berikut untuk provinsi {province}, "
            f"buat insight singkat (3-5 kalimat) mencakup: "
            f"komoditas apa yang harganya tinggi/rendah, tren terkini, "
            f"dan rekomendasi untuk petani.\n\n"
            f"Data:\n{combined[:1500]}\n\n"
            f"Insight:"
        )
        insight = _ask_llm(prompt, fallback=f"Data harga tersedia untuk {province}: {combined[:300]}")
        return _base_doc(
            category  = "price_insight",
            text      = combined,
            province  = province,
            insight   = insight,
            doc_count = len(texts),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda item: _process_group(*item), groups.items()))

    print(f"   📊  {len(results)} price insights dari {len(price_docs)} dokumen.")
    return results


# ─── 2. Cuaca & Prakiraan ─────────────────────────────────────────────────────

def generate_weather_insights() -> list[dict]:
    """
    Ambil dokumen cuaca dari koleksi weather_data, buat insight per wilayah.
    """
    print("\n🌤️   Generating weather insights...")
    docs = _fetch_all_docs("weather_data")

    if not docs:
        print("   ℹ️  Tidak ada data cuaca di weather_data collection.")
        return []

    # Kelompokkan per lokasi
    groups: dict[str, list[str]] = {}
    for d in docs:
        loc = (
            d["meta"].get("location")
            or d["meta"].get("city")
            or d["meta"].get("region")
            or "Indonesia"
        )
        groups.setdefault(loc, []).append(d["text"])

    results = []

    def _process_weather_group(location, texts):
        combined = "\n".join(texts[:8])
        prompt = (
            f"Kamu adalah konsultan cuaca pertanian. "
            f"Berdasarkan data cuaca berikut untuk {location}, "
            f"buat insight singkat (3-4 kalimat) mencakup: "
            f"kondisi cuaca terkini, risiko cuaca ekstrem, "
            f"dan dampaknya terhadap kegiatan pertanian.\n\n"
            f"Data:\n{combined[:1200]}\n\n"
            f"Insight:"
        )
        insight = _ask_llm(prompt, fallback=f"Data cuaca tersedia untuk {location}: {combined[:300]}")
        return _base_doc(
            category  = "weather_insight",
            text      = combined,
            location  = location,
            insight   = insight,
            doc_count = len(texts),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda item: _process_weather_group(*item), groups.items()))

    print(f"   🌦️   {len(results)} weather insights dari {len(docs)} dokumen.")
    return results


# ─── 3. Saran Tanam ───────────────────────────────────────────────────────────

# Pemetaan bulan ke musim tanam Indonesia
_MUSIM_MAP = {
    1: "Musim Hujan", 2: "Musim Hujan", 3: "Peralihan",
    4: "Peralihan",   5: "Awal Kemarau", 6: "Musim Kemarau",
    7: "Musim Kemarau", 8: "Musim Kemarau", 9: "Akhir Kemarau",
    10: "Peralihan", 11: "Awal Hujan", 12: "Musim Hujan",
}

_KOMODITAS_UTAMA = [
    "padi", "jagung", "kedelai", "cabai", "bawang merah",
    "tomat", "singkong", "kelapa sawit", "kopi", "kakao",
]

def generate_planting_suggestions() -> list[dict]:
    """
    Buat saran tanam per komoditas berdasarkan data budidaya dari ChromaDB
    dan musim tanam saat ini.
    """
    print("\n🌱  Generating planting suggestions...")
    docs = _fetch_all_docs("main_dataset")

    budidaya_keywords = {
        "tanam", "semai", "bibit", "benih", "varietas",
        "pupuk", "irigasi", "panen", "musim tanam", "budidaya",
    }
    budidaya_docs = [
        d for d in docs
        if any(kw in d["text"].lower() for kw in budidaya_keywords)
    ]

    musim_sekarang = _MUSIM_MAP.get(NOW_UTC.month, "Tidak Diketahui")
    results        = []

    def _process_planting_group(komoditas):
        relevant = [
            d["text"] for d in budidaya_docs
            if komoditas.lower() in d["text"].lower()
        ]
        combined = "\n".join(relevant[:6]) if relevant else "(tidak ada data spesifik di database)"

        prompt = (
            f"Kamu adalah penyuluh pertanian Indonesia. "
            f"Buat saran tanam praktis untuk {komoditas} "
            f"di bulan {NOW_UTC.strftime('%B')} ({musim_sekarang}).\n"
            f"Saran harus mencakup: waktu tanam ideal, varietas yang direkomendasikan, "
            f"kebutuhan pupuk, dan peringatan hama/penyakit musiman. "
            f"Maksimal 5 poin singkat.\n\n"
            f"Data referensi dari database:\n{combined[:1000]}\n\n"
            f"Saran:"
        )
        insight = _ask_llm(
            prompt,
            fallback=f"Saran tanam {komoditas} untuk {musim_sekarang}: data terbatas di database."
        )
        return _base_doc(
            category       = "planting_suggestion",
            text           = combined,
            komoditas      = komoditas,
            musim          = musim_sekarang,
            bulan          = NOW_UTC.strftime("%B"),
            insight        = insight,
            data_available = len(relevant) > 0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(_process_planting_group, _KOMODITAS_UTAMA))

    print(f"   🌾  {len(results)} planting suggestions ({musim_sekarang}).")
    return results


# ─── 4. Insight Umum ──────────────────────────────────────────────────────────

def generate_general_insights() -> list[dict]:
    """
    Dari semua dokumen kebijakan, laporan, dan berita di ChromaDB,
    buat 3-5 insight tren/topik terpenting hari ini.
    """
    print("\n📰  Generating general insights...")
    docs = _fetch_all_docs("main_dataset")

    policy_keywords = {
        "kebijakan", "regulasi", "subsidi", "program", "bantuan",
        "ekspor", "impor", "harga acuan", "hpp", "ketahanan pangan",
        "produksi", "produktivitas", "inovasi", "teknologi pertanian",
    }
    policy_docs = [
        d for d in docs
        if any(kw in d["text"].lower() for kw in policy_keywords)
    ]

    if not policy_docs:
        print("   ℹ️  Tidak ada dokumen kebijakan/umum ditemukan.")
        return []

    combined = "\n\n".join(d["text"] for d in policy_docs[:15])[:3000]
    prompt = (
        "Kamu adalah analis kebijakan pertanian Indonesia. "
        "Berdasarkan dokumen-dokumen berikut, identifikasi "
        "3 sampai 5 insight terpenting terkait situasi pertanian Indonesia saat ini. "
        "Setiap insight harus memiliki judul singkat dan penjelasan 2-3 kalimat. "
        "Format: **[Judul]** — [Penjelasan]\n\n"
        f"Dokumen:\n{combined}\n\n"
        "Insight:"
    )
    insight_text = _ask_llm(
        prompt,
        fallback=f"Insight umum: {len(policy_docs)} dokumen kebijakan tersedia di database."
    )

    # Parse insight teks jadi beberapa dokumen jika berformat **Judul** — Isi
    import re
    parts = re.split(r"\n(?=\*\*)", insight_text)
    if len(parts) <= 1:
        parts = [insight_text]    # tidak bisa diparsing, simpan sebagai satu blok

    results = []
    for i, part in enumerate(parts, 1):
        title_match = re.match(r"\*\*(.+?)\*\*\s*[—-]\s*(.*)", part, re.DOTALL)
        title   = title_match.group(1).strip() if title_match else f"Insight {i}"
        content = title_match.group(2).strip() if title_match else part.strip()

        doc = _base_doc(
            category   = "general_insight",
            text       = combined[:500],
            title      = title,
            insight    = content,
            index      = i,
            doc_count  = len(policy_docs),
        )
        results.append(doc)

    print(f"   📋  {len(results)} general insights dari {len(policy_docs)} dokumen.")
    return results


# ─── 5. Session Summaries (dari ChromaDB memory store) ───────────────────────

def push_session_summaries() -> list[dict]:
    """
    Ambil ringkasan sesi dari conversation_memory ChromaDB collection
    dan kirim ke MongoDB sugi_insights.session_summaries.
    Hanya kirim yang belum ada (berdasarkan session_id + timestamp).
    """
    print("\n💬  Pushing session summaries...")
    docs = _fetch_all_docs("conversation_memory")

    if not docs:
        print("   ℹ️  Tidak ada session summary di conversation_memory.")
        return []

    results = []
    for d in docs:
        meta       = d["meta"]
        session_id = meta.get("session_id", "unknown")
        timestamp  = meta.get("timestamp", DATE_STR)
        content    = d["text"]

        doc = {
            "insight_id":  _doc_hash(content, "session_summary", session_id),
            "category":    "session_summary",
            "session_id":  session_id,
            "timestamp":   timestamp,
            "pushed_at":   NOW_UTC,
            "summary":     content,
            "char_count":  len(content),
        }
        results.append(doc)

    print(f"   💾  {len(results)} session summaries akan di-push.")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main: kirim semua insight ke MongoDB
# ═════════════════════════════════════════════════════════════════════════════

def run_once():
    """
    Generate semua insight dan kirim ke MongoDB.
    Dipanggil saat startup dan setiap INTERVAL_HOURS.
    """
    global NOW_UTC, DATE_STR
    NOW_UTC  = datetime.now(timezone.utc)
    DATE_STR = NOW_UTC.strftime("%Y-%m-%d")

    print("\n" + "═" * 60)
    print(f"🌾  SUGI Daily Insight — {NOW_UTC.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("═" * 60)

    try:
        # 1. Price insights
        price_docs = generate_price_insights()
        if price_docs:
            _upsert_many(COL_PRICE, price_docs)

        # 2. Weather insights
        weather_docs = generate_weather_insights()
        if weather_docs:
            _upsert_many(COL_WEATHER, weather_docs)

        # 3. Planting suggestions
        planting_docs = generate_planting_suggestions()
        if planting_docs:
            _upsert_many(COL_PLANTING, planting_docs)

        # 4. General insights
        general_docs = generate_general_insights()
        if general_docs:
            _upsert_many(COL_GENERAL, general_docs)

        # 5. Session summaries
        session_docs = push_session_summaries()
        if session_docs:
            _upsert_many(COL_SESSION, session_docs)

        total = sum(len(x) for x in [
            price_docs, weather_docs, planting_docs, general_docs, session_docs
        ])
        print(f"\n✅  Selesai — {total} dokumen diproses pada {NOW_UTC.strftime('%H:%M')} UTC")
        print("─" * 60)

    except Exception:
        print("❌  Error saat generate/kirim insight:")
        traceback.print_exc()


def run_loop(interval_hours: float = INTERVAL_HOURS):
    """
    Jalankan run_once() saat startup, lalu ulangi setiap interval_hours.
    """
    interval_sec = interval_hours * 3600
    print(f"⏰  Interval: setiap {interval_hours} jam "
          f"({interval_sec/3600:.1f}h = {interval_sec:.0f}s)")

    while True:
        run_once()
        next_run = datetime.now(timezone.utc)
        print(f"⏳  Insight berikutnya: "
              f"{(NOW_UTC.timestamp() + interval_sec - time.time()):.0f}s lagi "
              f"(~{interval_hours}h).\n"
              f"    Ctrl+C untuk berhenti.\n")
        time.sleep(interval_sec)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SUGI Daily Insight — kirim insight pertanian ke MongoDB"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Kirim insight sekali lalu exit (tanpa loop otomatis)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=INTERVAL_HOURS,
        metavar="JAM",
        help=f"Interval pengiriman dalam jam (default: {INTERVAL_HOURS})",
    )
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_loop(interval_hours=args.interval)