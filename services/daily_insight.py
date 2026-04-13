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
  CHROMA_HOST — Host ChromaDB (default: localhost)
  CHROMA_PORT — Port ChromaDB (default: 8000)

Cara pakai:
  python daily_insight.py               # jalankan & biarkan loop
  python daily_insight.py --once        # kirim sekali lalu exit
  python daily_insight.py --interval 6  # loop setiap 6 jam
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import ssl
import sys
import time
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
    pass  # dotenv opsional

MONGO_URI      = os.getenv("MONGO_URI", "").strip()
CHROMA_HOST    = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT    = int(os.getenv("CHROMA_PORT", "8000"))
INTERVAL_HOURS = float(os.getenv("INSIGHT_INTERVAL_HOURS", "12"))
BATCH_SIZE     = 500
MAX_WORKERS    = 5

if not MONGO_URI:
    print("❌  MONGO_URI tidak di-set di .env atau environment variable.")
    print("    Isi MONGO_URI di file config/.env dan coba lagi.")
    sys.exit(1)

# ── Debug: print OpenSSL version ──────────────────────────────────────────────
print(f"🔒  OpenSSL version: {ssl.OPENSSL_VERSION}")

# ── MongoDB ───────────────────────────────────────────────────────────────────
try:
    import certifi
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import PyMongoError
except ImportError:
    print("❌  pymongo / certifi belum terinstall.")
    print("    Jalankan: pip install pymongo certifi")
    sys.exit(1)


def _build_mongo_client() -> MongoClient:
    """
    Buat MongoClient dengan konfigurasi TLS yang benar untuk MongoDB Atlas.

    - tlsCAFile=certifi.where()      → pakai CA bundle dari certifi (up-to-date)
    - tlsAllowInvalidCertificates=False → validasi sertifikat dengan benar (JANGAN True)
    - tls=True                       → aktifkan TLS secara eksplisit
    - retryWrites=True               → retry otomatis pada transient error
    """
    return MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=15_000,
        socketTimeoutMS=20_000,
        connectTimeoutMS=20_000,
        tls=True,
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=False,   # ← WAJIB False agar handshake benar
        retryWrites=True,
        retryReads=True,
    )


try:
    _mongo_client = _build_mongo_client()
    # Test koneksi saat startup
    _mongo_client.admin.command("ping")
    print("✅  MongoDB Atlas terhubung.")
except Exception as e:
    print(f"❌  Gagal koneksi ke MongoDB Atlas: {e}")
    print("    Pastikan:")
    print("    1. MONGO_URI benar di .env")
    print("    2. IP kamu sudah di-whitelist di Atlas Network Access")
    print("    3. pymongo & certifi sudah di-upgrade: pip install -U pymongo certifi")
    sys.exit(1)

_db          = _mongo_client["sugi_insights"]
COL_PRICE    = _db["price_insights"]
COL_WEATHER  = _db["weather_insights"]
COL_PLANTING = _db["planting_suggestions"]
COL_GENERAL  = _db["general_insights"]
COL_SESSION  = _db["session_summaries"]

# Buat index unik agar upsert cepat
for _col in [COL_PRICE, COL_WEATHER, COL_PLANTING, COL_GENERAL, COL_SESSION]:
    try:
        _col.create_index("insight_id", unique=True, background=True)
    except PyMongoError:
        pass  # index mungkin sudah ada

# ── ChromaDB ──────────────────────────────────────────────────────────────────
try:
    import chromadb
except ImportError:
    print("❌  chromadb belum terinstall. Jalankan: pip install chromadb")
    sys.exit(1)

try:
    _chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    _chroma.heartbeat()
    print(f"✅  ChromaDB terhubung di {CHROMA_HOST}:{CHROMA_PORT}")
except Exception as e:
    print(f"❌  Tidak bisa konek ke ChromaDB: {e}")
    sys.exit(1)

# ── Ollama LLM ────────────────────────────────────────────────────────────────
try:
    from langchain_ollama.llms import OllamaLLM
    _llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0.4, repeat_penalty=1.1)
    print("✅  LLM (qwen2.5:1.5b) siap.")
except Exception as e:
    _llm = None
    print(f"⚠️   LLM tidak tersedia ({e}) — insight akan berupa ringkasan data mentah.")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_all_docs(collection_name: str) -> list[dict]:
    """
    Fetch semua dokumen dari koleksi ChromaDB dengan batching.
    Return list of {text, meta}.
    """
    try:
        col = _chroma.get_collection(collection_name)
    except Exception as e:
        print(f"   ⚠️  Koleksi '{collection_name}' tidak ditemukan: {e}")
        return []

    try:
        all_ids = col.get(include=[]).get("ids", [])
    except Exception as e:
        print(f"   ⚠️  Gagal fetch IDs dari '{collection_name}': {e}")
        return []

    if not all_ids:
        return []

    docs: list[dict] = []
    for offset in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[offset: offset + BATCH_SIZE]
        try:
            batch = col.get(ids=batch_ids, include=["documents", "metadatas"])
            for text, meta in zip(
                batch.get("documents", []),
                batch.get("metadatas", []),
            ):
                docs.append({"text": text or "", "meta": meta or {}})
        except Exception as e:
            print(f"   ⚠️  Batch {offset}–{offset + BATCH_SIZE} gagal: {e}")
            continue

    return docs


def _ask_llm(prompt: str, fallback: str = "") -> str:
    """Invoke LLM. Kembalikan fallback jika LLM tidak tersedia atau error."""
    if _llm is None:
        return fallback
    try:
        return _llm.invoke(prompt).strip()
    except Exception as e:
        print(f"   ⚠️  LLM error: {e}")
        return fallback


def _doc_hash(text: str, category: str, scope: str) -> str:
    """Buat ID unik deterministik dari konten + kategori + scope (tanggal/session)."""
    raw = f"{category}|{scope}|{text[:200]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _upsert_many(collection, docs: list[dict]) -> None:
    """
    Upsert batch ke MongoDB.
    Pakai $setOnInsert agar dokumen yang sudah ada tidak di-overwrite.
    Tambah retry sederhana (3x) untuk transient network error.
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

    for attempt in range(1, 4):
        try:
            result = collection.bulk_write(ops, ordered=False)
            inserted = result.upserted_count
            skipped  = len(docs) - inserted
            print(f"   ✅  {inserted} baru ditambahkan / {skipped} sudah ada")
            return
        except PyMongoError as e:
            print(f"   ⚠️  MongoDB upsert attempt {attempt}/3 gagal: {e}")
            if attempt < 3:
                time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s
            else:
                print(f"   ❌  Upsert gagal setelah 3 percobaan: {e}")


def _base_doc(category: str, text: str, date_str: str, **extra) -> dict:
    """Buat skeleton dokumen insight dengan field standar."""
    now = datetime.now(timezone.utc)
    return {
        "insight_id":   _doc_hash(text, category, date_str),
        "category":     category,
        "generated_at": now,
        "date":         date_str,
        "raw_text":     text[:2000],
        **extra,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Musim tanam
# ═════════════════════════════════════════════════════════════════════════════

_MUSIM_MAP: dict[int, str] = {
    1: "Musim Hujan",   2: "Musim Hujan",   3: "Peralihan",
    4: "Peralihan",     5: "Awal Kemarau",  6: "Musim Kemarau",
    7: "Musim Kemarau", 8: "Musim Kemarau", 9: "Akhir Kemarau",
    10: "Peralihan",    11: "Awal Hujan",   12: "Musim Hujan",
}

_KOMODITAS_UTAMA: list[str] = [
    "padi", "jagung", "kedelai", "cabai", "bawang merah",
    "tomat", "singkong", "kelapa sawit", "kopi", "kakao",
]


# ═════════════════════════════════════════════════════════════════════════════
# Insight Generators
# ═════════════════════════════════════════════════════════════════════════════

def generate_price_insights(date_str: str) -> list[dict]:
    """
    Ambil dokumen harga dari ChromaDB, kelompokkan per provinsi,
    minta LLM buat ringkasan insight per grup.
    """
    print("\n💰  Generating price insights...")
    docs = _fetch_all_docs("main_dataset")

    price_keywords = {"harga", "price", "komoditas", "pasar", "rp", "rupiah", "kg", "ton"}
    price_docs = [
        d for d in docs
        if any(kw in d["text"].lower() for kw in price_keywords)
        and d["meta"].get("type") in ("price", "tabular", None)
    ]

    if not price_docs:
        print("   ℹ️  Tidak ada dokumen harga ditemukan.")
        return []

    groups: dict[str, list[str]] = {}
    for d in price_docs:
        prov = (
            d["meta"].get("province")
            or d["meta"].get("region")
            or "Nasional"
        )
        groups.setdefault(prov, []).append(d["text"])

    def _process(province: str, texts: list[str]) -> dict:
        combined = "\n".join(texts[:10])
        prompt = (
            f"Kamu adalah analis pertanian Indonesia. "
            f"Berdasarkan data harga berikut untuk provinsi {province}, "
            f"buat insight singkat (3-5 kalimat) mencakup: "
            f"komoditas apa yang harganya tinggi/rendah, tren terkini, "
            f"dan rekomendasi untuk petani.\n\n"
            f"Data:\n{combined[:1500]}\n\nInsight:"
        )
        insight = _ask_llm(
            prompt,
            fallback=f"Data harga tersedia untuk {province}: {combined[:300]}",
        )
        return _base_doc(
            category  = "price_insight",
            text      = combined,
            date_str  = date_str,
            province  = province,
            insight   = insight,
            doc_count = len(texts),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(lambda kv: _process(*kv), groups.items()))

    print(f"   📊  {len(results)} price insights dari {len(price_docs)} dokumen.")
    return results


def generate_weather_insights(date_str: str) -> list[dict]:
    """Ambil dokumen cuaca, buat insight per lokasi."""
    print("\n🌤️   Generating weather insights...")
    docs = _fetch_all_docs("weather_data")

    if not docs:
        print("   ℹ️  Tidak ada data cuaca di weather_data collection.")
        return []

    groups: dict[str, list[str]] = {}
    for d in docs:
        loc = (
            d["meta"].get("location")
            or d["meta"].get("city")
            or d["meta"].get("region")
            or "Indonesia"
        )
        groups.setdefault(loc, []).append(d["text"])

    def _process(location: str, texts: list[str]) -> dict:
        combined = "\n".join(texts[:8])
        prompt = (
            f"Kamu adalah konsultan cuaca pertanian. "
            f"Berdasarkan data cuaca berikut untuk {location}, "
            f"buat insight singkat (3-4 kalimat) mencakup: "
            f"kondisi cuaca terkini, risiko cuaca ekstrem, "
            f"dan dampaknya terhadap kegiatan pertanian.\n\n"
            f"Data:\n{combined[:1200]}\n\nInsight:"
        )
        insight = _ask_llm(
            prompt,
            fallback=f"Data cuaca tersedia untuk {location}: {combined[:300]}",
        )
        return _base_doc(
            category  = "weather_insight",
            text      = combined,
            date_str  = date_str,
            location  = location,
            insight   = insight,
            doc_count = len(texts),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(lambda kv: _process(*kv), groups.items()))

    print(f"   🌦️   {len(results)} weather insights dari {len(docs)} dokumen.")
    return results


def generate_planting_suggestions(date_str: str) -> list[dict]:
    """Buat saran tanam per komoditas berdasarkan musim tanam saat ini."""
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

    now          = datetime.now(timezone.utc)
    musim        = _MUSIM_MAP.get(now.month, "Tidak Diketahui")
    bulan_nama   = now.strftime("%B")

    def _process(komoditas: str) -> dict:
        relevant = [
            d["text"] for d in budidaya_docs
            if komoditas.lower() in d["text"].lower()
        ]
        combined = "\n".join(relevant[:6]) if relevant else "(tidak ada data spesifik)"
        prompt = (
            f"Kamu adalah penyuluh pertanian Indonesia. "
            f"Buat saran tanam praktis untuk {komoditas} "
            f"di bulan {bulan_nama} ({musim}).\n"
            f"Saran harus mencakup: waktu tanam ideal, varietas yang direkomendasikan, "
            f"kebutuhan pupuk, dan peringatan hama/penyakit musiman. "
            f"Maksimal 5 poin singkat.\n\n"
            f"Data referensi:\n{combined[:1000]}\n\nSaran:"
        )
        insight = _ask_llm(
            prompt,
            fallback=f"Saran tanam {komoditas} untuk {musim}: data terbatas di database.",
        )
        return _base_doc(
            category       = "planting_suggestion",
            text           = combined,
            date_str       = date_str,
            komoditas      = komoditas,
            musim          = musim,
            bulan          = bulan_nama,
            insight        = insight,
            data_available = len(relevant) > 0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(_process, _KOMODITAS_UTAMA))

    print(f"   🌾  {len(results)} planting suggestions ({musim}).")
    return results


def generate_general_insights(date_str: str) -> list[dict]:
    """Buat 3-5 insight tren/topik terpenting dari dokumen kebijakan."""
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
        f"Dokumen:\n{combined}\n\nInsight:"
    )
    insight_text = _ask_llm(
        prompt,
        fallback=f"Insight umum: {len(policy_docs)} dokumen kebijakan tersedia di database.",
    )

    parts = re.split(r"\n(?=\*\*)", insight_text)
    if len(parts) <= 1:
        parts = [insight_text]

    results = []
    for i, part in enumerate(parts, 1):
        m       = re.match(r"\*\*(.+?)\*\*\s*[—-]\s*(.*)", part, re.DOTALL)
        title   = m.group(1).strip() if m else f"Insight {i}"
        content = m.group(2).strip() if m else part.strip()
        if not content:
            continue
        results.append(_base_doc(
            category  = "general_insight",
            text      = combined[:500],
            date_str  = date_str,
            title     = title,
            insight   = content,
            index     = i,
            doc_count = len(policy_docs),
        ))

    print(f"   📋  {len(results)} general insights dari {len(policy_docs)} dokumen.")
    return results


def push_session_summaries(date_str: str) -> list[dict]:
    """
    Ambil ringkasan sesi dari conversation_memory ChromaDB
    dan kirim ke MongoDB session_summaries.
    """
    print("\n💬  Pushing session summaries...")
    docs = _fetch_all_docs("conversation_memory")

    if not docs:
        print("   ℹ️  Tidak ada session summary di conversation_memory.")
        return []

    results = []
    now = datetime.now(timezone.utc)
    for d in docs:
        meta       = d["meta"]
        session_id = meta.get("session_id", "unknown")
        timestamp  = meta.get("timestamp", date_str)
        content    = d["text"]

        results.append({
            "insight_id": _doc_hash(content, "session_summary", session_id),
            "category":   "session_summary",
            "session_id": session_id,
            "timestamp":  timestamp,
            "pushed_at":  now,
            "summary":    content,
            "char_count": len(content),
        })

    print(f"   💾  {len(results)} session summaries akan di-push.")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════════════

def run_once() -> None:
    """Generate semua insight dan kirim ke MongoDB."""
    now      = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    print("\n" + "═" * 60)
    print(f"🌾  SUGI Daily Insight — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("═" * 60)

    try:
        tasks: list[tuple[Any, list[dict]]] = [
            (COL_PRICE,    generate_price_insights(date_str)),
            (COL_WEATHER,  generate_weather_insights(date_str)),
            (COL_PLANTING, generate_planting_suggestions(date_str)),
            (COL_GENERAL,  generate_general_insights(date_str)),
            (COL_SESSION,  push_session_summaries(date_str)),
        ]

        total = 0
        for col, docs in tasks:
            if docs:
                _upsert_many(col, docs)
                total += len(docs)

        print(f"\n✅  Selesai — {total} dokumen diproses pada {now.strftime('%H:%M')} UTC")
        print("─" * 60)

    except Exception:
        print("❌  Error tidak terduga saat generate/kirim insight:")
        traceback.print_exc()


def run_loop(interval_hours: float = INTERVAL_HOURS) -> None:
    """Jalankan run_once() saat startup, lalu ulangi setiap interval_hours."""
    interval_sec = interval_hours * 3600
    print(f"⏰  Loop aktif — interval setiap {interval_hours:.1f} jam. "
          f"Ctrl+C untuk berhenti.\n")

    while True:
        run_once()
        next_run = datetime.now(timezone.utc).timestamp() + interval_sec
        remaining = next_run - time.time()
        print(f"⏳  Insight berikutnya dalam ~{remaining / 3600:.1f} jam "
              f"({interval_hours:.0f}h).\n")
        try:
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n👋  Loop dihentikan. Sampai jumpa!")
            sys.exit(0)


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