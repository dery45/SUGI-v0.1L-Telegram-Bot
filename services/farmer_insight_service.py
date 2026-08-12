"""
farmer_insight_service.py — Farmer Insights & Government Policy Recommendation Engine

Generates:
  - 10 farmer-focused market insights (~150 chars each) → saved to `farmerinsights`
  - 1 Government Policy Recommendation (400-500 chars) → saved to `governmentinsights`

SUGI Ecosystem:
  - MongoDB (test) for raw data aggregation
  - Hybrid RAG (ChromaDB main_dataset) for agricultural knowledge
  - Weather Engine (ChromaDB weather_data) for climate context
  - ChromaDB memory (insights_memory) for historical learning loop
  - Dual-save to `test` + `sugi_insights` databases
  - Perenual API for plant-related insights

Execution:
  - Startup: full generation
  - Daily: scheduled full refresh
  - On change: regenerate only affected insights
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "config" / ".env")

MONGO_URI = os.getenv("MONGO_URI", "").strip()
if not MONGO_URI:
    print("MONGO_URI not set. Check config/.env")
    sys.exit(1)

import certifi
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

MODEL_NAME = os.getenv("UTILITY_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

FARMER_INSIGHT_MAX_CHARS = 150
POLICY_MIN_CHARS = 400
POLICY_MAX_CHARS = 500

DAILY_INTERVAL = 86400
CHANGE_CHECK_INTERVAL = 3600

# M0: jeda antar LLM call + startup grace — jangan banjiri Ollama lokal saat
# full refresh (chatbot & embedding ikut pakai mesin yang sama).
INSIGHT_LLM_DELAY = float(os.getenv("INSIGHT_LLM_DELAY", "1"))
STARTUP_GRACE_SECONDS = int(os.getenv(
    "FARMER_INSIGHT_GRACE_SECONDS",
    os.getenv("STARTUP_GRACE_SECONDS", "300"),
))

# ─── Farmer Insight Definitions ───────────────────────────────────────────

FARMER_INSIGHT_DEFS = [
    {
        "key": "pph_score",
        "title": "Skor PPH Nasional",
        "collections": ["skorpphs"],
        "plant_related": False,
        "tags": ["nutrition", "diet", "national"],
    },
    {
        "key": "best_commodity",
        "title": "Komoditas Terbaik",
        "collections": ["hargaprodusenprovinsis"],
        "plant_related": True,
        "tags": ["price", "commodity", "producer"],
    },
    {
        "key": "margin_status",
        "title": "Margin Positif",
        "collections": ["hargaprodusenprovinsis", "hargakonsumenprovinsis"],
        "plant_related": True,
        "tags": ["margin", "price", "spread"],
    },
    {
        "key": "food_balance",
        "title": "Surplus Pangan",
        "collections": ["proyeksineracas"],
        "plant_related": True,
        "tags": ["supply", "demand", "balance"],
    },
    {
        "key": "regional_reserve",
        "title": "Cadangan Pangan Daerah",
        "collections": ["cadanganpanganprovinsis"],
        "plant_related": False,
        "tags": ["reserve", "region", "stock"],
    },
    {
        "key": "commodity_surplus",
        "title": "Surplus per Komoditas",
        "collections": ["proyeksineracas"],
        "plant_related": True,
        "tags": ["surplus", "commodity", "ranking"],
    },
    {
        "key": "monthly_opportunity",
        "title": "Peluang Bulanan",
        "collections": ["hargaprodusenprovinsis", "hargakonsumenprovinsis"],
        "plant_related": True,
        "tags": ["monthly", "opportunity", "margin"],
    },
    {
        "key": "best_province",
        "title": "Provinsi Terbaik",
        "collections": ["hargaprodusenprovinsis"],
        "plant_related": False,
        "tags": ["province", "price", "producer"],
    },
    {
        "key": "plant_advice",
        "title": "Rekomendasi Tanam",
        "collections": ["hargaprodusenprovinsis", "proyeksineracas", "skorpphs"],
        "plant_related": True,
        "tags": ["planting", "recommendation", "commodity"],
    },
    {
        "key": "sell_advice",
        "title": "Rekomendasi Jual",
        "collections": ["hargaprodusenprovinsis"],
        "plant_related": True,
        "tags": ["selling", "price", "market"],
    },
]

FARMER_COLLECTION_MAP: dict[str, set[str]] = {}
for _def in FARMER_INSIGHT_DEFS:
    for col in _def["collections"]:
        FARMER_COLLECTION_MAP.setdefault(col, set()).add(_def["key"])

POLICY_COLLECTIONS = [
    "proyeksineracas",
    "skorpphs",
    "ketidakcukupannasionals",
    "hargakonsumennasionals",
    "cadanganpanganprovinsis",
    "penyalurandonasis",
]
POLICY_INSIGHT_KEY = "policy_recommendation"

# ─── Prompts ─────────────────────────────────────────────────────────────

FARMER_PROMPT_TEMPLATE = """Kamu adalah asisten pertanian untuk petani Indonesia.

Buat insight SANGAT RINGKAS dengan gaya seperti contoh berikut:

  --- CONTOH GAYA YANG BENAR ---
  Skor PPH Nasional
  Skor PPH 2025 sebesar 97.55. Meningkat Infinity% — kualitas konsumsi pangan membaik.

  Komoditas Terbaik
  Sapi Hidup (Rp/kg Berat Hidup) adalah komoditas dengan harga produsen tertinggi (Rp 52.883/Kg).

  Margin Positif
  Rata-rata margin produsen-konsumen Rp 16.550/Kg — peluang pasar menguntungkan.

  Surplus Pangan
  Neraca pangan surplus 294.404.023 ton. Pasokan mencukupi.

  Surplus per Komoditas
  Beras memiliki surplus terbesar (194.733.724 ton).

  Peluang Bulanan
  Bulan Januari menunjukkan margin tertinggi untuk Jagung Pipilan Kering Tk. Petani (Rp/Kg) (Rp 0/Kg).

  Provinsi Terbaik
  Aceh menawarkan harga produsen tertinggi untuk Beras Medium (Rp 14.440/Kg).
  --- AKHIR CONTOH ---

CIRI KHAS GAYA:
- Judul sudah ditentukan: "{insight_title}"
- Kalimat sangat singkat, langsung ke inti
- Angka sebagai fokus utama — selalu sertakan data spesifik
- Tanda "—" untuk menghubungkan data dengan interpretasi
- Informasi tambahan dalam kurung ()
- Maksimal 2 kalimat pendek per insight
- MAKSIMAL {max_chars} KARAKTER — kurang dari itu lebih baik
- BAHASA INDONESIA SAJA, jangan campur bahasa Inggris
- Jangan gunakan label seperti [info] [positive] [warning]
- Langsung mulai kalimat pertama tanpa prefiks atau judul ulang

Data yang dianalisis: {data_summary}

{extra_context}

TULIS INSIGHT (gaya seperti contoh, maks {max_chars} karakter):"""

GOVERNMENT_PROMPT_TEMPLATE = """Kamu adalah analis kebijakan pangan Pemerintah Indonesia.

Buat SATU PARAGRAF rekomendasi kebijakan strategis ({min_chars}-{max_chars} karakter).

ATURAN KETAT:
- BAHASA INDONESIA FORMAL
- {min_chars}-{max_chars} KARAKTER
- SATU PARAGRAF — jangan pakai list, bullet, dash (-), atau angka (1. 2. 3.)
- Langsung mulai dengan kalimat pertama — tanpa judul, tanpa label, tanpa "Rekomendasi:"
- Jangan pernah memulai dengan meta-instruksi seperti "Berikut adalah..." atau "Untuk mengurangi..."
- Cukup tulis rekomendasinya langsung, seolah-olah itu本身就是 paragraf final
- Sebutkan 1-2 angka kunci nasional
- Akhiri dengan kalimat lengkap bertanda titik

CONTOH OUTPUT YANG BENAR:
"Neraca pangan nasional menunjukkan surplus 294 juta ton dengan skor PPH 97,5. Pemerintah perlu memperkuat cadangan pangan daerah dan memperbaiki distribusi donasi ke wilayah rawan pangan guna menjaga stabilitas harga dan ketahanan pangan nasional."

Sumber data: proyeksi neraca pangan, skor PPH, ketidakcukupan pangan, harga konsumen nasional, cadangan pangan daerah, dan penyaluran donasi.

{data_summary}

{extra_context}

TULIS SATU PARAGRAF REKOMENDASI ({min_chars}-{max_chars} karakter — tanpa list, tanpa bullet, tanpa meta-instruksi):"""

# ═════════════════════════════════════════════════════════════════════════════
# ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _ping_with_retry(mongo: MongoClient, attempts: int = 3, base_wait: float = 5.0) -> None:
    """A7: ping MongoDB dengan retry + backoff linear (5s, 10s).

    Atlas kadang transient "No primary found" saat pemilihan topologi baru
    (ServerSelectionTimeoutError). Dulu __init__ ping sekali tanpa retry dan
    hanya mengandalkan restart proses luar di start_all.py — tambahkan retry
    di dalam service sendiri (sama seperti government_insight_service).
    """
    for attempt in range(attempts):
        try:
            mongo.admin.command("ping")
            return
        except ServerSelectionTimeoutError as e:
            if attempt == attempts - 1:
                raise
            wait = base_wait * (attempt + 1)
            print(f"  FarmerInsight: Mongo ping failed (attempt {attempt+1}/{attempts}), "
                  f"retrying in {wait}s: {e}")
            time.sleep(wait)


class FarmerInsightEngine:

    def __init__(self):
        print("  FarmerInsight: Initializing...")

        # ── MongoDB ──
        self._mongo = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=15_000,
            socketTimeoutMS=30_000,
            connectTimeoutMS=20_000,
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=False,
            retryWrites=True,
        )
        _ping_with_retry(self._mongo)
        self._source_db = self._mongo["test"]
        self._target_sugi = self._mongo["sugi_insights"]
        self._target_test = self._mongo["test"]

        # Ensure indexes
        self._target_sugi["farmerinsights"].create_index("insightKey", unique=True, background=True)
        self._target_test["farmerinsights"].create_index("insightKey", unique=True, background=True)
        self._target_sugi["governmentinsights"].create_index("sourceCollection", unique=True, background=True)
        self._target_test["governmentinsights"].create_index("sourceCollection", unique=True, background=True)
        print("  FarmerInsight: MongoDB ready, indexes ensured.")

        # ── LLM ──
        self._llm = OllamaLLM(
            model=MODEL_NAME,
            temperature=0.3,
            repeat_penalty=1.1,
            num_ctx=4096,
            client_kwargs={"timeout": 240},  # A8: insight generate menjalankan banyak invoke — 90s terlalu ketat saat Ollama sedang sibuk
        )
        print(f"  FarmerInsight: LLM ready ({MODEL_NAME}).")

        # ── ChromaDB ──
        self._chroma_collection = None
        self._init_chromadb()

        # ── State ──
        self._known_state: dict[str, dict] = {}
        self._last_full_refresh: Optional[datetime] = None
        self._load_state()

    # ─── ChromaDB ────────────────────────────────────────────────────────

    def _init_chromadb(self):
        try:
            import chromadb as _cdb
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings

            _client = _cdb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _client.heartbeat()
            _embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            self._chroma_collection = Chroma(
                collection_name="insights_memory",
                client=_client,
                embedding_function=_embeddings,
            )
            print("  FarmerInsight: ChromaDB connected, collection 'insights_memory' ready.")
        except Exception as e:
            print(f"  FarmerInsight: ChromaDB not available — {e}")
            print("  FarmerInsight: ChromaDB memory disabled.")

    def _chromadb_ready(self) -> bool:
        return self._chroma_collection is not None

    def _save_to_chromadb(self, key: str, text: str, metadata: dict):
        if not self._chromadb_ready():
            return
        doc_id = hashlib.md5(f"fi:{key}:{metadata.get('version', 0)}".encode()).hexdigest()
        try:
            existing = self._chroma_collection.get(ids=[doc_id])
            if existing["ids"]:
                return
        except Exception:
            pass
        doc = Document(
            page_content=text,
            metadata={
                "source": "farmer_insight" if key != POLICY_INSIGHT_KEY else "policy_recommendation",
                "insightKey": key,
                "version": metadata.get("version", 0),
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                **metadata,
            },
            id=doc_id,
        )
        try:
            self._chroma_collection.add_documents(documents=[doc], ids=[doc_id])
        except Exception as e:
            print(f"    [chroma] store error: {e}")

    def _get_chromadb_memory(self, query: str, k: int = 3) -> str:
        if not self._chromadb_ready():
            return ""
        try:
            docs = self._chroma_collection.similarity_search(query, k=k)
            lines = []
            for d in docs:
                ik = d.metadata.get("insightKey", "?")
                lines.append(f"[{ik}]: {d.page_content[:200]}")
            if lines:
                return "INSIGHT SEBELUMNYA:\n" + "\n".join(lines)
        except Exception:
            pass
        return ""

    # ─── SUGI Ecosystem ─────────────────────────────────────────────────

    def _get_rag_context(self, query: str) -> str:
        try:
            from services.vectorCSV import vector_store as _vs
            docs = _vs.similarity_search(query, k=2)
            if docs:
                return "PENGETAHUAN PERTANIAN:\n" + "\n".join(d.page_content[:250] for d in docs)
        except Exception:
            pass
        return ""

    def _get_weather_context(self) -> str:
        try:
            from services.vectorWeather import weather_store as _ws
            docs = _ws.similarity_search("cuaca pertanian Indonesia", k=2)
            if docs:
                return "KONDISI CUACA:\n" + "\n".join(d.page_content[:300] for d in docs)
        except Exception:
            pass
        return ""

    def _get_plant_context(self, commodity: str) -> str:
        try:
            import chromadb as _cdb
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings
            _client = _cdb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _emb = OllamaEmbeddings(model=EMBED_MODEL)
            _pc = Chroma(collection_name="plant_data", client=_client, embedding_function=_emb)
            docs = _pc.similarity_search(commodity, k=1)
            if docs:
                return f"INFO TANAMAN:\n{docs[0].page_content[:300]}"
        except Exception:
            pass
        return ""

    # ═════════════════════════════════════════════════════════════════════
    # DATA AGGREGATION — Farmer Insights
    # ═════════════════════════════════════════════════════════════════════

    def _count(self, col: str) -> int:
        try:
            return self._source_db[col].count_documents({})
        except Exception:
            return 0

    def _sample(self, col: str, limit: int = 3) -> list:
        try:
            return list(self._source_db[col].find().sort("_id", -1).limit(limit))
        except Exception:
            return []

    def _agg_pph_score(self) -> str:
        col = self._source_db["skorpphs"]
        total = self._count("skorpphs")
        if total == 0:
            return "Data skor PPH belum tersedia."

        latest = list(col.find().sort("_id", -1).limit(5))
        parts = [f"Total data: {total}"]
        for doc in latest:
            clean = {k: v for k, v in doc.items() if k not in ("_id", "__v")}
            parts.append(str(clean))
        return "\n".join(parts)

    def _agg_best_commodity(self) -> str:
        col = self._source_db["hargaprodusenprovinsis"]
        pipeline = [
            {"$group": {"_id": "$komoditas", "avgHarga": {"$avg": "$harga"}, "count": {"$sum": 1}}},
            {"$sort": {"avgHarga": -1}},
            {"$limit": 5},
        ]
        try:
            results = list(col.aggregate(pipeline, allowDiskUse=True))
        except Exception:
            results = []
        if not results:
            return "Data harga produsen belum tersedia."
        return "\n".join(
            f"{r['_id']}: rata-rata Rp{r['avgHarga']:,.0f} ({r['count']} data)"
            for r in results
        )

    def _agg_margin_status(self) -> str:
        prod = self._source_db["hargaprodusenprovinsis"]
        kons = self._source_db["hargakonsumenprovinsis"]
        p_count = self._count("hargaprodusenprovinsis")
        k_count = self._count("hargakonsumenprovinsis")
        if p_count == 0 or k_count == 0:
            return f"Data tidak lengkap: produsen={p_count}, konsumen={k_count}."

        p_avg_pipeline = [{"$group": {"_id": "$komoditas", "avg": {"$avg": "$harga"}}}]
        k_avg_pipeline = [{"$group": {"_id": "$komoditas", "avg": {"$avg": "$harga"}}}]
        try:
            p_avgs = {r["_id"]: r["avg"] for r in prod.aggregate(p_avg_pipeline, allowDiskUse=True)}
            k_avgs = {r["_id"]: r["avg"] for r in kons.aggregate(k_avg_pipeline, allowDiskUse=True)}
        except Exception:
            return "Gagal aggregasi margin."

        common = set(p_avgs.keys()) & set(k_avgs.keys())
        margins = []
        for c in common:
            pv, kv = p_avgs[c], k_avgs[c]
            if pv is not None and kv is not None:
                margins.append((c, kv - pv, pv, kv))
        margins.sort(key=lambda x: x[1], reverse=True)

        lines = [f"Komoditas umum: {len(common)}"]
        for komo, margin, p, k in margins[:5]:
            lines.append(f"{komo}: margin Rp{margin:,.0f} (produsen Rp{p:,.0f} → konsumen Rp{k:,.0f})")
        return "\n".join(lines)

    def _agg_food_balance(self) -> str:
        col = self._source_db["proyeksineracas"]
        total = self._count("proyeksineracas")
        if total == 0:
            return "Data proyeksi neraca belum tersedia."

        latest = list(col.find().sort("_id", -1).limit(10))
        parts = [f"Total data: {total}"]
        for doc in latest:
            clean = {k: v for k, v in doc.items() if k not in ("_id", "__v")}
            parts.append(str(clean))
        return "\n".join(parts)

    def _agg_regional_reserve(self) -> str:
        col = self._source_db["cadanganpanganprovinsis"]
        total = self._count("cadanganpanganprovinsis")
        if total == 0:
            return "Data cadangan pangan belum tersedia."

        try:
            pf = self._province_field(col)
            pipeline = [
                {"$group": {"_id": f"${pf}", "totalTon": {"$sum": "$cppd_ton"}, "count": {"$sum": 1}}},
                {"$sort": {"totalTon": -1}},
                {"$limit": 5},
            ]
            results = list(col.aggregate(pipeline, allowDiskUse=True))
        except Exception:
            results = []

        parts = [f"Total data: {total}"]
        for r in results:
            parts.append(f"{r['_id']}: {r['totalTon']:,.0f} ton ({r['count']} data)")
        return "\n".join(parts) if parts else "Cadangan pangan per provinsi tidak tersedia."

    def _agg_commodity_surplus(self) -> str:
        col = self._source_db["proyeksineracas"]
        total = self._count("proyeksineracas")
        if total == 0:
            return "Data proyeksi neraca belum tersedia."

        try:
            pipeline = [
                {"$group": {
                    "_id": "$komoditas",
                    "avgNeraca": {"$avg": "$neraca"},
                    "count": {"$sum": 1},
                }},
                {"$sort": {"avgNeraca": -1}},
                {"$limit": 5},
            ]
            results = list(col.aggregate(pipeline, allowDiskUse=True))
        except Exception:
            results = []

        parts = []
        for r in results:
            status = "surplus" if r["avgNeraca"] > 0 else "defisit"
            parts.append(f"{r['_id']}: {r['avgNeraca']:,.0f} ({status}, {r['count']} data)")
        return "\n".join(parts) if parts else "Data surplus komoditas tidak tersedia."

    def _agg_monthly_opportunity(self) -> str:
        prod = self._source_db["hargaprodusenprovinsis"]
        kons = self._source_db["hargakonsumenprovinsis"]
        try:
            p_sample = list(prod.find().sort("_id", -1).limit(3))
            k_sample = list(kons.find().sort("_id", -1).limit(3))
        except Exception:
            p_sample, k_sample = [], []
        return (
            f"Data produsen: {self._count('hargaprodusenprovinsis')} docs\n"
            f"Data konsumen: {self._count('hargakonsumenprovinsis')} docs\n"
            f"Contoh produsen: {[str({k:v for k,v in d.items() if k not in ('_id','__v')}) for d in p_sample]}\n"
            f"Contoh konsumen: {[str({k:v for k,v in d.items() if k not in ('_id','__v')}) for d in k_sample]}"
        )

    def _agg_best_province(self) -> str:
        col = self._source_db["hargaprodusenprovinsis"]
        try:
            pf = self._province_field(col)
            pipeline = [
                {"$group": {"_id": f"${pf}", "avgHarga": {"$avg": "$harga"}, "count": {"$sum": 1}}},
                {"$sort": {"avgHarga": -1}},
                {"$limit": 5},
            ]
            results = list(col.aggregate(pipeline, allowDiskUse=True))
        except Exception:
            results = []
        if not results:
            return "Data provinsi belum tersedia."
        return "\n".join(
            f"{r['_id']}: Rp{r['avgHarga']:,.0f} ({r['count']} data)"
            for r in results
        )

    def _agg_plant_advice(self) -> str:
        return (
            f"Data harga produsen: {self._count('hargaprodusenprovinsis')} docs\n"
            f"Data proyeksi neraca: {self._count('proyeksineracas')} docs\n"
            f"Data skor PPH: {self._count('skorpphs')} docs\n"
            f"Sample harga: {self._sample('hargaprodusenprovinsis', 2)}\n"
            f"Sample neraca: {self._sample('proyeksineracas', 2)}"
        )

    def _agg_sell_advice(self) -> str:
        col = self._source_db["hargaprodusenprovinsis"]
        try:
            # Detect the province field name from a sample doc
            sample = col.find_one()
            province_field = None
            for candidate in ("provinsi", "province", "nama_provinsi", "wilayah", "region"):
                if sample and candidate in sample:
                    province_field = candidate
                    break
            if not province_field:
                province_field = "provinsi"

            pipeline = [
                {"$group": {"_id": {"komoditas": "$komoditas", "provinsi": f"${province_field}"},
                            "avgHarga": {"$avg": "$harga"}, "count": {"$sum": 1}}},
                {"$sort": {"avgHarga": -1}},
                {"$limit": 5},
            ]
            results = list(col.aggregate(pipeline, allowDiskUse=True))
        except Exception:
            results = []
        if not results:
            return "Data penjualan belum tersedia."
        return "\n".join(
            f"{r['_id']['komoditas']} di {r['_id']['provinsi']}: Rp{r['avgHarga']:,.0f} ({r['count']} data)"
            for r in results
        )

    @staticmethod
    def _province_field(col) -> str:
        try:
            sample = col.find_one()
            if sample:
                for c in ("provinsi", "province", "nama_provinsi", "wilayah", "region"):
                    if c in sample:
                        return c
        except Exception:
            pass
        return "provinsi"

    _AGG_MAP = {
        "pph_score": _agg_pph_score,
        "best_commodity": _agg_best_commodity,
        "margin_status": _agg_margin_status,
        "food_balance": _agg_food_balance,
        "regional_reserve": _agg_regional_reserve,
        "commodity_surplus": _agg_commodity_surplus,
        "monthly_opportunity": _agg_monthly_opportunity,
        "best_province": _agg_best_province,
        "plant_advice": _agg_plant_advice,
        "sell_advice": _agg_sell_advice,
    }

    # ═════════════════════════════════════════════════════════════════════
    # DATA AGGREGATION — Government Policy
    # ═════════════════════════════════════════════════════════════════════

    def _agg_policy_data(self) -> tuple[str, dict]:
        parts = []
        meta: dict[str, Any] = {"priorities": []}

        # proyeksineracas
        try:
            col = self._source_db["proyeksineracas"]
            pipeline = [{"$group": {"_id": None, "avgNeraca": {"$avg": "$neraca"},
                                     "totalKetersediaan": {"$sum": "$ketersediaan"},
                                     "totalKebutuhan": {"$sum": "$kebutuhan"}}}]
            r = list(col.aggregate(pipeline, allowDiskUse=True))
            if r:
                meta["neraca"] = round(r[0]["avgNeraca"], 0)
                parts.append(f"Neraca rata-rata: {r[0]['avgNeraca']:,.0f}, "
                             f"ketersediaan: {r[0]['totalKetersediaan']:,.0f}, "
                             f"kebutuhan: {r[0]['totalKebutuhan']:,.0f}")
        except Exception as e:
            parts.append(f"Proyeksi neraca: error ({e})")

        # skorpphs
        try:
            col = self._source_db["skorpphs"]
            latest = list(col.find().sort("_id", -1).limit(3))
            if latest:
                pph_vals = [d.get("skor_pph") or d.get("pph") or d.get("score") for d in latest if d]
                pph_vals = [v for v in pph_vals if v is not None]
                if pph_vals:
                    meta["pphScore"] = round(pph_vals[0], 1)
                    parts.append(f"Skor PPH terbaru: {pph_vals[0]}")
                parts.append(f"Data skor PPH: {len(latest)} record terbaru")
        except Exception as e:
            parts.append(f"Skor PPH: error ({e})")

        # ketidakcukupannasionals
        try:
            col = self._source_db["ketidakcukupannasionals"]
            latest = list(col.find().sort("_id", -1).limit(3))
            if latest:
                pou_vals = [d.get("pou") or d.get("prevalence") or d.get("persen") for d in latest if d]
                pou_vals = [v for v in pou_vals if v is not None]
                if pou_vals:
                    parts.append(f"Prevalence of undernourishment: {pou_vals[0]}%")
                    meta["pou"] = round(pou_vals[0], 1)
        except Exception as e:
            parts.append(f"Ketidakcukupan: error ({e})")

        # hargakonsumennasionals
        try:
            col = self._source_db["hargakonsumennasionals"]
            total = col.count_documents({})
            sample = list(col.find().sort("_id", -1).limit(3))
            if sample:
                parts.append(f"Harga konsumen nasional: {total} data, contoh: "
                             f"{[str({k:v for k,v in d.items() if k not in ('_id','__v')}) for d in sample]}")
        except Exception as e:
            parts.append(f"Harga konsumen: error ({e})")

        # cadanganpanganprovinsis
        try:
            col = self._source_db["cadanganpanganprovinsis"]
            pipeline = [{"$group": {"_id": None, "totalTon": {"$sum": "$cppd_ton"}}}]
            r = list(col.aggregate(pipeline, allowDiskUse=True))
            if r:
                meta["totalReserve"] = round(r[0]["totalTon"], 0)
                parts.append(f"Total cadangan pangan: {r[0]['totalTon']:,.0f} ton")
        except Exception as e:
            parts.append(f"Cadangan pangan: error ({e})")

        # penyalurandonasis
        try:
            col = self._source_db["penyalurandonasis"]
            pipeline = [{"$group": {"_id": None, "totalDonasi": {"$sum": "$jumlah_donasi_kg"},
                                     "totalPenerima": {"$sum": "$penerima_manfaat_jiwa"}}}]
            r = list(col.aggregate(pipeline, allowDiskUse=True))
            if r:
                meta["totalDonasi"] = round(r[0]["totalDonasi"], 0)
                meta["totalPenerima"] = round(r[0]["totalPenerima"], 0)
                parts.append(f"Total donasi: {r[0]['totalDonasi']:,.0f} kg, "
                             f"penerima: {r[0]['totalPenerima']:,.0f} jiwa")
        except Exception as e:
            parts.append(f"Penyaluran donasi: error ({e})")

        return "\n".join(parts), meta

    # ═════════════════════════════════════════════════════════════════════
    # GENERATION
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_type(text: str, key: str) -> str:
        """Determine insight type programmatically based on content heuristics."""
        t = text.lower()
        if any(w in t for w in ["defisit", "menurun", "turun", "risiko", "ancaman", "waspada", "kritis", "bahaya"]):
            return "danger"
        if any(w in t for w in ["meningkat", "naik", "baik", "positif", "membaik", "surplus", "optimal"]):
            return "positive"
        if any(w in t for w in ["hati-hati", "perhatikan", "waspadai", "fluktuasi", "tidak stabil"]):
            return "warning"
        return "info"

    @staticmethod
    def _has_truncation(text: str) -> bool:
        """Check if text ends with an incomplete sentence."""
        text = text.strip()
        if not text:
            return True
        if text[-1] in ".!?":
            return False
        if text[-3:] in ["...", "…”", ".\""]:
            return False
        if re.search(r'\b(dan|atau|serta|yang|di|ke|dari|dengan|untuk|pada|sebagai|akan|telah|sudah|belum)\s*$', text, re.IGNORECASE):
            return True
        return True

    @staticmethod
    def _detect_english(text: str) -> bool:
        """Heuristic check: if >30% of words are English, flag it."""
        t = text.strip()
        if not t:
            return False
        eng_indicators = {"the", "is", "are", "was", "were", "this", "that", "these", "those",
                          "and", "or", "for", "with", "from", "has", "have", "been", "being",
                          "will", "would", "could", "should", "may", "might", "shall", "can",
                          "which", "what", "when", "where", "how", "than", "there", "their",
                          "more", "most", "some", "any", "each", "every", "both", "all",
                          "best", "worst", "trend", "score", "price", "market", "food",
                          "increase", "decrease", "surplus", "deficit", "commodity", "margin",
                          "nutrition", "reserve", "province", "regional", "national", "total",
                          "current", "average", "based", "data", "production", "consumption",
                          "through", "while", "still", "remain", "over", "under", "between"}
        words = re.findall(r'\b[a-zA-Z]+\b', t)
        if not words:
            return False
        eng_count = sum(1 for w in words if w.lower() in eng_indicators)
        return (eng_count / len(words)) > 0.3

    def _clean_text(self, text: str, is_policy: bool = False) -> str:
        """Standard text cleanup for generated insights."""
        t = text.strip()
        t = t.replace("**", "").replace("__", "").replace("*", "")
        t = re.sub(r"\n\s*\n", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if is_policy:
            # Remove numbered lists
            t = re.sub(r"(?:^|\s)\d+[.>)\]]\s*", " ", t)
            # Remove dash/bullet lists
            t = re.sub(r"(?:^|\s)[-*•·]\s+", " ", t, flags=re.MULTILINE)
            # Remove meta-instruction prefixes
            for pat in [
                r'^Berikut\s+adalah\s+.*?[:.>-]+\s*',
                r'^Untuk\s+mengurangi\s+.*?[:.>-]+\s*',
                r'^Berikut\s+ini\s+.*?[:.>-]+\s*',
                r'^Berikut\s+rekomendasi\s+.*?[:.>-]+\s*',
                r'^Inilah\s+.*?[:.>-]+\s*',
            ]:
                t = re.sub(pat, '', t, flags=re.IGNORECASE)
            # Remove stray labels
            for pat in [
                r'^Rekomendasi\s*[:.>-]?\s*',
                r'^Kesimpulan\s*[:.>-]?\s*',
                r'^Policy\s*[:.>-]?\s*',
                r'^Government\s*[:.>-]?\s*',
                r'^Kebijakan\s*[:.>-]?\s*',
                r'^Peningkatan\s+\w+\s*[:.>-]?\s*',
                r'^Strategi\s*[:.>-]?\s*',
                r'^Prioritas\s*[:.>-]?\s*',
            ]:
                t = re.sub(pat, '', t, flags=re.IGNORECASE)
            # Collapse multiple colons+spaces into a single space
            t = re.sub(r'\s*[:;]\s+', ' ', t)
        t = self._strip_prefixes(t)
        t = t.strip()
        if t and t[-1] not in ".!?":
            t += "."
        return t

    @staticmethod
    def _strip_prefixes(text: str) -> str:
        """Remove any stray labels, brackets, or prefixes from generated text."""
        t = text.strip()
        t = re.sub(r'^\s*\[.*?\]\s*', '', t)
        for pat in [
            r'^(Best Commodity|Food Surplus|Regional Food Reserve|Best Province|'
            r'Planting Recommendation|Selling Recommendation|Monthly Opportunity|'
            r'Margin Status|PPH Score|Commodity Surplus|Market Opportunity|'
            r'National PPH Score|Producer.Consumer Margin|'
            r'Skor PPH Nasional|Komoditas Terbaik|Margin Positif|Surplus Pangan|'
            r'Cadangan Pangan Daerah|Surplus per Komoditas|Peluang Bulanan|'
            r'Provinsi Terbaik|Rekomendasi Tanam|Rekomendasi Jual|'
            r'Peningkatan\s+\w+|Ketahanan\s+Pangan\s+Daerah|'
            r'Info|Positive|Warning|Danger)\s*[:.>-]?\s*',
        ]:
            t = re.sub(pat, '', t, flags=re.IGNORECASE)
        t = t.strip()
        return t

    def _generate_farmer_insight(self, defn: dict) -> Optional[dict]:
        key = defn["key"]
        print(f"  [{key}] Generating...")

        agg_fn = self._AGG_MAP.get(key)
        if not agg_fn:
            data_summary = f"Koleksi: {', '.join(defn['collections'])}\nTotal data: {sum(self._count(c) for c in defn['collections'])}"
        else:
            data_summary = agg_fn(self)

        extra_parts = []
        rag = self._get_rag_context(f"{defn['title']} {' '.join(defn['tags'])}")
        if rag:
            extra_parts.append(rag)
        wth = self._get_weather_context()
        if wth:
            extra_parts.append(wth)
        mem = self._get_chromadb_memory(f"{defn['title']} {key}")
        if mem:
            extra_parts.append(mem)
        if defn["plant_related"]:
            plant = self._get_plant_context("padi jagung kedelai sayur")
            if plant:
                extra_parts.append(plant)
        extra = "\n\n".join(extra_parts) if extra_parts else "Tidak ada konteks tambahan."

        prompt = FARMER_PROMPT_TEMPLATE.format(
            insight_title=defn["title"],
            data_summary=data_summary,
            extra_context=extra,
            max_chars=str(FARMER_INSIGHT_MAX_CHARS),
        )

        raw = self._ask_llm(prompt)
        if not raw:
            print(f"  [{key}] LLM returned empty")
            return None

        clean_text = self._clean_text(raw)

        # Validate and retry if needed (up to 2 additional attempts)
        for attempt in range(3):
            issues = self._validate_insight(clean_text, key, max_chars=FARMER_INSIGHT_MAX_CHARS)
            if not issues:
                break
            if attempt < 2:
                print(f"  [{key}] Validation issues: {issues}, retrying ({attempt+2}/3)...")
                fix_prompt = (
                    f"Teks berikut memiliki masalah: {', '.join(issues)}.\n"
                    f"Tulis ulang dalam BAHASA INDONESIA SAJA, maksimal {FARMER_INSIGHT_MAX_CHARS} karakter, "
                    f"satu kalimat lengkap, tanpa label, tanpa kurung, tanpa prefiks.\n\n"
                    f"Teks asli: {clean_text}\n\nTeks perbaikan:"
                )
                raw = self._ask_llm(fix_prompt)
                if raw:
                    clean_text = self._clean_text(raw)
            else:
                # Final fallback: smart truncation at last complete sentence
                if len(clean_text) > FARMER_INSIGHT_MAX_CHARS:
                    truncated = clean_text[:FARMER_INSIGHT_MAX_CHARS]
                    last_period = truncated.rfind(".")
                    if last_period > 40:
                        clean_text = truncated[:last_period + 1]
                    else:
                        clean_text = truncated[:FARMER_INSIGHT_MAX_CHARS]

        msg_type = self._detect_type(clean_text, key)
        print(f"  [{key}] {len(clean_text)} chars, type={msg_type}")
        return {
            "insightKey": key,
            "type": msg_type,
            "title": defn["title"],
            "message": clean_text,
            "metadata": {},
        }

    def _generate_policy_recommendation(self) -> Optional[dict]:
        print("  [policy] Generating Government Policy Recommendation...")

        data_summary, meta = self._agg_policy_data()

        extra_parts = []
        rag = self._get_rag_context("kebijakan pangan nasional ketahanan pangan")
        if rag:
            extra_parts.append(rag)
        wth = self._get_weather_context()
        if wth:
            extra_parts.append(wth)
        mem = self._get_chromadb_memory("kebijakan pangan rekomendasi pemerintah", k=4)
        if mem:
            extra_parts.append(mem)
        extra = "\n\n".join(extra_parts) if extra_parts else "Tidak ada konteks tambahan."

        prompt = GOVERNMENT_PROMPT_TEMPLATE.format(
            data_summary=data_summary,
            extra_context=extra,
            min_chars=str(POLICY_MIN_CHARS),
            max_chars=str(POLICY_MAX_CHARS),
        )

        raw = self._ask_llm(prompt)
        if not raw:
            print("  [policy] LLM returned empty")
            return None

        cleaned = self._clean_text(raw, is_policy=True)

        # Validate and retry
        for attempt in range(3):
            issues = self._validate_insight(cleaned, POLICY_INSIGHT_KEY, max_chars=POLICY_MAX_CHARS)
            if not issues and POLICY_MIN_CHARS <= len(cleaned) <= POLICY_MAX_CHARS:
                break
            if not issues and len(cleaned) < POLICY_MIN_CHARS:
                issues.append(f"terlalu pendek ({len(cleaned)} < {POLICY_MIN_CHARS})")
            if attempt < 2:
                print(f"  [policy] Issues: {issues}, retrying ({attempt+2}/3)...")
                retry_min = POLICY_MIN_CHARS + 80
                retry_max = POLICY_MAX_CHARS + 80
                fix_prompt = (
                    f"Teks rekomendasi kebijakan berikut memiliki masalah: {', '.join(issues)}.\n"
                    f"Tulis ulang dalam BAHASA INDONESIA FORMAL, {retry_min}-{retry_max} karakter, "
                    f"satu paragraf padat, tanpa label, tanpa poin, tanpa bullet (-), tanpa sub-heading.\n"
                    f"JANGAN gunakan awalan seperti 'Berikut adalah' atau 'Untuk mengurangi'.\n"
                    f"Langsung tulis isi rekomendasinya.\n\n"
                    f"Teks asli: {cleaned}\n\nTeks perbaikan:"
                )
                raw = self._ask_llm(fix_prompt)
                if raw:
                    cleaned = self._clean_text(raw, is_policy=True)
            else:
                # Final fallback: smart truncation at last complete sentence
                if len(cleaned) > POLICY_MAX_CHARS:
                    truncated = cleaned[:POLICY_MAX_CHARS]
                    last_period = truncated.rfind(".")
                    if last_period > POLICY_MIN_CHARS - 50:
                        cleaned = truncated[:last_period + 1]
                    else:
                        cleaned = truncated[:POLICY_MAX_CHARS]
                if len(cleaned) < POLICY_MIN_CHARS:
                    suffix_variants = [
                        " Pemerintah perlu menyinergikan data lintas sektor untuk memperkuat ketahanan pangan nasional.",
                        " Sinergi lintas sektor diperlukan untuk memperkuat ketahanan pangan nasional.",
                        " Intervensi terpadu sangat diperlukan untuk menjaga stabilitas pangan nasional.",
                    ]
                    for suffix in suffix_variants:
                        if len(cleaned) + len(suffix) <= POLICY_MAX_CHARS:
                            cleaned += suffix
                            break

        # Build priority list from meta
        priorities = meta.get("priorities", [])
        if not priorities:
            if meta.get("neraca", 0) < 0:
                priorities.append("Tingkatkan produksi pangan pokok untuk menutup defisit neraca")
            if meta.get("pphScore", 100) < 80:
                priorities.append("Perbaiki skor PPH melalui diversifikasi konsumsi pangan lokal")
            if meta.get("pou", 0) > 10:
                priorities.append("Perkuat jaring pengaman sosial untuk daerah rawan pangan")
            priorities.append("Optimalkan penyaluran cadangan pangan dan donasi ke wilayah rentan")

        meta["priorities"] = priorities

        print(f"  [policy] {len(cleaned)} chars, {len(priorities)} priorities")
        return {
            "sourceCollection": POLICY_INSIGHT_KEY,
            "insight": cleaned,
            "metadata": meta,
        }

    def _validate_insight(self, text: str, key: str, max_chars: int) -> list[str]:
        """Validate insight quality. Returns list of issues (empty = valid)."""
        issues = []
        if not text or len(text.strip()) < 10:
            issues.append("teks terlalu pendek")
        # Check bracketed prefixes
        if re.search(r'^\s*\[.*?\]', text):
            issues.append("mengandung kurung siku di awal")
        if re.search(r'^\s*\w+[\s:>-]+', text) and not text[0].isalpha():
            issues.append("diawali label/prefiks")
        # Check length
        if len(text) > max_chars:
            issues.append(f"terlalu panjang ({len(text)} > {max_chars})")
        # Check truncation
        if self._has_truncation(text):
            issues.append("kalimat terpotong")
        # Check language
        if self._detect_english(text):
            issues.append("mengandung bahasa Inggris")
        return issues

    def _ask_llm(self, prompt: str) -> str:
        try:
            return self._llm.invoke(prompt).strip()
        except Exception as e:
            print(f"    [LLM error] {e}")
            return ""

    # ═════════════════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════════════════

    def _save_farmer_insight(self, result: dict):
        issues = self._validate_insight(result["message"], result["insightKey"], FARMER_INSIGHT_MAX_CHARS)
        if issues:
            print(f"  [WARN] {result['insightKey']} has issues: {issues} — saving anyway with best effort")
        now = datetime.now(timezone.utc)
        existing = self._target_sugi["farmerinsights"].find_one({"insightKey": result["insightKey"]})
        version = (existing.get("version", 0) if existing else 0) + 1

        doc = {
            "insightKey": result["insightKey"],
            "type": result["type"],
            "title": result["title"],
            "message": result["message"],
            "version": version,
            "generatedAt": now,
            "metadata": result.get("metadata", {}),
        }

        for db, label in [(self._target_sugi, "sugi_insights"), (self._target_test, "test")]:
            try:
                db["farmerinsights"].replace_one({"insightKey": result["insightKey"]}, doc, upsert=True)
                print(f"    [saved] {label}.farmerinsights ({result['insightKey']}) v{version}")
            except PyMongoError as e:
                print(f"    [error] {label}.farmerinsights save: {e}")

        self._save_to_chromadb(result["insightKey"], result["message"], {
            "version": version,
            "type": result["type"],
            "title": result["title"],
            "insightKey": result["insightKey"],
            "source": "farmer_insight",
            **result.get("metadata", {}),
        })

    def _save_policy(self, result: dict):
        issues = self._validate_insight(result["insight"], POLICY_INSIGHT_KEY, POLICY_MAX_CHARS)
        if issues:
            print(f"  [WARN] policy has issues: {issues} — saving anyway with best effort")
        now = datetime.now(timezone.utc)
        existing = self._target_sugi["governmentinsights"].find_one(
            {"sourceCollection": POLICY_INSIGHT_KEY}
        )
        version = (existing.get("version", 0) if existing else 0) + 1

        doc = {
            "sourceCollection": POLICY_INSIGHT_KEY,
            "insight": result["insight"],
            "version": version,
            "generatedAt": now,
            "metadata": result.get("metadata", {}),
        }

        for db, label in [(self._target_sugi, "sugi_insights"), (self._target_test, "test")]:
            try:
                db["governmentinsights"].replace_one(
                    {"sourceCollection": POLICY_INSIGHT_KEY}, doc, upsert=True
                )
                print(f"    [saved] {label}.governmentinsights (policy) v{version}")
            except PyMongoError as e:
                print(f"    [error] {label}.governmentinsights save: {e}")

        self._save_to_chromadb(POLICY_INSIGHT_KEY, result["insight"], {
            "version": version,
            "source": "policy_recommendation",
            "sourceCollection": POLICY_INSIGHT_KEY,
            **result.get("metadata", {}),
        })

    # ═════════════════════════════════════════════════════════════════════
    # CHANGE DETECTION
    # ═════════════════════════════════════════════════════════════════════

    def _compute_signature(self, col_name: str) -> Optional[dict]:
        try:
            col = self._source_db[col_name]
            count = col.count_documents({})
            if count == 0:
                return {"count": 0, "signature": "empty"}
            pipeline = [{"$sort": {"_id": -1}}, {"$limit": 50}, {"$project": {"_id": 1}}]
            ids = [str(doc["_id"]) for doc in col.aggregate(pipeline, allowDiskUse=True)]
            signature = hashlib.md5("".join(ids).encode()).hexdigest()
            return {"count": count, "signature": signature}
        except Exception as e:
            print(f"    [sig] error {col_name}: {e}")
            return None

    def _detect_changed_collections(self) -> set[str]:
        changed = set()
        all_cols = set(FARMER_COLLECTION_MAP.keys()) | set(POLICY_COLLECTIONS)
        for col in all_cols:
            sig = self._compute_signature(col)
            if sig is None:
                continue
            old = self._known_state.get(col)
            if old is None or old.get("count") != sig["count"] or old.get("signature") != sig["signature"]:
                changed.add(col)
                self._known_state[col] = sig
        if changed:
            self._save_state()
        return changed

    def _get_affected_insights(self, changed_cols: set[str]) -> set[str]:
        affected: set[str] = set()
        for col in changed_cols:
            if col in FARMER_COLLECTION_MAP:
                affected |= FARMER_COLLECTION_MAP[col]
            if col in POLICY_COLLECTIONS:
                affected.add(POLICY_INSIGHT_KEY)
        return affected

    # ═════════════════════════════════════════════════════════════════════
    # RUNNER
    # ═════════════════════════════════════════════════════════════════════

    def run_all(self):
        print("\n  ── Generating all farmer insights (10) ──")
        for i, defn in enumerate(FARMER_INSIGHT_DEFS):
            try:
                result = self._generate_farmer_insight(defn)
                if result:
                    self._save_farmer_insight(result)
            except Exception as e:
                print(f"  [ERROR] {defn['key']}: {e}")
                traceback.print_exc()
            finally:
                # M0b: jeda antar LLM call agar tidak membanjiri Ollama lokal.
                if i < len(FARMER_INSIGHT_DEFS) - 1 and INSIGHT_LLM_DELAY > 0:
                    time.sleep(INSIGHT_LLM_DELAY)

        print("\n  ── Generating Government Policy Recommendation ──")
        try:
            result = self._generate_policy_recommendation()
            if result:
                self._save_policy(result)
        except Exception as e:
            print(f"  [ERROR] policy: {e}")
            traceback.print_exc()

        self._last_full_refresh = datetime.now(timezone.utc)
        self._save_state()

    def run_affected(self, changed_cols: set[str]):
        affected = self._get_affected_insights(changed_cols)
        if not affected:
            print("  No affected insights.")
            return

        print(f"  Regenerating {len(affected)} affected insights...")
        for defn in FARMER_INSIGHT_DEFS:
            if defn["key"] in affected:
                try:
                    result = self._generate_farmer_insight(defn)
                    if result:
                        self._save_farmer_insight(result)
                except Exception as e:
                    print(f"  [ERROR] {defn['key']}: {e}")

        if POLICY_INSIGHT_KEY in affected:
            try:
                result = self._generate_policy_recommendation()
                if result:
                    self._save_policy(result)
            except Exception as e:
                print(f"  [ERROR] policy: {e}")

    def run_change_check(self):
        changed = self._detect_changed_collections()
        if changed:
            print(f"  Detected changes in: {', '.join(sorted(changed))}")
            self.run_affected(changed)
        else:
            print("  No data changes detected.")

    def run_loop(self):
        print(f"\n  [loop] Polling every {CHANGE_CHECK_INTERVAL}s, "
              f"full refresh every {DAILY_INTERVAL}s. Ctrl+C to stop.\n")

        if STARTUP_GRACE_SECONDS > 0:
            print(f"  ⏳  Startup grace {STARTUP_GRACE_SECONDS}s — biarkan chatbot "
                  f"dan watchers warm-up dulu (hindari kontes Ollama)...")
            time.sleep(STARTUP_GRACE_SECONDS)

        self.run_all()  # full generation on startup

        while True:
            try:
                time.sleep(CHANGE_CHECK_INTERVAL)

                # Check if daily refresh is needed
                now = datetime.now(timezone.utc)
                need_full = False
                if self._last_full_refresh:
                    elapsed = (now - self._last_full_refresh).total_seconds()
                    if elapsed >= DAILY_INTERVAL:
                        need_full = True
                else:
                    need_full = True

                if need_full:
                    print(f"\n  [daily] Full refresh due (last: {self._last_full_refresh})")
                    self.run_all()
                else:
                    self.run_change_check()

            except KeyboardInterrupt:
                print("\n  FarmerInsight: Stopped by user.")
                break

    # ═════════════════════════════════════════════════════════════════════
    # STATE
    # ═════════════════════════════════════════════════════════════════════

    def _state_path(self) -> Path:
        p = _ROOT / "data" / "db" / "farmer_insight_state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_state(self):
        p = self._state_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self._known_state = data.get("signatures", {})
                lrf = data.get("last_full_refresh")
                if lrf:
                    self._last_full_refresh = datetime.fromisoformat(lrf)
            except Exception:
                self._known_state = {}

    def _save_state(self):
        try:
            data = {
                "signatures": self._known_state,
                "last_full_refresh": self._last_full_refresh.isoformat() if self._last_full_refresh else None,
            }
            self._state_path().write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            print(f"    [state] save error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Farmer Insights & Government Policy Engine")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--check", action="store_true", help="Check for data changes and regenerate affected")
    parser.add_argument("--force", action="store_true", help="Force regenerate all")
    args = parser.parse_args()

    engine = FarmerInsightEngine()

    if args.check:
        engine.run_change_check()
    elif args.once or args.force:
        engine.run_all()
    else:
        engine.run_loop()


if __name__ == "__main__":
    main()
