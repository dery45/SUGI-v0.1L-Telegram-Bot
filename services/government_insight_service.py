"""
government_insight_service.py — Government Insight Engine for SUGI v0.1L

Generates AI-powered government insights from 14 MongoDB collections using
the full SUGI AI ecosystem (Hybrid RAG, Weather Engine, LLM reasoning, ChromaDB memory).

Storage:
  - MongoDB:  sugi_insights.governmentinsights  (primary, structured)
  - ChromaDB: government_memory                  (semantic, for retrieval-augmented generation)

Learning Loop:
  1. Generate insight → store in MongoDB + ChromaDB
  2. Next generation → retrieve past insights from ChromaDB via semantic search
  3. Include past insights as context → better, more consistent insights over time

Run standalone:  python services/government_insight_service.py
Run once:        python services/government_insight_service.py --once
Force all:       python services/government_insight_service.py --once --force
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

if Path(__file__).resolve().parent.parent not in map(Path, sys.path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.insight_common import build_insight_llm, get_rag_context, get_weather_context, ping_with_retry

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "config" / ".env")

MONGO_URI = os.getenv("MONGO_URI", "").strip()
if not MONGO_URI:
    print("MONGO_URI not set. Check config/.env")
    sys.exit(1)

import certifi
from pymongo import MongoClient, ReplaceOne
from pymongo.errors import PyMongoError

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL = int(os.getenv("GOV_INSIGHT_INTERVAL", "3600"))
INSIGHT_LLM_DELAY = float(os.getenv("INSIGHT_LLM_DELAY", "1"))
STARTUP_GRACE_SECONDS = int(os.getenv(
    "GOVERNMENT_INSIGHT_GRACE_SECONDS",
    os.getenv("STARTUP_GRACE_SECONDS", "240"),
))
MONTHLY_REFRESH_DAYS = 30
INSIGHT_MIN_CHARS = 400
INSIGHT_MAX_CHARS = 500

MODEL_NAME = os.getenv("UTILITY_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

SOURCE_COLLECTIONS = [
    "hargakonsumenprovinsis",
    "hargaprodusenprovinsis",
    "cadanganpanganprovinsis",
    "hargakonsumennasionals",
    "konsumsiperjenis",
    "ketidakcukupanprovinsis",
    "hargaprodusennasionals",
    "proyeksineracas",
    "gerakanpanganmurahs",
    "penyalurandonasis",
    "panganterselamatkans",
    "ketidakcukupannasionals",
    "skorpphs",
    "variasihargaprodusens",
]

COLLECTION_THEMES = {
    "hargakonsumenprovinsis":  "harga konsumen pangan per provinsi",
    "hargaprodusenprovinsis":  "harga produsen pangan per provinsi",
    "cadanganpanganprovinsis": "cadangan pangan pemerintah daerah",
    "hargakonsumennasionals":  "harga konsumen pangan nasional",
    "konsumsiperjenis":        "konsumsi pangan per kapita",
    "ketidakcukupanprovinsis": "kerawanan pangan per provinsi",
    "hargaprodusennasionals":  "harga produsen pangan nasional",
    "proyeksineracas":         "neraca ketersediaan pangan nasional",
    "gerakanpanganmurahs":     "operasi pasar pangan murah",
    "penyalurandonasis":       "penyaluran donasi pangan",
    "panganterselamatkans":    "penyelamatan pangan food rescue",
    "ketidakcukupannasionals": "kerawanan pangan nasional",
    "skorpphs":                "skor pola pangan harapan",
    "variasihargaprodusens":   "variasi harga produsen antar wilayah",
}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

GOV_INSIGHT_PROMPT = ChatPromptTemplate.from_template(
    "Kamu adalah analis kebijakan pangan dan pertanian untuk Pemerintah Indonesia.\n\n"
    "Tugas: Buat SATU PARAGRAF insight strategis berdasarkan data berikut.\n\n"
    "ATURAN:\n"
    "- Tulis dalam SATU PARAGRAF padat (bukan list, bukan poin, bukan sub-heading)\n"
    "- Bahasa Indonesia formal untuk laporan pemerintah\n"
    "- Soroti: tren nasional, anomali, risiko, dan peluang utama\n"
    "- Sebutkan 1-2 angka kunci sebagai pendukung, jangan banjir data\n"
    "- Jangan generic — spesifik berdasarkan data yang diberikan\n"
    "- Akhiri dengan kalimat lengkap bertanda titik\n"
    "- Panjang: antara {min_chars} sampai {max_chars} karakter\n\n"
    "SUMBER DATA:\n"
    "{data_summary}\n\n"
    "{extra_context}\n\n"
    "INSIGHT (SATU PARAGRAF, 400-500 karakter):"
)


def _ask_llm(llm: OllamaLLM, prompt: str) -> str:
    try:
        return llm.invoke(prompt).strip()
    except Exception as e:
        print(f"    [LLM error] {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class GovernmentInsightEngine:

    def __init__(self):
        print(f"  GovInsight: Initializing (model={MODEL_NAME})...")

        # — MongoDB —
        # Hormati tls=false di URI (self-hosted tanpa TLS seperti
        # sugiecosystem.cloud) — jangan paksa tls=True kalau URI minta false.
        _use_tls = "tls=false" not in MONGO_URI.lower() and "ssl=false" not in MONGO_URI.lower()
        _tls_kwargs = dict(tls=True, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=False) if _use_tls else {}
        self._mongo = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=15_000,
            socketTimeoutMS=30_000,
            connectTimeoutMS=20_000,
            retryWrites=True,
            **_tls_kwargs,
        )
        ping_with_retry(self._mongo, label="GovInsight")
        print("  GovInsight: MongoDB connected.")

        self._source_db = self._mongo["test"]
        self._target_db = self._mongo["sugi_insights"]
        self._target_test_db = self._mongo["test"]

        try:
            self._target_db["governmentinsights"].create_index(
                "sourceCollection", unique=True, background=True
            )
            print("  GovInsight: Index ensured on governmentinsights.sourceCollection.")
        except Exception as e:
            # Best-effort: don't crash if user lacks dbAdmin on sugi_insights (code 13 Unauthorized)
            print(f"  GovInsight: skip index sugi_insights.governmentinsights: {e.__class__.__name__}: {e}")
            print("  GovInsight: continuing without index (ensure sugi_user has dbAdmin on sugi_insights if you want indexes).")

        # — LLM —
        self._llm = build_insight_llm(MODEL_NAME, temperature=0.3)
        print(f"  GovInsight: LLM ready ({MODEL_NAME}).")

        # — ChromaDB (government memory) —
        self._gov_memory = None
        self._init_chromadb()

        # — State —
        self._known_state: dict[str, dict] = {}
        self._load_state()

    # ─── ChromaDB Initialization ─────────────────────────────────────────

    def _init_chromadb(self):
        """Connect to ChromaDB and create/get government_memory collection."""
        try:
            import chromadb as _cdb
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings

            _client = _cdb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _client.heartbeat()

            _embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            self._gov_memory = Chroma(
                collection_name="government_memory",
                client=_client,
                embedding_function=_embeddings,
            )
            print(f"  GovInsight: ChromaDB connected ({CHROMA_HOST}:{CHROMA_PORT}), "
                  f"collection 'government_memory' ready.")
        except Exception as e:
            print(f"  GovInsight: ChromaDB not available — {e}")
            print("  GovInsight: Government memory disabled. Insights will not persist to ChromaDB.")

    def _chromadb_ready(self) -> bool:
        return self._gov_memory is not None

    # ─── ChromaDB: Store Insight ────────────────────────────────────────

    def _save_to_chromadb(self, col_name: str, insight: str, version: int, doc_count: int):
        """Store insight in ChromaDB for semantic retrieval on future generations."""
        if not self._chromadb_ready():
            return

        theme = COLLECTION_THEMES.get(col_name, col_name)
        doc_id = hashlib.md5(f"gov_mem:{col_name}:v{version}".encode()).hexdigest()

        # Check if this exact version already exists (dedup)
        try:
            existing = self._gov_memory.get(ids=[doc_id])
            if existing["ids"]:
                return
        except Exception:
            pass

        doc = Document(
            page_content=insight,
            metadata={
                "source": "government_insight",
                "sourceCollection": col_name,
                "theme": theme,
                "insightVersion": version,
                "totalDocuments": doc_count,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
            },
            id=doc_id,
        )
        try:
            self._gov_memory.add_documents(documents=[doc], ids=[doc_id])
            print(f"    [chroma] stored version {version} for {col_name}")
        except Exception as e:
            print(f"    [chroma] store error: {e}")

    # ─── ChromaDB: Retrieve Past Insights (Learning Loop) ───────────────

    def _get_memory_from_chromadb(self, col_name: str) -> str:
        """
        Retrieve semantically related past government insights from ChromaDB.
        This creates a learning loop: past insights inform future generations.
        """
        if not self._chromadb_ready():
            return ""

        theme = COLLECTION_THEMES.get(col_name, col_name)

        # Query by both the collection name and theme for broad retrieval
        query = f"government insight {theme} {col_name}"
        try:
            docs = self._gov_memory.similarity_search(query, k=4)

            # Filter: exclude current collection's own latest to avoid circular reference
            # but DO include other collections' insights as cross-domain knowledge
            past_lines = []
            for d in docs:
                src = d.metadata.get("sourceCollection", "unknown")
                ver = d.metadata.get("insightVersion", "?")
                label = f"[{src} v{ver}]"
                content = d.page_content[:250]
                past_lines.append(f"{label}: {content}")

            if past_lines:
                result = "POLA DARI INSIGHT SEBELUMNYA (cross-collection learning):\n" + "\n".join(past_lines)
                print(f"    [memory] retrieved {len(docs)} past insights from ChromaDB")
                return result
        except Exception as e:
            print(f"    [memory] retrieval error: {e}")

        return ""

    # ─── Public API ─────────────────────────────────────────────────────

    def run_once(self, force_all: bool = False):
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'=' * 60}")
        print(f"  Government Insight Engine — {now_str} UTC")
        print(f"{'=' * 60}")

        for i, col_name in enumerate(SOURCE_COLLECTIONS):
            try:
                if force_all:
                    self._process_collection(col_name)
                else:
                    self._process_if_changed(col_name)
            except Exception as e:
                # A1: sebut tipe exception + nama field agar diagnosable. See docs/decisions.md#a1
                print(f"  [ERROR] {col_name}: {type(e).__name__}: {e} (field: tahun)")
                traceback.print_exc()
            finally:
                # M0b: jeda antar LLM call. See docs/decisions.md#m0
                if i < len(SOURCE_COLLECTIONS) - 1 and INSIGHT_LLM_DELAY > 0:
                    time.sleep(INSIGHT_LLM_DELAY)

        self._check_monthly_refresh()

    def run_loop(self, interval: int = POLL_INTERVAL):
        print(f"\n  [loop] Polling every {interval}s. Ctrl+C to stop.\n")
        if STARTUP_GRACE_SECONDS > 0:
            print(f"  ⏳  Startup grace {STARTUP_GRACE_SECONDS}s — biarkan chatbot "
                  f"dan watchers warm-up dulu (hindari kontes Ollama)...")
            time.sleep(STARTUP_GRACE_SECONDS)
        self.run_once(force_all=True)
        while True:
            try:
                time.sleep(interval)
                self.run_once(force_all=False)
            except KeyboardInterrupt:
                print("\n  GovInsight: Stopped by user.")
                break

    # ─── Change Detection ───────────────────────────────────────────────

    def _compute_signature(self, col_name: str) -> Optional[dict]:
        try:
            col = self._source_db[col_name]
            count = col.count_documents({})
            if count == 0:
                return {"count": 0, "signature": "empty"}
            pipeline = [{"$sort": {"_id": -1}}, {"$limit": 100}, {"$project": {"_id": 1}}]
            ids = [str(doc["_id"]) for doc in col.aggregate(pipeline, allowDiskUse=True)]
            signature = hashlib.md5("".join(ids).encode()).hexdigest()
            return {"count": count, "signature": signature}
        except Exception as e:
            print(f"    [sig] error: {e}")
            return None

    def _has_changed(self, col_name: str) -> bool:
        sig = self._compute_signature(col_name)
        if sig is None:
            return False
        old = self._known_state.get(col_name)
        if old is None:
            return True
        return old.get("count") != sig["count"] or old.get("signature") != sig["signature"]

    def _process_if_changed(self, col_name: str):
        if self._has_changed(col_name):
            print(f"  [changed] {col_name} — regenerating...")
            self._process_collection(col_name)
        else:
            existing = self._target_db["governmentinsights"].find_one(
                {"sourceCollection": col_name}
            )
            if existing and len(existing.get("insight", "")) >= INSIGHT_MIN_CHARS:
                print(f"  [stable]  {col_name} — insight up to date.")
            else:
                print(f"  [missing] {col_name} — insight missing/short, generating...")
                self._process_collection(col_name)

    # ─── Data Fetching ──────────────────────────────────────────────────

    def _build_data_summary(self, col_name: str) -> str:
        col = self._source_db[col_name]
        total = col.count_documents({})

        if total == 0:
            return f"Koleksi '{col_name}' saat ini kosong — belum ada data yang tercatat."

        sample = list(col.find().sort("_id", 1).limit(5))
        fields = [k for k in sample[0].keys() if k not in ("_id", "__v", "createdAt", "updatedAt")]

        parts = [f"Koleksi: {col_name}", f"Jumlah dokumen: {total}", f"Kolom: {', '.join(fields)}"]

        if "tahun" in fields:
            # A1: distinct("tahun") bisa campuran None/str/int — sort aman.
            # See docs/decisions.md#a1
            raw_years = col.distinct("tahun")
            years = sorted(
                (y for y in raw_years if y is not None),
                key=lambda y: str(y),
            )
            if not years:
                parts.append("Tahun: tidak ada data tahun tersedia")
            elif len(years) > 1:
                parts.append(f"Periode data: {str(years[0])} - {str(years[-1])} ({len(years)} tahun)")
            else:
                parts.append(f"Tahun: {years[0]}")

        for k in ("provinsi", "wilayah", "nama_provinsi"):
            if k in fields:
                provs = col.distinct(k)
                prov_list = [str(p) for p in provs if p]
                display = ", ".join(sorted(set(prov_list))[:7])
                if len(set(prov_list)) > 7:
                    display += f" dan {len(set(prov_list)) - 7} lainnya"
                parts.append(f"Wilayah: {display}")
                break

        if "komoditas" in fields:
            komoditas = col.distinct("komoditas")
            komo_list = [str(k) for k in komoditas if k]
            display = ", ".join(sorted(set(komo_list))[:8])
            if len(set(komo_list)) > 8:
                display += f" dan {len(set(komo_list)) - 8} lainnya"
            parts.append(f"Komoditas: {display}")

        numeric_candidates = {
            "harga", "cppd_ton", "jumlah_donasi_kg", "penerima_manfaat_jiwa",
            "jumlah_penduduk", "penduduk_undernourish", "ketersediaan",
            "kebutuhan", "neraca", "konsumsi_pangan", "koefisien_variasi",
            "pou", "pph_ketersediaan", "pelaksana",
        }
        for nf in numeric_candidates:
            if nf in fields:
                try:
                    pipeline = [
                        {"$match": {nf: {"$exists": True, "$type": ["int", "long", "double", "decimal"]}}},
                        {"$group": {
                            "_id": None,
                            "min": {"$min": f"${nf}"},
                            "max": {"$max": f"${nf}"},
                            "avg": {"$avg": f"${nf}"},
                        }},
                    ]
                    result = list(col.aggregate(pipeline, allowDiskUse=True))
                    if result:
                        r = result[0]
                        if nf in ("pou", "koefisien_variasi", "pph_ketersediaan"):
                            parts.append(f"{nf}: {r['min']} - {r['max']} (rata-rata {r['avg']:.2f})")
                        else:
                            parts.append(f"{nf}: {r['min']:,.0f} - {r['max']:,.0f} (rata-rata {r['avg']:,.0f})")
                except Exception:
                    pass

        parts.append("\nContoh data (3 baris pertama):")
        for doc in sample[:3]:
            clean = {k: v for k, v in doc.items() if k not in ("_id", "__v", "createdAt", "updatedAt")}
            parts.append(str(clean))

        return "\n".join(parts)

    def _get_rag_context(self, theme: str) -> str:
        return get_rag_context(
            theme, k=3, label="KONTEKS DARI BASIS PENGETAHUAN PERTANIAN:", content_chars=300
        )

    def _get_weather_context(self) -> str:
        return get_weather_context(k=2, label="KONDISI CUACA TERKINI:", content_chars=400)

    # ─── Insight Generation ─────────────────────────────────────────────

    def _generate_insight(self, col_name: str, data_summary: str) -> str:
        theme = COLLECTION_THEMES.get(col_name, col_name)

        # Gather context from all SUGI ecosystem sources
        contexts = {
            "rag": self._get_rag_context(theme),
            "weather": self._get_weather_context(),
            "mongodb_prev": self._get_previous_insight(col_name),
            "chromadb_memory": self._get_memory_from_chromadb(col_name),
        }

        # Build extra context section with labeled sources
        extra_parts = []
        for label, text in contexts.items():
            if text:
                extra_parts.append(text)

        extra = "\n\n".join(extra_parts) if extra_parts else "Tidak ada konteks tambahan yang tersedia."

        prompt = GOV_INSIGHT_PROMPT.format(
            data_summary=data_summary,
            extra_context=extra,
            min_chars=str(INSIGHT_MIN_CHARS),
            max_chars=str(INSIGHT_MAX_CHARS),
        )

        raw = _ask_llm(self._llm, prompt)
        return self._validate_and_fix(raw, col_name)

    def _validate_and_fix(self, text: str, col_name: str) -> str:
        if not text:
            return f"Insight untuk {col_name} belum dapat dihasilkan. Silakan coba lagi nanti."

        cleaned = text.strip()
        cleaned = cleaned.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
        for token in ("Insight:", "INSIGHT:", "insight:", "Insight", "INSIGHT"):
            if token in cleaned[:40]:
                cleaned = cleaned.split(token, 1)[-1].strip()
        if ":" in cleaned[:20] and len(cleaned) > 40:
            cleaned = cleaned.split(":", 1)[-1].strip()
        cleaned = re.sub(r"\n\s*\n", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"(?:^|\s)\d+\.\s*", " ", cleaned).strip()

        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."

        length = len(cleaned)
        theme = COLLECTION_THEMES.get(col_name, col_name)

        if length < INSIGHT_MIN_CHARS:
            suffix_variants = [
                f" Pemantauan berkelanjutan terhadap {theme} perlu terus dilakukan untuk mendukung ketahanan pangan nasional.",
                f" Data {theme} ini menjadi bahan evaluasi bagi pemerintah dalam merumuskan kebijakan ke depan.",
                f" Pemerintah perlu mencermati perkembangan {theme} ini secara berkala.",
            ]
            for suffix in suffix_variants:
                if length + len(suffix) <= INSIGHT_MAX_CHARS:
                    cleaned += suffix
                    length = len(cleaned)
                    break

        if len(cleaned) > INSIGHT_MAX_CHARS:
            truncated = cleaned[: INSIGHT_MAX_CHARS - 3]
            last_period = truncated.rfind(".")
            if last_period > INSIGHT_MIN_CHARS - 50:
                cleaned = truncated[: last_period + 1]
            else:
                cleaned = truncated + "."

        print(f"    [insight] {len(cleaned)} chars — valid")
        return cleaned

    # ─── Save to MongoDB + ChromaDB ─────────────────────────────────────

    def _save(self, col_name: str, insight: str, doc_count: int):
        now = datetime.now(timezone.utc)
        existing = self._target_db["governmentinsights"].find_one(
            {"sourceCollection": col_name}
        )
        version = (existing.get("insightVersion", 0) if existing else 0) + 1

        doc = {
            "sourceCollection": col_name,
            "insight": insight,
            "generatedAt": now,
            "lastDataUpdate": now,
            "totalDocuments": doc_count,
            "insightVersion": version,
            "status": "active",
            "metadata": {
                "model": MODEL_NAME,
                "rag_context": True,
                "weather_context": True,
                "chromadb_memory": True,
            },
        }
        try:
            self._target_db["governmentinsights"].replace_one(
                {"sourceCollection": col_name}, doc, upsert=True
            )
            print(f"    [saved] sugi_insights.governmentinsights version {version}")
        except PyMongoError as e:
            print(f"    [error] MongoDB (sugi_insights) save failed: {e}")

        # Duplicate to test database (same collection name)
        try:
            self._target_test_db["governmentinsights"].replace_one(
                {"sourceCollection": col_name}, doc, upsert=True
            )
            print(f"    [saved] test.governmentinsights version {version}")
        except PyMongoError as e:
            print(f"    [error] MongoDB (test.governmentinsights) save failed: {e}")

        # Also persist to ChromaDB for the learning loop
        self._save_to_chromadb(col_name, insight, version, doc_count)

    def _get_previous_insight(self, col_name: str) -> str:
        try:
            doc = self._target_db["governmentinsights"].find_one(
                {"sourceCollection": col_name},
                sort=[("insightVersion", -1)],
            )
            if doc and doc.get("insight"):
                return f"INSIGHT SEBELUMNYA ({col_name} versi {doc.get('insightVersion', '?')}):\n{doc['insight'][:300]}"
        except Exception:
            pass
        return ""

    def _process_collection(self, col_name: str):
        print(f"\n  {'-' * 50}")
        print(f"  Processing: {col_name}")

        data_summary = self._build_data_summary(col_name)
        doc_count = self._source_db[col_name].count_documents({})

        print(f"    Data summary: {len(data_summary)} chars, {doc_count} docs")

        insight = self._generate_insight(col_name, data_summary)
        self._save(col_name, insight, doc_count)

        sig = self._compute_signature(col_name)
        if sig:
            self._known_state[col_name] = sig
            self._save_state()

    # ─── State Persistence ──────────────────────────────────────────────

    def _state_path(self) -> Path:
        p = _ROOT / "data" / "db" / "gov_insight_state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_state(self):
        p = self._state_path()
        if p.exists():
            try:
                self._known_state = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                self._known_state = {}

    def _save_state(self):
        try:
            self._state_path().write_text(
                json.dumps(self._known_state, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"    [state] save error: {e}")

    # ─── Monthly Refresh ────────────────────────────────────────────────

    def _check_monthly_refresh(self):
        now = datetime.now(timezone.utc)
        all_insights = list(
            self._target_db["governmentinsights"].find({}, {"generatedAt": 1})
        )
        if not all_insights:
            print("\n  [monthly] No insights exist. Running full generation.")
            self.run_once(force_all=True)
            return

        oldest = now
        for doc in all_insights:
            ga = doc.get("generatedAt")
            if ga:
                if ga.tzinfo is None:
                    ga = ga.replace(tzinfo=timezone.utc)
                if ga < oldest:
                    oldest = ga

        days = (now - oldest).days
        if days >= MONTHLY_REFRESH_DAYS:
            print(f"\n  [monthly] {days} days since oldest — regenerating all.")
            self.run_once(force_all=True)
        else:
            remaining = MONTHLY_REFRESH_DAYS - days
            print(f"\n  [monthly] Next full refresh in ~{remaining} days.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Government Insight Engine")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--force", action="store_true", help="Force regenerate all")
    parser.add_argument(
        "--interval", type=int, default=POLL_INTERVAL,
        help=f"Polling interval seconds (default: {POLL_INTERVAL})",
    )
    args = parser.parse_args()

    engine = GovernmentInsightEngine()

    if args.once:
        engine.run_once(force_all=args.force)
    else:
        engine.run_loop(interval=args.interval)


if __name__ == "__main__":
    main()
