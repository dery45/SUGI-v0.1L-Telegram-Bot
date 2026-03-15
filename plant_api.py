import os
import hashlib
import time
import random
import threading
import requests
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

load_dotenv()

PERENUAL_KEY     = os.getenv("PERENUAL_API_KEY", "")
PERENUAL_BASE    = "https://perenual.com/api/v2"
PERENUAL_BASE_V1 = "https://perenual.com/api"

EMBED_MODEL = "mxbai-embed-large"
# ─── ChromaDB server connection ──────────────────────────────────────────────
# Ganti host/port sesuai setup kamu.
# Default: server jalan di mesin yang sama (localhost:8000)
# Remote server: ganti "localhost" dengan IP/hostname server
import chromadb as _chromadb
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
_chroma_client = _chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

if not PERENUAL_KEY or PERENUAL_KEY == "sk-your-api-key-here":
    print("⚠️  PERENUAL_API_KEY not set in .env — plant API features disabled.")

_embeddings = OllamaEmbeddings(model=EMBED_MODEL)
plant_store = Chroma(
    collection_name="plant_data",
    client=_chroma_client,
    embedding_function=_embeddings,
)


# ─── Token bucket rate limiter ────────────────────────────────────────────────
class _ApiQueue:
    """
    Token bucket yang membatasi laju request ke Perenual API.
    Default: max 1 request per detik (sesuai free tier ~60 req/menit).
    Thread-safe menggunakan Lock.
    """
    def __init__(self, min_interval: float = 1.1):
        self._min_interval = min_interval   # detik antar request
        self._last_call    = 0.0
        self._lock         = threading.Lock()

    def wait(self):
        """Blokir sampai boleh mengirim request berikutnya."""
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                sleep_for = self._min_interval - elapsed
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


_queue = _ApiQueue(min_interval=1.1)


# ─── HTTP helper dengan exponential backoff ───────────────────────────────────
_MAX_RETRIES  = 4
_BACKOFF_BASE = 2.0   # detik, digandakan setiap retry
_JITTER_MAX   = 0.5   # detik acak tambahan untuk menghindari thundering herd


def _get(url: str, params: dict) -> Optional[dict]:
    """
    GET request ke Perenual API dengan:
      - Token bucket rate limiting (tunggu giliran sebelum kirim)
      - Exponential backoff untuk 429 Too Many Requests
      - Menghormati header Retry-After kalau ada
      - Jitter acak di setiap delay

    Retry schedule (tanpa jitter):
      Attempt 1: langsung
      Attempt 2: tunggu 2s
      Attempt 3: tunggu 4s
      Attempt 4: tunggu 8s
      Attempt 5: tunggu 16s → lalu return None
    """
    attempt      = 0
    backoff_secs = _BACKOFF_BASE

    while attempt <= _MAX_RETRIES:
        _queue.wait()   # rate limit: tunggu giliran

        try:
            resp = requests.get(url, params=params, timeout=12)

            # 429 → backoff dan retry
            if resp.status_code == 429:
                attempt += 1
                if attempt > _MAX_RETRIES:
                    print(f"   ❌ 429 after {_MAX_RETRIES} retries — giving up: {url}")
                    return None

                # Cek header Retry-After dari server
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = backoff_secs
                else:
                    wait = backoff_secs

                jitter  = random.uniform(0, _JITTER_MAX)
                total   = wait + jitter
                print(f"   ⏳ 429 rate limit — retry {attempt}/{_MAX_RETRIES} "
                      f"in {total:.1f}s (backoff={wait:.0f}s + jitter={jitter:.2f}s)...")
                time.sleep(total)
                backoff_secs *= 2   # eksponensial
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            attempt += 1
            jitter  = random.uniform(0, _JITTER_MAX)
            wait    = backoff_secs + jitter
            print(f"   ⏳ Timeout — retry {attempt}/{_MAX_RETRIES} in {wait:.1f}s...")
            time.sleep(wait)
            backoff_secs *= 2

        except requests.exceptions.RequestException as e:
            # Error non-429 (500, network, dll.) — tidak di-retry
            print(f"   ⚠️  API error: {e}")
            return None

    return None


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _doc_id(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def _already_cached(cache_key: str) -> bool:
    result = plant_store.get(where={"cache_key": cache_key}, limit=1)
    return bool(result["ids"])


def _store_docs(documents: list[Document]):
    if not documents:
        return
    ids      = [doc.id for doc in documents]
    existing = plant_store.get(ids=ids)["ids"]
    new_docs = [d for d in documents if d.id not in existing]
    new_ids  = [d.id for d in new_docs]
    if new_docs:
        plant_store.add_documents(documents=new_docs, ids=new_ids)
        print(f"   💾 Stored {len(new_docs)} new plant docs to ChromaDB.")


# ─── API 1: Species search + details ─────────────────────────────────────────
def _fetch_species_list(plant_name: str, page: int = 1) -> list[dict]:
    data = _get(f"{PERENUAL_BASE}/species-list", {
        "q": plant_name, "page": page, "key": PERENUAL_KEY
    })
    return data.get("data", []) if data else []


def _fetch_species_detail(plant_id: int) -> Optional[dict]:
    return _get(f"{PERENUAL_BASE}/species/details/{plant_id}", {"key": PERENUAL_KEY})


def _species_to_text(detail: dict) -> str:
    lines = [
        f"Plant: {detail.get('common_name', 'Unknown')}",
        f"Scientific name: {', '.join(detail.get('scientific_name', []))}",
        f"Family: {detail.get('family', '')}",
        f"Type: {detail.get('type', '')}",
        f"Cycle: {detail.get('cycle', '')}",
        f"Watering: {detail.get('watering', '')}",
        f"Sunlight: {', '.join(detail.get('sunlight', []))}",
        f"Care level: {detail.get('care_level', '')}",
        f"Growth rate: {detail.get('growth_rate', '')}",
        f"Propagation: {', '.join(detail.get('propagation', []))}",
        f"Soil: {', '.join(detail.get('soil', []))}",
        f"Origin: {', '.join(detail.get('origin', []))}",
        f"Indoor: {detail.get('indoor', '')}",
        f"Tropical: {detail.get('tropical', '')}",
        f"Drought tolerant: {detail.get('drought_tolerant', '')}",
        f"Edible fruit: {detail.get('edible_fruit', '')}",
        f"Harvest season: {detail.get('harvest_season', '')}",
        f"Medicinal: {detail.get('medicinal', '')}",
        f"Pest susceptibility: {', '.join(detail.get('pest_susceptibility', []))}",
    ]
    desc = detail.get("description", "")
    if desc:
        lines.append(f"Description: {desc}")
    return "\n".join(l for l in lines if not l.endswith(": "))


def fetch_plant_species(plant_name: str) -> list[Document]:
    cache_key = f"species:{plant_name.lower().strip()}"
    if _already_cached(cache_key):
        print(f"   ✅ Plant '{plant_name}' found in local cache.")
        results = plant_store.get(
            where={"cache_key": cache_key}, include=["documents", "metadatas"]
        )
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    print(f"   🌐 Fetching species data for '{plant_name}' from Perenual...")
    species_list = _fetch_species_list(plant_name)
    if not species_list:
        print(f"   ℹ️  No species results found for '{plant_name}'.")
        return []

    documents = []
    for item in species_list[:5]:
        plant_id = item.get("id")
        if not plant_id:
            continue
        detail = _fetch_species_detail(plant_id)
        if not detail:
            continue
        text   = _species_to_text(detail)
        doc_id = _doc_id(f"species:{plant_id}")
        img_url = ""
        if detail.get("default_image"):
            img_url = detail["default_image"].get("regular_url", "")
        documents.append(Document(
            page_content=text,
            metadata={
                "source":      "perenual_species",
                "cache_key":   cache_key,
                "plant_id":    str(plant_id),
                "common_name": detail.get("common_name", ""),
                "image_url":   img_url,
            },
            id=doc_id,
        ))

    _store_docs(documents)
    return documents


# ─── API 2: Pest & Disease ────────────────────────────────────────────────────
def _disease_to_text(item: dict) -> str:
    lines = [
        f"Pest/Disease: {item.get('common_name', 'Unknown')}",
        f"Scientific name: {item.get('scientific_name', '')}",
        f"Host plants: {', '.join(item.get('host', []))}",
    ]
    for section in item.get("description", []):
        subtitle = section.get("subtitle", "")
        desc     = section.get("description", "")
        if subtitle and desc:
            lines.append(f"{subtitle}: {desc}")
    for section in item.get("solution", []):
        subtitle = section.get("subtitle", "")
        desc     = section.get("description", "")
        if subtitle and desc:
            lines.append(f"Solution - {subtitle}: {desc}")
    return "\n".join(l for l in lines if not l.endswith(": "))


def fetch_pest_disease(query: str) -> list[Document]:
    cache_key = f"disease:{query.lower().strip()}"
    if _already_cached(cache_key):
        print(f"   ✅ Disease '{query}' found in local cache.")
        results = plant_store.get(
            where={"cache_key": cache_key}, include=["documents", "metadatas"]
        )
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    print(f"   🌐 Fetching pest/disease data for '{query}' from Perenual...")
    data = _get(f"{PERENUAL_BASE_V1}/pest-disease-list", {
        "q": query, "page": 1, "key": PERENUAL_KEY
    })
    if not data:
        return []

    items     = data.get("data", [])
    documents = []
    for item in items[:5]:
        text   = _disease_to_text(item)
        doc_id = _doc_id(f"disease:{item.get('id', query)}")
        documents.append(Document(
            page_content=text,
            metadata={
                "source":      "perenual_disease",
                "cache_key":   cache_key,
                "disease_id":  str(item.get("id", "")),
                "common_name": item.get("common_name", ""),
            },
            id=doc_id,
        ))

    _store_docs(documents)
    return documents


# ─── API 3: Care guides ───────────────────────────────────────────────────────
def _care_guide_to_text(species_id: int, guides: list[dict]) -> str:
    lines = [f"Care guides for species ID {species_id}:"]
    for guide in guides:
        for s in guide.get("section", []):
            subtitle = s.get("type", "")
            desc     = s.get("description", "")
            if subtitle and desc:
                lines.append(f"{subtitle}: {desc}")
    return "\n".join(lines)


def fetch_care_guides(species_id: int) -> list[Document]:
    cache_key = f"care:{species_id}"
    if _already_cached(cache_key):
        print(f"   ✅ Care guides for species {species_id} found in local cache.")
        results = plant_store.get(
            where={"cache_key": cache_key}, include=["documents", "metadatas"]
        )
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    print(f"   🌐 Fetching care guides for species {species_id} from Perenual...")
    data = _get(f"{PERENUAL_BASE_V1}/species-care-guide-list", {
        "species_id": species_id, "page": 1, "key": PERENUAL_KEY
    })
    if not data:
        return []

    items = data.get("data", [])
    if not items:
        return []

    text   = _care_guide_to_text(species_id, items)
    doc_id = _doc_id(f"care:{species_id}")
    doc    = Document(
        page_content=text,
        metadata={
            "source":     "perenual_care_guide",
            "cache_key":  cache_key,
            "species_id": str(species_id),
        },
        id=doc_id,
    )
    _store_docs([doc])
    return [doc]


# ─── Public API ───────────────────────────────────────────────────────────────
def is_plant_cached(plant_name: str) -> bool:
    name = plant_name.lower().strip()
    return (
        _already_cached(f"species:{name}") and
        _already_cached(f"disease:{name}")
    )


def get_cached_plant_docs(plant_name: str, k: int = 6) -> list[Document]:
    try:
        return plant_store.similarity_search(plant_name, k=k)
    except Exception:
        return []


def search_plant_info(plant_name: str) -> list[Document]:
    if not PERENUAL_KEY or PERENUAL_KEY == "sk-your-api-key-here":
        print("   ⚠️  PERENUAL_API_KEY not configured — skipping API fetch.")
        return []

    all_docs: list[Document] = []

    species_docs = fetch_plant_species(plant_name)
    all_docs.extend(species_docs)

    disease_docs = fetch_pest_disease(plant_name)
    all_docs.extend(disease_docs)

    if species_docs:
        species_id = species_docs[0].metadata.get("plant_id")
        if species_id:
            care_docs = fetch_care_guides(int(species_id))
            all_docs.extend(care_docs)

    return all_docs


def search_plant_rag(query: str, k: int = 5) -> list[Document]:
    try:
        return plant_store.similarity_search(query, k=k)
    except Exception:
        return []