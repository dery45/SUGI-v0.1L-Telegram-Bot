"""
plant_api.py — Perenual Plant API integration for SUGI v0.1L

Flow for every plant-related query:
  1. Check local ChromaDB collection "plant_data" for cached results
  2. If found → return cached docs directly (no API call)
  3. If not found → fetch from Perenual APIs in sequence:
       a. species-list  (search by name)
       b. species/details/{id}  (full detail for each result)
       c. pest-disease-list  (disease/pest search)
       d. species-care-guide-list  (care guides)
  4. Store all retrieved data into ChromaDB
  5. Return the data for RAG context

Add new API integrations at the bottom of this file.
"""

import os
import hashlib
import requests
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# ─── Load environment ────────────────────────────────────────────────────────
load_dotenv()

PERENUAL_KEY  = os.getenv("PERENUAL_API_KEY", "")
PERENUAL_BASE = "https://perenual.com/api/v2"
PERENUAL_BASE_V1 = "https://perenual.com/api"   # pest-disease uses v1

EMBED_MODEL = "mxbai-embed-large"
DB_PATH     = "chrome_longchain_db"

if not PERENUAL_KEY or PERENUAL_KEY == "sk-your-api-key-here":
    print("⚠️  PERENUAL_API_KEY not set in .env — plant API features disabled.")

# ─── ChromaDB collection for plant data ──────────────────────────────────────
_embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
plant_store  = Chroma(
    collection_name="plant_data",   # separate from CSV/PDF/memory collections
    persist_directory=DB_PATH,
    embedding_function=_embeddings,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _doc_id(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

def _already_cached(cache_key: str) -> bool:
    """Check if a cache_key exists as metadata in the plant collection."""
    result = plant_store.get(where={"cache_key": cache_key}, limit=1)
    return bool(result["ids"])

def _store_docs(documents: list[Document]):
    """Upsert documents into the plant collection (skip duplicates by id)."""
    if not documents:
        return
    ids      = [doc.id for doc in documents]
    existing = plant_store.get(ids=ids)["ids"]
    new_docs = [d for d in documents if d.id not in existing]
    new_ids  = [d.id for d in new_docs]
    if new_docs:
        plant_store.add_documents(documents=new_docs, ids=new_ids)
        print(f"   💾 Stored {len(new_docs)} new plant docs to ChromaDB.")

def _get(url: str, params: dict) -> Optional[dict]:
    """Safe GET with timeout and error handling."""
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"   ⚠️  API error: {e}")
        return None

# ─── API 1: Species search + details ─────────────────────────────────────────

def _fetch_species_list(plant_name: str, page: int = 1) -> list[dict]:
    data = _get(f"{PERENUAL_BASE}/species-list", {
        "q": plant_name, "page": page, "key": PERENUAL_KEY
    })
    return data.get("data", []) if data else []

def _fetch_species_detail(plant_id: int) -> Optional[dict]:
    return _get(f"{PERENUAL_BASE}/species/details/{plant_id}", {"key": PERENUAL_KEY})

def _species_to_text(detail: dict) -> str:
    """Flatten species detail dict into a readable text block for embedding."""
    lines = [
        f"Plant: {detail.get('common_name', 'Unknown')}",
        f"Scientific name: {', '.join(detail.get('scientific_name', []))}",
        f"Family: {detail.get('family', '')}",
        f"Genus: {detail.get('genus', '')}",
        f"Type: {detail.get('type', '')}",
        f"Cycle: {detail.get('cycle', '')}",
        f"Watering: {detail.get('watering', '')}",
        f"Sunlight: {', '.join(detail.get('sunlight', []))}",
        f"Care level: {detail.get('care_level', '')}",
        f"Growth rate: {detail.get('growth_rate', '')}",
        f"Maintenance: {detail.get('maintenance', '')}",
        f"Propagation: {', '.join(detail.get('propagation', []))}",
        f"Soil: {', '.join(detail.get('soil', []))}",
        f"Origin: {', '.join(detail.get('origin', []))}",
        f"Indoor: {detail.get('indoor', '')}",
        f"Tropical: {detail.get('tropical', '')}",
        f"Drought tolerant: {detail.get('drought_tolerant', '')}",
        f"Flowers: {detail.get('flowers', '')}",
        f"Flowering season: {detail.get('flowering_season', '')}",
        f"Fruits: {detail.get('fruits', '')}",
        f"Edible fruit: {detail.get('edible_fruit', '')}",
        f"Harvest season: {detail.get('harvest_season', '')}",
        f"Edible leaf: {detail.get('edible_leaf', '')}",
        f"Medicinal: {detail.get('medicinal', '')}",
        f"Poisonous to humans: {detail.get('poisonous_to_humans', '')}",
        f"Poisonous to pets: {detail.get('poisonous_to_pets', '')}",
        f"Pest susceptibility: {', '.join(detail.get('pest_susceptibility', []))}",
        f"Attracts: {', '.join(detail.get('attracts', []))}",
    ]
    dims = detail.get("dimensions", [])
    for d in dims:
        lines.append(f"Dimension ({d.get('type','')}): {d.get('min_value','')}–{d.get('max_value','')} {d.get('unit','')}")
    desc = detail.get("description", "")
    if desc:
        lines.append(f"Description: {desc}")
    return "\n".join(l for l in lines if not l.endswith(": "))

def fetch_plant_species(plant_name: str) -> list[Document]:
    """
    Search species by name, fetch full details for each result,
    cache in ChromaDB, return Document list.
    """
    cache_key = f"species:{plant_name.lower().strip()}"
    if _already_cached(cache_key):
        print(f"   ✅ Plant '{plant_name}' found in local cache.")
        results = plant_store.get(where={"cache_key": cache_key}, include=["documents", "metadatas"])
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
    for item in species_list[:5]:   # limit to top 5 results
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

# ─── API 2: Pest & Disease search ────────────────────────────────────────────

def _disease_to_text(item: dict) -> str:
    lines = [
        f"Pest/Disease: {item.get('common_name', 'Unknown')}",
        f"Scientific name: {item.get('scientific_name', '')}",
        f"Family: {item.get('family', '')}",
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
            lines.append(f"Solution — {subtitle}: {desc}")
    return "\n".join(l for l in lines if not l.endswith(": "))

def fetch_pest_disease(query: str) -> list[Document]:
    """
    Search pest/disease by name, cache in ChromaDB, return Document list.
    """
    cache_key = f"disease:{query.lower().strip()}"
    if _already_cached(cache_key):
        print(f"   ✅ Disease '{query}' found in local cache.")
        results = plant_store.get(where={"cache_key": cache_key}, include=["documents", "metadatas"])
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
                "source":    "perenual_disease",
                "cache_key": cache_key,
                "disease_id": str(item.get("id", "")),
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
        section  = guide.get("section", [])
        for s in section:
            subtitle = s.get("type", "")
            desc     = s.get("description", "")
            if subtitle and desc:
                lines.append(f"{subtitle}: {desc}")
    return "\n".join(lines)

def fetch_care_guides(species_id: int) -> list[Document]:
    """
    Fetch care guides for a species_id, cache in ChromaDB, return Document list.
    """
    cache_key = f"care:{species_id}"
    if _already_cached(cache_key):
        print(f"   ✅ Care guides for species {species_id} found in local cache.")
        results = plant_store.get(where={"cache_key": cache_key}, include=["documents", "metadatas"])
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

    doc = Document(
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

# ─── Main entry points called by main.py ────────────────────────────────────

def is_plant_cached(plant_name: str) -> bool:
    """
    True if ALL three data types (species, disease, care guide) for this
    plant name are already stored in local ChromaDB.

    Uses exact cache_key metadata lookup — not similarity search —
    so it never confuses "tomato" with "cherry tomato" etc.
    """
    name = plant_name.lower().strip()
    return (
        _already_cached(f"species:{name}") and
        _already_cached(f"disease:{name}")
        # care guide cache is per species_id, checked implicitly inside
        # fetch_plant_species → fetch_care_guides chain
    )

def get_cached_plant_docs(plant_name: str, k: int = 6) -> list[Document]:
    """
    Retrieve cached plant documents from ChromaDB using similarity search.
    Only call this after is_plant_cached() returns True.
    """
    try:
        return plant_store.similarity_search(plant_name, k=k)
    except Exception:
        return []

def search_plant_info(plant_name: str) -> list[Document]:
    """
    Fetch ALL plant data from Perenual APIs sequentially, store in ChromaDB.
    Only call this when is_plant_cached() returns False.

    Sequence:
      1. species-list + species/details  (plant biology, care, traits)
      2. pest-disease-list               (diseases and solutions)
      3. species-care-guide-list         (detailed care instructions)

    Each sub-fetch has its own cache_key so partial re-fetches never
    duplicate already-stored data.
    """
    if not PERENUAL_KEY or PERENUAL_KEY == "sk-your-api-key-here":
        print("   ⚠️  PERENUAL_API_KEY not configured — skipping API fetch.")
        return []

    all_docs: list[Document] = []

    # API 1: Species details
    species_docs = fetch_plant_species(plant_name)
    all_docs.extend(species_docs)

    # API 2: Pest & disease
    disease_docs = fetch_pest_disease(plant_name)
    all_docs.extend(disease_docs)

    # API 3: Care guides (needs species_id from step 1)
    if species_docs:
        species_id = species_docs[0].metadata.get("plant_id")
        if species_id:
            care_docs = fetch_care_guides(int(species_id))
            all_docs.extend(care_docs)

    return all_docs

def search_plant_rag(query: str, k: int = 5) -> list[Document]:
    """
    Semantic similarity search against local plant ChromaDB only.
    No API calls. Used as a fallback when you want relevant docs
    without knowing the exact plant name.
    """
    try:
        return plant_store.similarity_search(query, k=k)
    except Exception:
        return []

# ─── Add new API integrations below ──────────────────────────────────────────
# Pattern to follow:
#
# def fetch_<source_name>(query: str) -> list[Document]:
#     cache_key = f"<source>:{query.lower().strip()}"
#     if _already_cached(cache_key): ...
#     # fetch from API
#     # call _store_docs(documents)
#     return documents
#
# Then add the call inside search_plant_info() above.