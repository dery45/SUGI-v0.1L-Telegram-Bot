import hashlib
import pickle
import os
import re
from datetime import datetime
import configparser as _cp

from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from vectorCSV import vector_store
from vectorWeather import weather_store, weather_retriever
from plant_api import (
    search_plant_info,      # fetch from API + store in ChromaDB
    search_plant_rag,       # semantic search against plant ChromaDB
    is_plant_cached,        # exact cache_key check — no false positives
    get_cached_plant_docs,  # retrieve already-cached docs by plant name
    plant_store,
)


# ─────────────────────────────────────────────
# Models & Paths
# ─────────────────────────────────────────────
EMBED_MODEL       = "mxbai-embed-large"
DB_PATH           = "chrome_longchain_db"
BM25_CACHE_PATH   = "bm25_cache.pkl"
SCOPE_CONFIG_PATH = "scope_config.ini"

model         = OllamaLLM(model="sugi-v0.1L", temperature=0.3, repeat_penalty=1.15)
rewrite_model = OllamaLLM(model="phi3", temperature=0)
# ^ change to "sugi-v0.1L" if phi3 is not installed

embeddings = OllamaEmbeddings(model=EMBED_MODEL)


# ─────────────────────────────────────────────
# Conversation Memory
# ─────────────────────────────────────────────
memory_store = Chroma(
    collection_name="conversation_memory",
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

def save_summary_to_memory(summary: str, session_id: str):
    content = summary.strip()
    if not content:
        return
    doc_id   = hashlib.md5(f"{session_id}_{content}".encode()).hexdigest()
    existing = memory_store.get(ids=[doc_id])
    if existing["ids"]:
        return
    doc = Document(
        page_content=content,
        metadata={
            "source":     "conversation_memory",
            "session_id": session_id,
            "timestamp":  datetime.now().isoformat(),
        },
        id=doc_id,
    )
    memory_store.add_documents(documents=[doc], ids=[doc_id])
    print(f"💾 Session summary saved (id: {doc_id[:8]}...)")


# ─────────────────────────────────────────────
# BM25 Cache
# ─────────────────────────────────────────────
def load_or_build_bm25() -> BM25Retriever:
    if os.path.exists(BM25_CACHE_PATH):
        print("📂 Loading BM25 from cache...")
        with open(BM25_CACHE_PATH, "rb") as f:
            retriever = pickle.load(f)
        print("✅ BM25 loaded from cache.")
        return retriever

    print("🔨 Building BM25 index (first run — may take a moment)...")
    all_docs = vector_store.get(include=["documents", "metadatas"])
    docs_for_bm25 = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    retriever   = BM25Retriever.from_documents(docs_for_bm25)
    retriever.k = 4
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(retriever, f)
    print(f"✅ BM25 index built ({len(docs_for_bm25):,} docs) and cached.")
    return retriever


# ─────────────────────────────────────────────
# Retrievers
# ─────────────────────────────────────────────
print("🔄 Initializing Retrievers...")
bm25_retriever   = load_or_build_bm25()
bm25_retriever.k = 4
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

primary_ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

memory_retriever  = memory_store.as_retriever(search_kwargs={"k": 2})
plant_retriever   = plant_store.as_retriever(search_kwargs={"k": 3})
weather_retriever = weather_store.as_retriever(search_kwargs={"k": 8})


# ─────────────────────────────────────────────
# Reranker
# ─────────────────────────────────────────────
print("🧠 Loading Reranker Model...")
reranker_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)


def build_retriever(
    has_history:     bool,
    include_plant:   bool = False,
    include_weather: bool = False,
) -> ContextualCompressionRetriever:
    """
    Dynamically compose retriever based on query type.
    Weights are normalised so they always sum to 1.0.
    """
    retrievers = [primary_ensemble]
    weights    = [0.60]

    if has_history:
        retrievers.append(memory_retriever)
        weights.append(0.10)
    if include_plant:
        retrievers.append(plant_retriever)
        weights.append(0.15)
    if include_weather:
        retrievers.append(weather_retriever)
        weights.append(0.15)

    total   = sum(weights)
    weights = [w / total for w in weights]

    base = (
        EnsembleRetriever(retrievers=retrievers, weights=weights)
        if len(retrievers) > 1
        else primary_ensemble
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base,
    )


# ─────────────────────────────────────────────
# Query Rewriting
# ─────────────────────────────────────────────
rewrite_template = """Rephrase the following question as a standalone question based on the history.
If no history is needed, return the original question as-is.
DO NOT answer. Return ONLY the rephrased question.
History: {history}
Question: {question}
Standalone Question:"""
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
rewrite_chain  = rewrite_prompt | rewrite_model | StrOutputParser()

def maybe_rewrite(question: str, history_text: str) -> str:
    if history_text == "No history.":
        return question
    print("✍️  Rewriting query...")
    result = rewrite_chain.invoke({"history": history_text, "question": question})
    return result.strip().split("\n")[0]


# ─────────────────────────────────────────────
# Plant Name Extraction + Translation
# ─────────────────────────────────────────────
PLANT_EXTRACT_TEMPLATE = """
You are a botanical & agricultural assistant.

Task:
1. Identify the main PLANT SPECIES the user is asking about.
2. Return ONLY the most precise English scientific name (genus + species) or the most common agricultural / food crop English name.
3. If the user says "cokelat", "cocoa", "coklat", "chocolate plant", "tanaman cokelat" → ALWAYS return "cacao" or "Theobroma cacao"
4. NEVER return just "chocolate" — that refers to many ornamental cultivars, not the real cocoa tree.
5. Do NOT add "plant", "tree", "tanaman", explanations, or punctuation.

Examples:
Question: tanaman cokelat        → Answer: cacao
Question: pohon coklat          → Answer: cacao
Question: tanaman sawo          → Answer: sapodilla
Question: penyakit tanaman tomat → Answer: tomato
Question: chocolate mint care   → Answer: chocolate mint

Question:
{question}

Answer:
"""
plant_extract_prompt = ChatPromptTemplate.from_template(PLANT_EXTRACT_TEMPLATE)
plant_extract_chain  = plant_extract_prompt | rewrite_model | StrOutputParser()

def extract_and_translate_plant(question: str) -> str:
    try:
        print("🌐 Extracting & translating plant name...")
        result     = plant_extract_chain.invoke({"question": question})
        plant_name = result.strip().split("\n")[0].lower().strip()
        print(f"🌿 Plant extracted → '{plant_name}'")
        return plant_name
    except Exception as e:
        print(f"⚠️  Plant extraction failed: {e}")
        return question.lower().strip()


# ─────────────────────────────────────────────
# Plant Query Detection
# ─────────────────────────────────────────────
PLANT_API_KEYWORDS = {
    "tanaman", "tumbuhan", "pohon", "bunga", "buah", "daun", "akar",
    "benih", "bibit", "budidaya", "perawatan tanaman", "cara menanam",
    "pupuk", "penyiraman", "penyakit tanaman", "hama tanaman",
    "pestisida", "fungisida", "herbisida", "varietas", "jenis tanaman",
    "species", "plant", "flower", "tree", "herb", "shrub", "vegetable",
    "fruit tree", "ornamental", "indoor plant", "outdoor plant",
    "watering", "sunlight", "soil", "care guide", "propagation",
    "pest", "disease", "fungal", "bacterial",
}

# Weather-specific keywords that activate the weather retriever
WEATHER_KEYWORDS = {
    # Indonesian
    "cuaca", "iklim", "hujan", "prakiraan cuaca", "curah hujan",
    "suhu", "suhu udara", "temperatur", "kelembaban", "angin",
    "kecepatan angin", "arah angin", "tekanan udara", "awan",
    "tutupan awan", "evapotranspirasi", "embun", "titik embun",
    "musim", "musim hujan", "musim kemarau", "kekeringan", "banjir",
    "el nino", "la nina", "perubahan iklim", "panas", "dingin",
    "kabut", "badai", "petir", "gerimis",
    "suhu tanah", "kelembaban tanah", "kadar air tanah",
    "prakiraan", "prediksi cuaca", "kondisi cuaca",
    # English
    "weather", "temperature", "humidity", "rainfall", "precipitation",
    "wind speed", "wind direction", "forecast", "climate",
    "cloud cover", "evapotranspiration", "dew point", "pressure",
    "soil temperature", "soil moisture", "drought", "flood",
    "storm", "sunny", "cloudy", "rain forecast",
}

def _is_weather_query(question: str) -> bool:
    """True if the question likely needs weather data from Open-Meteo."""
    q = question.lower()
    return any(_word_match(kw, q) for kw in WEATHER_KEYWORDS)

def _word_match(pattern: str, text: str) -> bool:
    """Whole-word regex match — prevents 'hi' matching inside 'bagahimana'."""
    escaped = re.escape(pattern)
    regex   = r"(?<![\w])" + escaped + r"(?![\w])"
    return bool(re.search(regex, text))

def is_plant_query(question: str) -> bool:
    q = question.lower()
    return any(_word_match(kw, q) for kw in PLANT_API_KEYWORDS)


# ─────────────────────────────────────────────
# Scope Guard (loaded from scope_config.ini)
# ─────────────────────────────────────────────
def _load_scope_config(path: str):
    cfg = _cp.ConfigParser()
    cfg.read(path, encoding="utf-8")

    def _parse_list(section, key) -> set:
        raw = cfg.get(section, key, fallback="")
        return {line.strip().lower() for line in raw.splitlines() if line.strip()}

    greetings = _parse_list("greetings", "patterns")
    allowed   = set()
    for section in cfg.sections():
        if section.startswith("allowed_"):
            allowed |= _parse_list(section, "keywords")
    blocked = _parse_list("blocked_topics", "keywords") if cfg.has_section("blocked_topics") else set()
    refusal = cfg.get("refusal", "message", fallback=(
        "Maaf, pertanyaan tersebut berada di luar bidang keahlian Sugi. "
        "Apakah ada pertanyaan seputar pertanian yang bisa Sugi bantu? 🌾"
    ))
    return greetings, allowed, blocked, refusal

GREETING_PATTERNS, ALLOWED_KEYWORDS, BLOCKED_KEYWORDS, REFUSAL_MESSAGE = _load_scope_config(SCOPE_CONFIG_PATH)
print(f"📋 Scope config loaded: {len(GREETING_PATTERNS)} greetings, "
      f"{len(ALLOWED_KEYWORDS)} allowed, {len(BLOCKED_KEYWORDS)} blocked.")

def is_in_scope(question: str) -> bool:
    q = question.lower().strip()
    for p in GREETING_PATTERNS:
        if _word_match(p, q):
            print(f"   [scope: greeting '{p}' → ALLOWED]")
            return True
    for kw in ALLOWED_KEYWORDS:
        if _word_match(kw, q):
            print(f"   [scope: '{kw}' → ALLOWED]")
            return True
    print(f"   [scope: no match → BLOCKED]")
    return False


# ─────────────────────────────────────────────
# Session Summarizer
# ─────────────────────────────────────────────
SUMMARIZE_TEMPLATE = """Kamu adalah asisten yang merangkum percakapan pertanian.
Berikut adalah percakapan antara pengguna dan Sugi (asisten pertanian):
{conversation}
Tugasmu:
1. Ekstrak HANYA fakta-fakta penting tentang pertanian yang dibahas
2. Tulis dalam format poin-poin singkat (bullet points)
3. Fokus pada: tanaman, teknik budidaya, hama/penyakit, harga, lokasi, dan rekomendasi spesifik
4. ABAIKAN basa-basi, sapaan, dan pertanyaan yang tidak terjawab
5. Maksimal 10 poin, setiap poin maksimal 2 kalimat
6. Tulis dalam Bahasa Indonesia
7. Jika tidak ada fakta penting, tulis: TIDAK ADA INFORMASI PENTING
Rangkuman Fakta Sesi:"""
summarize_prompt = ChatPromptTemplate.from_template(SUMMARIZE_TEMPLATE)
summarize_chain  = summarize_prompt | model | StrOutputParser()

def summarize_and_save_session(history: list, session_id: str):
    if not history:
        return
    conversation_text = "\n\n".join(f"User: {q}\nSugi: {a}" for q, a in history)
    print("\n📝 Summarizing session for long-term memory...")
    try:
        summary = summarize_chain.invoke({"conversation": conversation_text})
        summary = summary.strip()
        if "TIDAK ADA INFORMASI PENTING" in summary.upper():
            print("ℹ️  No important facts found — session not saved.")
            return
        print(f"\n--- Session Summary ---\n{summary}\n-----------------------")
        save_summary_to_memory(summary, session_id)
    except Exception as e:
        print(f"⚠️  Could not summarize session: {e}")


# ─────────────────────────────────────────────
# Answer Chain
# ─────────────────────────────────────────────
answer_template = """Data relevan dari database (pertanian, cuaca, tanaman):
{data}

PENTING:
- Jika data di atas berisi informasi cuaca (suhu, hujan, kelembaban, angin, dll.), WAJIB gunakan data tersebut untuk menjawab — jangan katakan tidak punya data cuaca.
- Jika data berisi informasi tanaman atau pertanian, prioritaskan data tersebut.
- Jawab secara ringkas dan jelas, tidak mengulang kalimat yang sama.
- Akhiri jawaban setelah selesai menjelaskan.

Pertanyaan: {question}"""
answer_prompt = ChatPromptTemplate.from_template(answer_template)
answer_chain  = answer_prompt | model


# ─────────────────────────────────────────────
# Chat History & Session ID
# ─────────────────────────────────────────────
chat_history = []
session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
print(f"\n🌾 Sugi v0.1L siap melayani! (session: {session_id})\n")

while True:
    print("\n-------------------------------------------")
    question = input("Tanyakan pertanyaan anda (q untuk keluar): ").strip()
    print("-------------------------------------------\n")

    if not question:
        print("⚠️  Pertanyaan tidak boleh kosong.")
        continue
    if question.lower() == "q":
        summarize_and_save_session(chat_history, session_id)
        print("Sampai jumpa! 🌾")
        break

    # Scope check
    print("🛡️  Checking scope...")
    if not is_in_scope(question):
        print(f"🚫 Out of scope.\n\n{REFUSAL_MESSAGE}\n")
        continue
    print("✅ Approved.")

    history_text = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]]
    ) or "No history."
    has_history = history_text != "No history."

    # Query rewriting
    standalone_query = maybe_rewrite(question, history_text)
    print(f"🔍 Query: {standalone_query}")

    # ── Plant data: local cache first, API only if missing ────────────────────
    include_plant  = is_plant_query(standalone_query)
    plant_query_en = standalone_query

    if include_plant:
        plant_query_en = extract_and_translate_plant(standalone_query)

        if is_plant_cached(plant_query_en):
            cached_docs = get_cached_plant_docs(plant_query_en, k=5)
            print(f"🌿 Local cache hit: {len(cached_docs)} docs for '{plant_query_en}' — no API call.")
        else:
            print(f"🌿 '{plant_query_en}' not in local cache — fetching from Perenual API...")
            fetched = search_plant_info(plant_query_en)
            if fetched:
                print(f"   ✅ Fetched and stored {len(fetched)} plant docs.")
            else:
                print(f"   ⚠️  No data returned from API for '{plant_query_en}'.")

    # ── Weather data: always from local ChromaDB (vectorWeather keeps it fresh)
    include_weather = _is_weather_query(standalone_query)
    if include_weather:
        print("🌤️  Weather query detected — using local weather_store.")

    # ── Retrieval + Rerank ────────────────────────────────────────────────────
    print("🗂️  Retrieving and reranking documents...")

    # Weather docs are fetched directly — bypassing the reranker because
    # tabular weather rows score poorly in cross-encoder semantic scoring.
    weather_context_docs = []
    if include_weather:
        weather_context_docs = weather_store.similarity_search(standalone_query, k=8)
        print(f"🌤️  Weather docs retrieved: {len(weather_context_docs)}")

    retriever = build_retriever(has_history, include_plant, include_weather=False)
    docs      = retriever.invoke(standalone_query)

    # Merge: weather docs first so they always appear in context
    all_docs = weather_context_docs + [d for d in docs if d not in weather_context_docs]
    context  = format_docs(all_docs) if all_docs else "Tidak ada data relevan di database."
    print(f"📊 Found {len(all_docs)} relevant chunks ({len(weather_context_docs)} weather + {len(docs)} RAG).")

    # ── Generate Answer ───────────────────────────────────────────────────────
    print("🤖 Generating answer:\n")
    full_response = ""
    for chunk in answer_chain.stream({"data": context, "question": standalone_query}):
        print(chunk, end="", flush=True)
        full_response += chunk
    print("\n")

    chat_history.append((question, full_response))