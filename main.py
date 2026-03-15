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
    search_plant_info,
    search_plant_rag,
    is_plant_cached,
    get_cached_plant_docs,
    plant_store,
)

from query_logger import (
    new_query_trace,
    set_docs,
    commit_trace,
    print_debug_report,
    flagged_logs,
    session_logs,
)
from eval_loop import evaluate


# ─────────────────────────────────────────────
# Models & Paths
# ─────────────────────────────────────────────
EMBED_MODEL       = "mxbai-embed-large"
# ─── ChromaDB server connection ──────────────────────────────────────────────
# Ganti host/port sesuai setup kamu.
# Default: server jalan di mesin yang sama (localhost:8000)
# Remote server: ganti "localhost" dengan IP/hostname server
import chromadb as _chromadb
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
_chroma_client = _chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
BM25_CACHE_PATH   = "bm25_cache.pkl"
SCOPE_CONFIG_PATH   = "word_config/scope_config.ini"

model         = OllamaLLM(model="sugi-v0.1L", temperature=0.3, repeat_penalty=1.15)
rewrite_model = OllamaLLM(model="phi3", temperature=0)

embeddings = OllamaEmbeddings(model=EMBED_MODEL)


# ─────────────────────────────────────────────
# Conversation Memory  — dengan TTL purge
# ─────────────────────────────────────────────
# Sesi lama bisa mencemari konteks kalau dibiarkan terus.
# TTL_DAYS = batas umur dokumen memory. Dokumen lebih tua dari ini
# akan dihapus saat startup dan setiap kali sesi baru disimpan.

MEMORY_TTL_DAYS = 14   # ganti sesuai kebutuhan (0 = tidak ada TTL)

memory_store = Chroma(
    collection_name="conversation_memory",
    client=_chroma_client,
    embedding_function=embeddings,
)


def purge_expired_memories(ttl_days: int = MEMORY_TTL_DAYS) -> int:
    """
    Hapus entri memory yang lebih tua dari ttl_days.
    Return jumlah dokumen yang dihapus.
    Dipanggil saat startup dan setiap kali menyimpan sesi baru.
    """
    if ttl_days <= 0:
        return 0
    try:
        all_docs = memory_store.get(include=["metadatas"])
        if not all_docs["ids"]:
            return 0

        cutoff    = datetime.now().timestamp() - (ttl_days * 86400)
        to_delete = []
        for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"]):
            ts_str = meta.get("timestamp", "")
            if not ts_str:
                continue
            try:
                doc_ts = datetime.fromisoformat(ts_str).timestamp()
                if doc_ts < cutoff:
                    to_delete.append(doc_id)
            except ValueError:
                continue

        if to_delete:
            memory_store.delete(ids=to_delete)
            print(f"🗑️  Memory TTL: removed {len(to_delete)} expired docs "
                  f"(older than {ttl_days} days).")
        return len(to_delete)
    except Exception as e:
        print(f"⚠️  Memory TTL purge error: {e}")
        return 0


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
    # Purge expired setiap kali simpan — ringan karena hanya iterasi metadata
    purge_expired_memories()


# Purge expired memory entries at startup
_startup_purged = purge_expired_memories()


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

    print("🔨 Building BM25 index (first run)...")
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
# Regex helper — dipakai oleh semua fungsi deteksi di bawah
# ─────────────────────────────────────────────

def _word_match(pattern: str, text: str) -> bool:
    """Whole-word regex match, case-insensitive. Mencegah 'hi' cocok di 'bagaimana'."""
    escaped = re.escape(pattern)
    regex   = r"(?<![a-zA-Z])" + escaped + r"(?![a-zA-Z])"
    return bool(re.search(regex, text, re.IGNORECASE))


# ─────────────────────────────────────────────
# Query Rewriting — rule-based (config dari rewriter_config.ini)
# ─────────────────────────────────────────────
# Semua data (referential_words, followup_patterns, topic_keywords) dimuat
# dari rewriter_config.ini. Edit INI, restart main.py — tidak perlu ubah kode.

REWRITER_CONFIG_PATH = "word_config/rewriter_config.ini"
PLANT_CONFIG_PATH    = "word_config/plant_keywords.ini"


def _load_rewriter_config(path: str) -> tuple[set, set, list, dict]:
    """
    Muat konfigurasi rewriter dari INI file.
    Return: (referential_words, referential_suffixes, followup_patterns, topic_keywords)
    """
    cfg = _cp.ConfigParser(
        comment_prefixes=(";", "#"),
        inline_comment_prefixes=(";",),
    )
    cfg.read(path, encoding="utf-8")

    def _lines(section: str, key: str) -> list[str]:
        """Ambil nilai multiline sebagai list baris non-kosong."""
        raw = cfg.get(section, key, fallback="")
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # [referential_words]
    ref_words = set(_lines("referential_words", "words"))

    # [referential_suffixes]
    ref_suffixes = set(_lines("referential_suffixes", "suffixes"))

    # [followup_patterns] — setiap baris adalah regex pattern
    followup_pats = _lines("followup_patterns", "patterns")

    # [topic_keywords] — semua key=value di section ini
    topic_kw: dict[str, str] = {}
    if cfg.has_section("topic_keywords"):
        for key, val in cfg.items("topic_keywords"):
            # configparser lowercases keys by default — itu yang kita mau
            topic_kw[key.strip()] = val.strip()

    return ref_words, ref_suffixes, followup_pats, topic_kw


# Load saat startup — reload otomatis tidak diperlukan (restart utk perubahan)
_REFERENTIAL_WORDS, _REFERENTIAL_SUFFIXES, _IMPLICIT_FOLLOWUP_PATTERNS, _TOPIC_KEYWORDS = \
    _load_rewriter_config(REWRITER_CONFIG_PATH)

print(f"📝 Rewriter config loaded: "
      f"{len(_REFERENTIAL_WORDS)} ref-words, "
      f"{len(_REFERENTIAL_SUFFIXES)} suffixes, "
      f"{len(_IMPLICIT_FOLLOWUP_PATTERNS)} patterns, "
      f"{len(_TOPIC_KEYWORDS)} topics.")


def _has_referential(question: str) -> bool:
    """True kalau pertanyaan mengandung kata/sufiks yang merujuk ke konteks sebelumnya."""
    q = question.lower().strip()
    for word in _REFERENTIAL_WORDS:
        if _word_match(word, q):
            return True
    # Cek sufiks referensial (merawatnya, harganya, dll.)
    for w in q.split():
        for suf in _REFERENTIAL_SUFFIXES:
            clean_suf = suf.rstrip("?.").strip()
            if w.endswith(clean_suf) and len(w) > len(clean_suf) + 2:
                return True
    return False


def _is_implicit_followup(question: str) -> bool:
    """True kalau pertanyaan adalah followup implisit yang butuh konteks history."""
    q = question.lower().strip()
    for pattern in _IMPLICIT_FOLLOWUP_PATTERNS:
        try:
            if re.match(pattern, q):
                return True
        except re.error:
            continue   # skip pattern yang invalid di INI
    return False


def _extract_last_subject(history_text: str) -> str:
    """
    Ekstrak subjek utama dari pertanyaan terakhir user di history.
    Prioritas: nama tanaman spesifik > topik non-tanaman > string kosong.
    """
    lines = history_text.strip().splitlines()
    last_user_line = ""
    for line in reversed(lines):
        if line.startswith("User:"):
            last_user_line = line[5:].strip().lower()
            break
    if not last_user_line:
        return ""

    # Prioritas 1: nama tanaman dari _PLANT_NAME_MAP (terpanjang dulu)
    for keyword in sorted(_PLANT_NAME_MAP.keys(), key=len, reverse=True):
        if _word_match(keyword, last_user_line):
            return keyword

    # Prioritas 2: topik non-tanaman dari INI
    for keyword, label in sorted(_TOPIC_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if _word_match(keyword, last_user_line):
            return label

    return ""


def _resolve_suffix_referential(question: str, subject: str) -> str:
    """
    Ganti sufiks referensial seperti "merawatnya" -> "merawat [subject]".
    Lebih halus dari ganti kata penuh.
    """
    result = question
    for suf in ["nya"]:
        pattern = r"(\w{3,})" + re.escape(suf) + r"\b"
        def replacer(m, _subj=subject):
            return f"{m.group(1)} {_subj}"
        new = re.sub(pattern, replacer, result, flags=re.IGNORECASE)
        if new != result:
            result = new
            break
    return result


def maybe_rewrite(question: str, history_text: str) -> str:
    """
    Rule-based query rewriting — config dari rewriter_config.ini.

    Logika:
    1. Tidak ada history → return apa adanya
    2. Ada referential word/sufiks → ekstrak subjek dari history, ganti
    3. Ada followup implisit → append subjek ke akhir pertanyaan
    4. Tidak ada sinyal apapun → return apa adanya (sudah standalone)
    """
    if history_text == "No history.":
        return question

    has_ref     = _has_referential(question)
    is_followup = _is_implicit_followup(question)

    if not has_ref and not is_followup:
        return question

    subject = _extract_last_subject(history_text)
    if not subject:
        return question

    q = question

    if has_ref:
        # Ganti referential word penuh (tersebut, itu, ini) — terpanjang dulu
        replaced = False
        for ref in sorted(_REFERENTIAL_WORDS, key=len, reverse=True):
            if _word_match(ref, q.lower()):
                q = re.sub(
                    r"(?<![a-zA-Z])" + re.escape(ref) + r"(?![a-zA-Z])",
                    subject,
                    q,
                    count=1,
                    flags=re.IGNORECASE,
                )
                replaced = True
                break
        if not replaced:
            q = _resolve_suffix_referential(q, subject)

    elif is_followup:
        # Followup implisit: append subjek kalau belum ada di pertanyaan
        all_known = list(_PLANT_NAME_MAP.keys()) + list(_TOPIC_KEYWORDS.keys())
        subject_in_q = any(_word_match(kw, q.lower()) for kw in all_known)
        if not subject_in_q:
            q = q.rstrip("?").rstrip() + f" {subject}?"

    if q != question:
        print(f"   [rewrite] '{question}' -> '{q}'")
    return q


# Plant Query Detection — loaded from word_config/plant_keywords.ini
# ─────────────────────────────────────────────


def _load_plant_config(path: str) -> tuple[set, set, dict]:
    """
    Muat PLANT_KEYWORDS_STRONG, PLANT_KEYWORDS_WEAK, dan _PLANT_NAME_MAP
    dari word_config/plant_keywords.ini.
    Return: (keywords_strong, keywords_weak, name_map)

    strict=False: duplicate keys tidak crash — nilai terakhir menang.
    Kalau file gagal dibaca sepenuhnya, print peringatan dan return set/dict kosong.
    """
    cfg = _cp.ConfigParser(
        comment_prefixes=(";", "#"),
        inline_comment_prefixes=(";",),
        strict=False,   # duplicate keys → last value wins, tidak crash
    )
    try:
        read_ok = cfg.read(path, encoding="utf-8")
        if not read_ok:
            print(f"⚠️  Plant config not found: {path}")
            return set(), set(), {}
    except _cp.Error as e:
        print(f"⚠️  Plant config parse error: {e}")
        return set(), set(), {}

    def _lines(section: str, key: str) -> list[str]:
        raw = cfg.get(section, key, fallback="")
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    strong = set(_lines("keywords_strong", "keywords"))
    weak   = set(_lines("keywords_weak",   "keywords"))

    name_map: dict[str, str] = {}
    if cfg.has_section("plant_name_map"):
        for key, val in cfg.items("plant_name_map"):
            name_map[key.strip()] = val.strip()

    return strong, weak, name_map


PLANT_KEYWORDS_STRONG, PLANT_KEYWORDS_WEAK, _PLANT_NAME_MAP =     _load_plant_config(PLANT_CONFIG_PATH)

print(f"🌿 Plant config loaded: "
      f"{len(PLANT_KEYWORDS_STRONG)} strong, "
      f"{len(PLANT_KEYWORDS_WEAK)} weak, "
      f"{len(_PLANT_NAME_MAP)} name mappings.")


def is_plant_query(original_question: str, rewritten_query: str) -> bool:
    orig = original_question.lower()
    rew  = rewritten_query.lower()

    for kw in PLANT_KEYWORDS_STRONG:
        if _word_match(kw, rew) or _word_match(kw, orig):
            print(f"   [plant: strong '{kw}' -> PLANT]")
            return True

    has_weak = any(_word_match(kw, orig) for kw in PLANT_KEYWORDS_WEAK)
    if has_weak:
        has_strong = any(
            _word_match(kw, orig) or _word_match(kw, rew)
            for kw in PLANT_KEYWORDS_STRONG
        )
        if has_strong:
            print(f"   [plant: weak+strong match -> PLANT]")
            return True
        print(f"   [plant: weak only, no specific plant -> SKIP plant API]")

    return False


# ─────────────────────────────────────────────
# Plant Name Extraction
# ─────────────────────────────────────────────
# _PLANT_NAME_MAP dimuat dari word_config/plant_keywords.ini
# lewat _load_plant_config() di atas. Tidak ada data hardcoded di sini.
# Untuk menambah tanaman baru: edit [plant_name_map] di plant_keywords.ini

_PLANT_FALLBACK_TEMPLATE = (
    "You are a botanical assistant. Extract the English plant name.\n"
    "Return ONLY the name (1-3 words), or NONE if no specific plant.\n\n"
    "Question: {question}\n"
    "English plant name (or NONE):"
)
_plant_fallback_prompt = ChatPromptTemplate.from_template(_PLANT_FALLBACK_TEMPLATE)
_plant_fallback_chain  = _plant_fallback_prompt | rewrite_model | StrOutputParser()


def extract_and_translate_plant(question: str) -> str | None:
    """
    Layer 1: direct map _PLANT_NAME_MAP (deterministik).
             Frasa lebih panjang dicek dulu (kelapa sawit sebelum kelapa).
    Layer 2: phi3 fallback hanya kalau tidak ada di map.
    """
    q = question.lower()

    for keyword in sorted(_PLANT_NAME_MAP.keys(), key=len, reverse=True):
        if _word_match(keyword, q):
            mapped = _PLANT_NAME_MAP[keyword]
            print(f"   [plant-map] '{keyword}' -> '{mapped}'")
            return mapped

    print("   [plant-map] no direct match -- trying phi3 fallback...")
    try:
        result     = _plant_fallback_chain.invoke({"question": question})
        plant_name = result.strip().split("\n")[0].lower().strip()
        if (
            plant_name == "none"
            or not plant_name
            or len(plant_name.split()) > 4
            or any(w in plant_name for w in ["question", "answer", "->", ":"])
        ):
            print(f"   [plant-map] phi3: no valid plant (got: '{plant_name}')")
            return None
        print(f"   [plant-map] phi3: '{plant_name}'")
        return plant_name
    except Exception as e:
        print(f"   [plant-map] phi3 error: {e}")
        return None


# Weather Detection
# ─────────────────────────────────────────────
WEATHER_KEYWORDS = {
    "cuaca", "iklim", "hujan", "prakiraan cuaca", "curah hujan",
    "suhu", "suhu udara", "temperatur", "kelembaban", "angin",
    "kecepatan angin", "arah angin", "tekanan udara", "awan",
    "tutupan awan", "evapotranspirasi", "embun", "titik embun",
    "musim", "musim hujan", "musim kemarau", "kekeringan", "banjir",
    "el nino", "la nina", "perubahan iklim", "panas", "dingin",
    "kabut", "badai", "petir", "gerimis",
    "suhu tanah", "kelembaban tanah", "kadar air tanah",
    "prakiraan", "prediksi cuaca", "kondisi cuaca",
    "weather", "temperature", "humidity", "rainfall", "precipitation",
    "wind speed", "wind direction", "forecast", "climate",
    "cloud cover", "evapotranspiration", "dew point", "pressure",
    "soil temperature", "soil moisture", "drought", "flood",
    "storm", "sunny", "cloudy", "rain forecast",
}


# Frasa yang menunjukkan user butuh konteks cuaca meski tidak menyebut
# kata cuaca secara eksplisit — "cocok menanam", "waktu tanam", dll.
_PLANTING_SUITABILITY_PHRASES = {
    "cocok untuk menanam", "cocok menanam", "waktu tanam",
    "saat tanam", "musim tanam", "kapan tanam", "kapan menanam",
    "apakah cocok", "apakah bagus", "waktu yang tepat",
    "suitable for planting", "good time to plant", "right time to plant",
    "cocok ditanam", "baik untuk menanam", "ideal untuk menanam",
}


def _is_weather_query(question: str) -> bool:
    q = question.lower()
    # Cek keyword cuaca eksplisit
    if any(_word_match(kw, q) for kw in WEATHER_KEYWORDS):
        return True
    # Cek frasa kesesuaian tanam — implisit butuh data cuaca
    for phrase in _PLANTING_SUITABILITY_PHRASES:
        if phrase in q:
            print(f"   [weather: planting suitability '{phrase}' -> WEATHER]")
            return True
    return False


# ─────────────────────────────────────────────
# Scope Guard
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


def _is_greeting(question: str) -> bool:
    """True kalau query adalah salam murni tanpa konten substantif."""
    q = question.lower().strip()
    for p in GREETING_PATTERNS:
        if _word_match(p, q) and len(q.split()) <= 6:
            return True
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
# Tanggal/waktu TIDAK pakai template variable {now_*} karena bisa bentrok
# dengan kurung kurawal dalam {data} (dokumen dari ChromaDB).
# Solusi: _inject_date() mengganti placeholder string biasa sebelum LangChain
# mem-parse template — sehingga tidak ada collision apapun.

_ANSWER_TEMPLATE_BASE = (
    "Informasi sesi saat ini:\n"
    "- Tanggal  : [[NOW_DATE]]\n"
    "- Hari     : [[NOW_DAY]]\n"
    "- Jam      : [[NOW_TIME]] WIB\n"
    "- Lokasi   : Jakarta, Indonesia\n\n"
    "Data relevan dari database (pertanian, cuaca, tanaman):\n"
    "{data}\n\n"
    "PENTING:\n"
    "- Gunakan tanggal, hari, dan jam di atas kalau pertanyaan menyebut "
    "\"hari ini\", \"sekarang\", \"saat ini\", atau \"today\".\n"
    "- Jika data berisi informasi cuaca (suhu, hujan, kelembaban, angin, dll.), "
    "WAJIB gunakan data tersebut untuk menjawab.\n"
    "- Jika data berisi informasi tanaman atau pertanian, prioritaskan data tersebut.\n"
    "- Jawab secara ringkas dan jelas, tidak mengulang kalimat yang sama.\n"
    "- Akhiri jawaban setelah selesai menjelaskan.\n\n"
    "Pertanyaan: {question}"
)


def _get_answer_prompt() -> ChatPromptTemplate:
    """
    Buat ChatPromptTemplate segar tiap query dengan tanggal/waktu aktual
    sudah disisipkan. [[NOW_*]] adalah placeholder biasa (bukan {var} LangChain)
    sehingga tidak konflik dengan kurung kurawal di dalam dokumen.
    """
    now   = __import__("datetime").datetime.now()
    hari  = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][now.weekday()]
    bulan = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember"
    ][now.month - 1]
    filled = (
        _ANSWER_TEMPLATE_BASE
        .replace("[[NOW_DATE]]", f"{now.day} {bulan} {now.year}")
        .replace("[[NOW_DAY]]",  hari)
        .replace("[[NOW_TIME]]", now.strftime("%H:%M"))
    )
    return ChatPromptTemplate.from_template(filled)
# Debug command handler
# ─────────────────────────────────────────────
def handle_debug_command(command: str, session_id: str) -> bool:
    cmd = command.strip().lower()
    if cmd == "!debug":
        print_debug_report(n=10)
        return True
    if cmd == "!flags":
        entries = flagged_logs()
        if not entries:
            print("✅ Tidak ada query yang di-flag.")
        else:
            print(f"\n🚩 {len(entries)} query di-flag:\n")
            for e in entries:
                print(f"  [{e['query_id']}] {e['question'][:60]}")
                print(f"   → {e['eval'].get('reason','')}")
                print(f"   → docs: {[d['source'] for d in e.get('docs_retrieved',[])]}")
                print()
        return True
    if cmd == "!session":
        entries = session_logs(session_id)
        print(f"\n📋 Log sesi {session_id}: {len(entries)} query")
        for e in entries:
            flag = "🚩" if e.get("eval", {}).get("flag") else "✅"
            print(f"  {flag} [{e['query_id']}] {e['question'][:55]} — {e.get('latency_ms',0)}ms")
        print()
        return True
    return False


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
print(f"\n🌾 Sugi v0.1L siap melayani! (session: {session_id})")
print("   Tips debug: ketik !debug / !flags / !session\n")

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

    if question.startswith("!"):
        handle_debug_command(question, session_id)
        continue

    trace             = new_query_trace(session_id)
    trace["question"] = question
    full_response     = ""
    error_msg         = None
    is_greeting_query = _is_greeting(question)

    try:
        # ── Scope check ───────────────────────────────────────────────────────
        print("🛡️  Checking scope...")
        in_scope = is_in_scope(question)
        trace["scope_passed"] = in_scope

        if not in_scope:
            print(f"🚫 Out of scope.\n\n{REFUSAL_MESSAGE}\n")
            trace["answer_preview"] = REFUSAL_MESSAGE[:200]
            commit_trace(trace)
            continue

        print("✅ Approved.")

        history_text = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]]
        ) or "No history."
        has_history = history_text != "No history."

        # ── [FIX 1] Query rewriting dengan validasi ───────────────────────────
        standalone_query   = maybe_rewrite(question, history_text)
        trace["rewritten"] = standalone_query
        print(f"🔍 Query: {standalone_query}")

        # ── [FIX 2] Deteksi dari original question, bukan rewritten ──────────
        include_plant   = is_plant_query(question, standalone_query)
        include_weather = _is_weather_query(standalone_query) or _is_weather_query(question)

        trace["flags"] = {
            "is_plant":    include_plant,
            "is_weather":  include_weather,
            "has_history": has_history,
            "is_greeting": is_greeting_query,
        }

        plant_query_en = None
        if include_plant:
            plant_query_en = extract_and_translate_plant(question)   # dari original

            if plant_query_en is None:
                print("🌿 No valid plant name — skipping plant API.")
                include_plant = False
            elif is_plant_cached(plant_query_en):
                cached_docs = get_cached_plant_docs(plant_query_en, k=5)
                print(f"🌿 Local cache hit: {len(cached_docs)} docs for '{plant_query_en}'.")
            else:
                print(f"🌿 Fetching '{plant_query_en}' from Perenual API...")
                fetched = search_plant_info(plant_query_en)
                if fetched:
                    print(f"   ✅ Fetched and stored {len(fetched)} plant docs.")
                else:
                    print(f"   ⚠️  No data returned for '{plant_query_en}'.")
                    include_plant = False

        if include_weather:
            print("🌤️  Weather query detected — using local weather_store.")

        # ── Retrieval ─────────────────────────────────────────────────────────
        print("🗂️  Retrieving and reranking documents...")

        weather_context_docs = []
        if include_weather:
            weather_context_docs = weather_store.similarity_search(standalone_query, k=8)
            print(f"🌤️  Weather docs retrieved: {len(weather_context_docs)}")

        retriever_obj = build_retriever(has_history, include_plant, include_weather=False)
        docs          = retriever_obj.invoke(standalone_query)

        all_docs = weather_context_docs + [d for d in docs if d not in weather_context_docs]
        context  = format_docs(all_docs) if all_docs else "Tidak ada data relevan di database."
        print(f"📊 Found {len(all_docs)} relevant chunks "
              f"({len(weather_context_docs)} weather + {len(docs)} RAG).")

        set_docs(trace, all_docs)

        # ── Generate answer ───────────────────────────────────────────────────
        print("🤖 Generating answer:\n")
        _live_chain = _get_answer_prompt() | model
        for chunk in _live_chain.stream({"data": context, "question": standalone_query}):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n")

        trace["answer_preview"] = full_response[:200]
        chat_history.append((question, full_response))

        # ── [FIX 3] Eval — skip untuk greeting ───────────────────────────────
        print("🔬 Running eval...")
        if is_greeting_query:
            eval_result = {
                "faithfulness": "SKIP",
                "relevance":    "SKIP",
                "flag":         False,
                "reason":       "greeting — eval skipped",
                "method":       "skip",
                "doc_count":    len(all_docs),
            }
            print("   ⏭️  Greeting — eval skipped.")
        else:
            eval_result = evaluate(
                question = standalone_query,
                docs     = all_docs,
                context  = context,
                answer   = full_response,
                use_llm  = True,
            )
        trace["eval"] = eval_result

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {e}")

    finally:
        commit_trace(trace, error=error_msg)