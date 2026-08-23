import hashlib
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Configuration ──────────────────────────────────────────────────────────
from pathlib import Path

from dotenv import load_dotenv
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "config" / ".env")

DATASET_DIR = str(_ROOT / "data" / "raw_pdfs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
# M0: tunda initial scan agar chatbot warm-up dulu. See docs/decisions.md#m0
STARTUP_GRACE_SECONDS = int(os.getenv("STARTUP_GRACE_SECONDS", "180"))
BM25_CACHE_PATH = os.getenv("BM25_CACHE_PATH", str(_ROOT / "data" / "db" / "bm25_cache.pkl"))

# ─── ChromaDB server connection ──────────────────────────────────────────────
import chromadb as _chromadb
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
_chroma_client = _chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# ─── Initialization ─────────────────────────────────────────────────────────
embeddings   = OllamaEmbeddings(model=EMBED_MODEL)
vector_store = Chroma(
    collection_name="main_dataset",  # intentionally shared with vectorCSV
    client=_chroma_client,
    embedding_function=embeddings
)

# ─── Chunk strategy per PDF type ─────────────────────────────────────────────
# PDF regulasi/kebijakan (panjang, butuh konteks luas) → 800/100
# PDF jurnal/penelitian (paragraf padat, satu ide per chunk) → 600/80
# PDF harga/tabel (baris pendek, perlu exact match) → 250/25
# Default (panduan budidaya, umum) → 500/50
#
# Deteksi otomatis berdasarkan nama file.
# Tambah kata kunci di _PDF_TYPE_RULES untuk menyesuaikan.

_PDF_TYPE_RULES = {
    # keyword di nama file → (chunk_size, chunk_overlap, label)
    "regulasi":   (800, 100, "regulasi"),
    "permentan":  (800, 100, "regulasi"),
    "kebijakan":  (800, 100, "regulasi"),
    "undang":     (800, 100, "regulasi"),
    "peraturan":  (800, 100, "regulasi"),
    "jurnal":     (600,  80, "jurnal"),
    "penelitian": (600,  80, "jurnal"),
    "laporan":    (600,  80, "jurnal"),
    "harga":      (250,  25, "harga"),
    "price":      (250,  25, "harga"),
    "komoditas":  (250,  25, "harga"),
}
_PDF_TYPE_DEFAULT = (500, 50, "default")


def _get_splitter(file_name: str) -> tuple:
    """
    Pilih chunk_size dan overlap berdasarkan nama file.
    Return: (splitter, type_label)
    """
    name_lower = file_name.lower()
    for keyword, (size, overlap, label) in _PDF_TYPE_RULES.items():
        if keyword in name_lower:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                add_start_index=True,
            )
            return splitter, label
    # Default
    size, overlap, label = _PDF_TYPE_DEFAULT
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    return splitter, label


# N1: naikkan versi ini setiap logika ekstraksi/metadata berubah — semua hash
# konten berubah sekali, memicu SATU reindex terkendali via mekanisme R4.
# See docs/decisions.md#n1
_METADATA_SCHEMA_VERSION = "v2"


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _file_content_hash(path: str) -> str:
    """R4 + N1 — MD5 isi file + _METADATA_SCHEMA_VERSION (per 8KB chunk)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    h.update(_METADATA_SCHEMA_VERSION.encode())
    return h.hexdigest()


def _stored_hash(existing: dict) -> str | None:
    """R4 — ambil file_hash dari hasil Chroma get() (kosong → None)."""
    metadatas = existing.get("metadatas") or []
    if metadatas and "file_hash" in metadatas[0]:
        return metadatas[0]["file_hash"]
    return None


def invalidate_bm25_cache():
    """R4 — PDF baru/diubah = korpus BM25 berubah; cache harus invalid."""
    if os.path.exists(BM25_CACHE_PATH):
        os.remove(BM25_CACHE_PATH)
        print("🗑️  BM25 cache invalidated — will rebuild on next main.py start.")


def index_file(file_path: str):
    """Index a single PDF file into the vector store."""
    file_name = os.path.basename(file_path)
    if not file_name.lower().endswith(".pdf"):
        return

    # R4: dedup by CONTENT HASH — PDF di-edit tetap ter-reindex. See docs/decisions.md#r4
    file_hash = _file_content_hash(file_path)
    existing  = vector_store.get(where={"source": file_name}, limit=1)
    if existing["ids"]:
        if _stored_hash(existing) == file_hash:
            print(f"⏭️  Unchanged (already indexed): {file_name}")
            return
        print(f"♻️  Content changed: {file_name} — replacing old chunks...")
        vector_store.delete(where={"source": file_name})

    print(f"📄 New PDF: {file_name}")
    try:
        loader              = PyPDFLoader(file_path)
        pages               = loader.load()
        splitter, pdf_type  = _get_splitter(file_name)
        chunks              = splitter.split_documents(pages)
        print(f"   Type: {pdf_type} | chunks: {len(chunks)}")

        documents = []
        ids       = []
        for chunk in chunks:
            chunk.metadata["source"]   = file_name
            chunk.metadata["pdf_type"] = pdf_type
            chunk.metadata["file_hash"] = file_hash
            content      = normalize(chunk.page_content)
            combined_str = (
                f"{content}_{file_name}"
                f"_{chunk.metadata.get('page', 0)}"
                f"_{chunk.metadata.get('start_index', 0)}"
            )
            doc_id             = hashlib.md5(combined_str.encode()).hexdigest()
            chunk.page_content = content
            documents.append(chunk)
            ids.append(doc_id)

        if documents:
            total      = len(documents)
            batch_size = 10
            print(f"📦 Indexing {file_name}: {total} chunks...")
            for i in range(0, total, batch_size):
                vector_store.add_documents(
                    documents=documents[i:i+batch_size],
                    ids=ids[i:i+batch_size]
                )
            print(f"✅ Indexed: {file_name}")
            # R4: semua path indexing (baru/changed) invalidate BM25. See docs/decisions.md#r4
            invalidate_bm25_cache()

    except Exception as e:
        err_str = str(e).lower()
        # A2: PDF terenkripsi (AES/password) bukan error generik. See docs/decisions.md#a2
        if "aes" in err_str or "encrypted" in err_str or "cryptography" in err_str:
            print(f"🔒 Skipped (encrypted, needs password): {file_name}")
        else:
            print(f"❌ Error processing {file_name}: {e}")

def index_all_existing():
    """Index any PDFs already present in the directory at startup."""
    import glob
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.pdf"))
    if not all_files:
        return
    for file_path in tqdm(all_files, desc="Initial PDF scan", leave=False):
        index_file(file_path)


# ─── Watchdog event handler ──────────────────────────────────────────────────
class PDFHandler(FileSystemEventHandler):
    def _handle(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(".pdf"):
            print(f"\n🔔 PDF event: {os.path.basename(event.src_path)}")
            time.sleep(0.5)
            index_file(event.src_path)

    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        self._handle(event)


# ─── Retriever (used by main.py when imported) ───────────────────────────────
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.7}
)

# ─── Main Server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"🚀 PDF Vector Server started. Watching '{DATASET_DIR}/'...")
    print(f"   ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")

    if STARTUP_GRACE_SECONDS > 0:
        print(f"⏳  Startup grace {STARTUP_GRACE_SECONDS}s sebelum initial scan "
              f"(hindari kontes embedding dengan chatbot)...")
        time.sleep(STARTUP_GRACE_SECONDS)

    index_all_existing()

    observer = Observer()
    observer.schedule(PDFHandler(), path=DATASET_DIR, recursive=False)
    observer.start()
    print("👀 Watchdog active — waiting for new PDFs...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Server stopped.")
    observer.join()