import hashlib
import os
import pickle
import time
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Configuration ──────────────────────────────────────────────────────────
DB_PATH        = "chrome_longchain_db"
DATASET_DIR    = "dataset"
EMBED_MODEL    = "mxbai-embed-large"
BM25_CACHE_PATH = "bm25_cache.pkl"

# Fix [5]: Only 3 metadata fields per chunk instead of 20.
# The actual data already lives in the chunk text — no need
# to duplicate every column as a metadata key.
METADATA_LIMIT = 20   # kept for backward compat but used minimally below

# ─── Initialization ─────────────────────────────────────────────────────────
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True
)

def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())

def read_csv_safe(file_path):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(file_path, dtype=str, encoding=enc).dropna(how="all")
        except (UnicodeDecodeError, Exception):
            continue
    return None

def read_xlsx_safe(file_path):
    try:
        xl     = pd.ExcelFile(file_path)
        sheets = {}
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name, dtype=str).dropna(how="all")
                if not df.empty:
                    sheets[sheet_name] = df
            except Exception as e:
                print(f"  ⚠️ Skipping sheet '{sheet_name}': {e}")
        return sheets if sheets else None
    except Exception as e:
        print(f"  ❌ Failed to open XLSX: {e}")
        return None

def process_dataframe(df, file_name, sheet_name=None):
    results      = []
    source_label = f"{file_name} (sheet: {sheet_name})" if sheet_name else file_name

    for i, row in df.iterrows():
        row_dict = row.dropna().to_dict()
        if not row_dict:
            continue

        # Build the full text content — all data goes here
        content_parts = [f"Source: {source_label}"]
        if sheet_name:
            content_parts.append(f"Sheet: {sheet_name}")
        for k, v in row_dict.items():
            content_parts.append(f"{k}: {v}")
        full_content = "\n".join(content_parts)

        # Fix [5]: Minimal metadata — only what's needed for filtering.
        # All field values are already present in the chunk text above,
        # so storing them again as metadata just bloats ChromaDB.
        base_metadata = {
            "source":  file_name,
            "row_id":  str(i),
            **({"sheet": sheet_name} if sheet_name else {}),
        }

        chunks = text_splitter.split_text(full_content)
        for chunk_idx, chunk_text in enumerate(chunks):
            content      = normalize(chunk_text)
            combined_str = f"{content}_{file_name}_{sheet_name or ''}_{i}_{chunk_idx}"
            doc_id       = hashlib.md5(combined_str.encode()).hexdigest()
            metadata     = {**base_metadata, "chunk_index": chunk_idx}
            results.append((
                Document(page_content=content, metadata=metadata, id=doc_id),
                doc_id
            ))
    return results

def invalidate_bm25_cache():
    """Delete the BM25 pickle so main.py rebuilds it on next startup."""
    if os.path.exists(BM25_CACHE_PATH):
        os.remove(BM25_CACHE_PATH)
        print("🗑️  BM25 cache invalidated — will rebuild on next main.py start.")

def index_file(file_path: str):
    """Index a single CSV or XLSX file. Invalidates BM25 cache on success."""
    file_name = os.path.basename(file_path)
    ext       = os.path.splitext(file_name)[1].lower()
    documents = []
    ids       = []
    indexed   = False

    if ext == ".csv":
        existing = vector_store.get(where={"source": file_name}, limit=1)
        if existing["ids"]:
            print(f"⏭️  Already indexed: {file_name}")
            return
        df = read_csv_safe(file_path)
        if df is not None:
            for doc, doc_id in process_dataframe(df, file_name):
                documents.append(doc)
                ids.append(doc_id)

    elif ext == ".xlsx":
        sheets = read_xlsx_safe(file_path)
        if sheets is None:
            return
        for sheet_name, df in sheets.items():
            existing = vector_store.get(
                where={"source": file_name, "sheet": sheet_name}, limit=1
            )
            if existing["ids"]:
                continue
            print(f"📄 New sheet: {file_name} → {sheet_name}")
            for doc, doc_id in process_dataframe(df, file_name, sheet_name=sheet_name):
                documents.append(doc)
                ids.append(doc_id)
    else:
        return

    if documents:
        total      = len(documents)
        batch_size = 10
        print(f"📦 Indexing {file_name}: {total} chunks...")
        for i in tqdm(range(0, total, batch_size), desc="Indexing", leave=False):
            vector_store.add_documents(
                documents=documents[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"✅ Indexed: {file_name}")
        indexed = True

    if indexed:
        invalidate_bm25_cache()

def index_all_existing():
    import glob
    all_files = (
        glob.glob(os.path.join(DATASET_DIR, "*.csv")) +
        glob.glob(os.path.join(DATASET_DIR, "*.xlsx"))
    )
    if not all_files:
        print(f"⚠️  No files found in {DATASET_DIR}/")
        return
    newly_indexed = False
    for file_path in tqdm(all_files, desc="Initial scan", leave=False):
        before = len(vector_store.get(limit=1)["ids"])
        index_file(file_path)
        # invalidate_bm25_cache already called inside index_file if needed


# ─── Watchdog handler ────────────────────────────────────────────────────────
class DatasetHandler(FileSystemEventHandler):
    def _handle(self, event):
        if event.is_directory:
            return
        ext = os.path.splitext(event.src_path)[1].lower()
        if ext in (".csv", ".xlsx"):
            print(f"\n🔔 File event: {os.path.basename(event.src_path)}")
            time.sleep(0.5)
            index_file(event.src_path)

    def on_created(self, event):  self._handle(event)
    def on_modified(self, event): self._handle(event)


# ─── Retriever ───────────────────────────────────────────────────────────────
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.7}
)

# ─── Main Server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"🚀 CSV/XLSX Vector Server started. Watching '{DATASET_DIR}/'...")

    index_all_existing()

    observer = Observer()
    observer.schedule(DatasetHandler(), path=DATASET_DIR, recursive=False)
    observer.start()
    print("👀 Watchdog active — waiting for new files...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Server stopped.")
    observer.join()