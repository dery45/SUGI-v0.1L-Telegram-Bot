"""
query_logger.py — Structured per-query logging untuk SUGI v0.1L

Setiap query menghasilkan satu log entry JSON di logs/queries.jsonl
Format JSONL (newline-delimited JSON) agar mudah di-grep, di-tail, dan
di-load ke pandas untuk analisis.

Fields per entry:
  ts            — ISO timestamp
  session_id    — ID sesi dari main.py
  query_id      — UUID unik per query
  question      — pertanyaan asli user
  rewritten     — hasil query rewriting (sama jika tidak ada history)
  scope_passed  — apakah lolos scope guard
  flags         — dict: is_plant, is_weather, has_history
  docs_retrieved — list ringkasan dokumen yang dipakai (source, score, snippet)
  answer_preview — 200 karakter pertama dari jawaban LLM
  eval          — dict hasil eval loop: faithfulness, relevance, flag
  latency_ms    — total waktu proses dalam milidetik
  error         — string error jika ada, null jika tidak
"""

import json
import os
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR  = _ROOT / "data" / "logs"
LOG_FILE = LOG_DIR / "queries.jsonl"
EVAL_FILE = LOG_DIR / "eval_flags.jsonl"  # hanya entry yang flagged

# P2-3: prompt versioning — bump when _ANSWER_TEMPLATE_BASE changes materially
PROMPT_VERSION = "v1"


def _ensure_log_dir():
    LOG_DIR.mkdir(exist_ok=True)


def new_query_trace(session_id: str) -> dict:
    """
    Buat trace baru untuk satu query. Panggil di awal,
    isi field-nya satu per satu, lalu panggil commit_trace() di akhir.
    P2-3: adds model_name + prompt_version for reproducibility (§19).
    """
    # Lazy import to avoid circular
    try:
        _model = os.getenv("LLM_MODEL", "sugi-v0.1L")
    except Exception:
        _model = "sugi-v0.1L"
    return {
        "ts":              datetime.now().isoformat(),
        "session_id":      session_id,
        "query_id":        uuid.uuid4().hex[:12],  # Part1: 12 hex (was 8) reduces collision 16^8→16^12
        "question":        "",
        "rewritten":       "",
        "scope_passed":    None,
        "flags":           {},
        "docs_retrieved":  [],
        "answer_preview":  "",
        "eval":            {},
        "latency_ms":      0,
        "error":           None,
        "model_name":      _model,
        "prompt_version":  PROMPT_VERSION,
        "_start_ts":       time.monotonic(),   # internal, tidak disimpan
        "_committed":      False,  # Part1: guard against double commit
    }


def set_docs(trace: dict, docs: list) -> None:
    """
    Isi docs_retrieved dari list LangChain Document.
    Simpan source, chunk_index, dan 120 karakter pertama konten.
    """
    trace["docs_retrieved"] = [
        {
            "source":      doc.metadata.get("source", "unknown"),
            "sheet":       doc.metadata.get("sheet", ""),
            "chunk_index": doc.metadata.get("chunk_index", ""),
            "snippet":     doc.page_content[:120].replace("\n", " "),
        }
        for doc in docs
    ]


def commit_trace(trace: dict, error: Optional[str] = None) -> None:
    """
    Finalisasi dan tulis trace ke JSONL. Hapus field internal.
    Part1: idempotent — second commit on same dict is no-op (prevents 9ms vs 3.2M ms duplicate).
    """
    _ensure_log_dir()
    # Part1: guard double commit (stale reference or query_id collision via truncation)
    if trace.get("_committed"):
        print(f"⚠️  commit_trace double-commit suppressed for {trace.get('query_id')}")
        return
    if "_start_ts" not in trace:
        print(f"⚠️  commit_trace missing _start_ts for {trace.get('query_id')} — suppressed")
        return
    trace["latency_ms"] = int((time.monotonic() - trace.pop("_start_ts")) * 1000)
    trace["_committed"] = True
    trace["error"]      = error

    # Part1: exclude internal fields from persisted JSON
    to_write = {k: v for k, v in trace.items() if not k.startswith("_")}
    # Tulis ke queries.jsonl
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_write, ensure_ascii=False) + "\n")

    # Kalau ada flag dari eval, tulis juga ke eval_flags.jsonl
    if trace.get("eval", {}).get("flag"):
        with open(EVAL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_write, ensure_ascii=False) + "\n")

    _print_trace_summary(trace)


def _print_trace_summary(trace: dict) -> None:
    """Cetak ringkasan singkat ke terminal setelah tiap query."""
    ev    = trace.get("eval", {})
    flag  = ev.get("flag", False)
    fth   = ev.get("faithfulness", "n/a")
    rel   = ev.get("relevance",    "n/a")
    lat   = trace.get("latency_ms", 0)
    docs  = len(trace.get("docs_retrieved", []))
    icon  = "🚩" if flag else "✅"
    print(f"\n{icon} [LOG {trace['query_id']}] "
          f"lat={lat}ms docs={docs} faith={fth} rel={rel}")
    if flag:
        print(f"   ⚠️  Flagged: {ev.get('reason', '')}")


# ─── Log reader helpers ───────────────────────────────────────────────────────

def tail_logs(n: int = 20) -> list[dict]:
    """Ambil n entry terakhir dari queries.jsonl."""
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines[-n:]]


def flagged_logs() -> list[dict]:
    """Ambil semua entry yang di-flag oleh eval loop."""
    if not EVAL_FILE.exists():
        return []
    lines = EVAL_FILE.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines]


def session_logs(session_id: str) -> list[dict]:
    """Ambil semua entry untuk satu session_id."""
    if not LOG_FILE.exists():
        return []
    results = []
    for line in LOG_FILE.read_text(encoding="utf-8").strip().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("session_id") == session_id:
                results.append(entry)
        except json.JSONDecodeError:
            continue
    return results


def _percentile(data: list[int], pct: float) -> int:
    """Same percentile logic as test harness — linear interpolation not needed, use sorted index."""
    if not data:
        return 0
    s = sorted(data)
    idx = int(len(s) * pct / 100)
    return s[min(idx, len(s)-1)]

def print_debug_report(n: int = 10) -> None:
    """
    Cetak ringkasan debug ke terminal.
    P2-3: now includes p50/p95 latency + model/prompt visibility.
    Panggil dengan: python -c "from query_logger import print_debug_report; print_debug_report()"
    """
    entries  = tail_logs(n)
    flagged  = flagged_logs()
    total    = len(entries)
    n_flag   = sum(1 for e in entries if e.get("eval", {}).get("flag"))
    lats     = [e.get("latency_ms", 0) for e in entries]
    avg_lat  = int(sum(lats) / max(total, 1))
    p50      = _percentile(lats, 50)
    p95      = _percentile(lats, 95)
    avg_docs = sum(len(e.get("docs_retrieved", [])) for e in entries) / max(total, 1)

    print("\n" + "="*55)
    print(f"  SUGI Query Debug Report — last {total} queries")
    print("="*55)
    print(f"  Flagged by eval : {n_flag} / {total}")
    print(f"  Avg latency     : {avg_lat} ms  p50 {p50} ms  p95 {p95} ms")
    print(f"  Avg docs used   : {avg_docs:.1f}")
    if entries:
        models = {e.get("model_name","?") for e in entries}
        prompts = {e.get("prompt_version","?") for e in entries}
        print(f"  Models        : {', '.join(models)}  Prompts: {', '.join(prompts)}")
    print("-"*55)
    for e in entries:
        ev   = e.get("eval", {})
        icon = "🚩" if ev.get("flag") else "  "
        print(f"  {icon} [{e['query_id']}] {e['question'][:45]:<45} "
              f"faith={ev.get('faithfulness','?')} rel={ev.get('relevance','?')} "
              f"{e.get('latency_ms',0)}ms model={e.get('model_name','?')}")
    print("="*55)
    if flagged:
        print(f"\n  Total flagged entries in eval_flags.jsonl: {len(flagged)}")
    print()