# SUGI v0.1L — Decision Log

Each entry: tag, what changed, where, why. Tag IDs match inline code comments —
`grep TAG_ID` in the codebase finds every touched line, and the same ID links
back here. This is the single index for the review-round tags spread across the
codebase (C/D/B/M/O/A/R/Q). See `CHANGELOG.md` for the versioned history; this
file is the *why* behind each tag.

Entries are grouped by the phase the work was actually done in. Superseded
approaches are noted inline (e.g. M1's compression wrapper was later removed by
A11) so the log reads as a narrative, not a flat list of coexisting features.

---

## Stage 1 — Critical (Latency)

### C1 — Background eval loop
**Files:** `core/sugi_core.py` (`ask()`, `_run_eval_and_commit`)
**Problem:** Eval ran synchronously in `ask()`, adding LLM-eval latency (5–15s) to
every user-facing response for a score only used for logging.
**Fix:** Extracted eval + `commit_trace` into `_run_eval_and_commit()` run on a
daemon thread; `ask()` returns as soon as the answer is generated. The error path
still commits the trace synchronously so a crash doesn't lose the log.
**Also:** `core/eval_loop.py` — eval model's request `timeout` must go through
`client_kwargs` (defense-in-depth against a hanging thread); langchain-ollama 1.0.1
swallows a bare `timeout=` kwarg.

### C2 — Model residency alignment
**Files:** `core/sugi_core.py` (`self.model`, `self.rewrite_model`),
`core/eval_loop.py` (`_eval_model`)
**Problem:** Each model had a different `keep_alive`, so a model could be unloaded
from Ollama between calls, forcing a per-query reload (seconds of latency).
**Fix:** Unified `keep_alive=600` (10 min) on the main model, rewrite model, plant
extraction model, and eval model.

### C3 — Eval/stability guard
**Files:** `core/sugi_core.py` (`self.model`), `core/eval_loop.py`
**Problem:** Long contexts / unbounded output could overflow Ollama's context and
error out mid-answer.
**Fix:** Main model `num_ctx=4096` + `num_predict=512` (token-output ceiling).
Eval model `num_ctx=4096`, request `timeout=15` via `client_kwargs`.

### C4 — Parallel plant-detail fetch
**Files:** `core/plant_api.py`
**Problem:** Species-detail fetches for the top-5 candidates ran sequentially —
wall-clock time was the sum of every request's RTT.
**Fix:** `ThreadPoolExecutor(max_workers=3)` + `as_completed`. The global `_ApiQueue`
spacing (1.1s) still enforces the Perenual rate limit; concurrency only removes the
serial RTT addition, it does not exceed the limit.

---

## Stage 2 — Timing & Caching

### D1 — Instrument ask() timing
**Files:** `core/sugi_core.py` (`ask()`, `_run_eval_and_commit`),
`interfaces/telegram/telegram_bot.py`
**Problem:** Needed to distinguish SugiCore-answer latency from wrapper/network
latency, and (after C1) to confirm the eval loop no longer blocks the reply.
**Fix:** Independent `time.monotonic()` wall-clock measurements in `ask()` (reply
path, eval excluded) and in the Telegram `to_thread` wrapper.

### D2 — Timeout via httpx client, not silent kwarg
**Files:** `core/sugi_core.py` (`self.rewrite_model`, `self._plant_extract_model`)
**Problem:** A bare `timeout=` kwarg on `OllamaLLM` is not a real parameter in
langchain-ollama 1.0.1 — it's absorbed by `*args` silently, so the intended timeout
never applied.
**Fix:** Pass `client_kwargs={"timeout": 30}` (httpx-level) instead.

### D3 — Don't pay Qwen for out-of-scope questions
**Files:** `core/sugi_core.py` (`ask()`)
**Problem:** Qwen rewriting ran even for questions already determined out-of-scope,
spending LLM time on a question that will be refused.
**Fix:** Only rule-based (suffix) rewriting may bypass the scope check; Qwen is
disabled (`allow_qwen=False`) for out-of-scope questions.

### B1 — Bounded rewrite cache
**Files:** `core/sugi_core.py` (`_rewrite_cache`, `_qwen_rewrite`)
**Problem:** The same referential question paid Qwen repeatedly; also set a worst-
case bound on rewrite-model cost.
**Fix:** `self.rewrite_model num_predict=40` (bounded output) + per-process,
thread-safe LRU-ish cache keyed on `(normalized question, history fingerprint)`.

### B2 — Bounded plant-extraction cache
**Files:** `core/sugi_core.py` (`_plant_extract_cache`, `_extract_and_translate_plant`)
**Problem:** Qwen plant-name fallback (the expensive path, after the cheap name-map)
was re-invoked for identical questions.
**Fix:** Cache Qwen extraction results per normalized question, capped at 500
entries with oldest-first eviction.

### B4 — Embed once, reuse for weather & memory
**Files:** `core/sugi_core.py` (`ask()`)
**Problem:** Each `similarity_search` re-embedded the query — redundant embedding
calls for weather and memory retrieval.
**Fix:** Embed the query once (`embeddings.embed_query`), then use
`similarity_search_by_vector` for the weather and memory stores.

### B5 — Parallel independent retrievals
**Files:** `core/sugi_core.py` (`ask()`)
**Problem:** Weather, RAG-ensemble, and memory searches ran sequentially; wall-clock
was the sum of all three.
**Fix:** Run them in a `ThreadPoolExecutor(max_workers=3)` and await all. Merge/dedup
stays sequential below, so retrieval results are identical to the sequential version.

### B8 (M2) — Cap context at 8 chunks
**Files:** `core/sugi_core.py` (`ask()`)
**Problem:** After M1 merged weather into the RAG pool, the prompt could exceed
`num_ctx=4096`.
**Fix:** Final context capped at 8 chunks (`all_docs[:8]`). Part of the M2 context
budget work.

---

## Stage 3 — Startup & Resource Contention

### M0 — Startup stagger + insight LLM pacing
**Files:** `services/farmer_insight_service.py`, `services/daily_insight.py`,
`services/vectorCSV.py`, `services/vectorpdf.py`, `services/government_insight_service.py`
**Problem:** Background services (insight engines, vector watchers) woke at startup
simultaneously and contended with the chatbot + embeddings for Ollama.
**Fix:** `STARTUP_GRACE_SECONDS` delays each service's first full run so the chatbot
wins the Ollama warm-up race; `INSIGHT_LLM_DELAY` paces LLM calls inside insight
loops (M0b).

### M1 — Reranker top_n raised to 8
**Files:** `core/sugi_core.py`
**Problem:** With weather merged into the RAG pool, `top_n=8` was needed so combined
weather+RAG results fit the final 6–8 chunk budget.
**Fix:** `CrossEncoderReranker(top_n=8)`.
**Superseded by A11:** the `ContextualCompressionRetriever` wrapper this set up for
double-reranking was later removed — see A11.

### M2 — Context cap & prompt trim
**Files:** `core/sugi_core.py`
**Problem:** Combined weather+RAG+history risked overflowing `num_ctx=4096`.
**Fix:** Context capped at 8 chunks (see B8), persona moved to the Modelfile SYSTEM
prompt, session date/time pinned to the bottom of the answer prompt, bullet depth
limited to 1 level.

---

## Stage 4 — Robustness & Observability

### O1 — Measure reranker cost in isolation
**Files:** `core/sugi_core.py` (`ask()`)
**Problem:** Needed data on whether to gate (skip) the cross-encoder reranker for
simple queries.
**Fix:** Timing of `compress_documents` measured in isolation
(`tests/test_stage4_o1.py`), feeding the hold/gate decision.

### O2 — Round time to 5 minutes
**Files:** `core/sugi_core.py` (`_round_time`)
**Problem:** Ollama prefix-cache is empirically active on this setup — identical
prompt prefixes are ~21% faster (verified in `tests/test_stage4_o2.py`).
**Fix:** Round `HH:MM` to the nearest 5 minutes so similar queries share a prompt
prefix; accuracy of "current session info" stays acceptable (±2.5 min).

### O3 — Fixed time phrases are not referential
**Files:** `core/sugi_core.py` (`_TIME_PHRASES`, `_mask_time_phrases`,
`_ref_in_time_phrase`)
**Problem:** "ini/itu" inside fixed time phrases ("hari ini", "malam itu") was
treated as a referential signal, so the rule-based rewriter substituted the prior
subject and broke queries like "harga cabe hari ini" (could even mis-detect weather).
**Fix:** Time phrases are masked (positions preserved with spaces) before
referential detection; helper `_ref_in_time_phrase` rejects refs inside them.

### A1 — Diagnosable one-line errors
**Files:** `services/government_insight_service.py`
**Problem:** Per-collection errors printed only `str(e)` — no exception type, no
field name — hard to diagnose from logs.
**Fix:** Log `{collection}: {type(e).__name__}: {e} (field: tahun)`; also normalize
mixed None/str/int year values in `distinct("tahun")` before sorting to avoid a
`TypeError: '<'`.

### A2 — Encrypted PDFs are not generic failures
**Files:** `services/vectorpdf.py`
**Problem:** Password-protected (AES) PDFs raised a generic "Error processing" that
read like a system failure.
**Fix:** Detect AES/encrypted markers and log a dedicated "🔒 Skipped (encrypted)"
message.

### A5 — Thread-safe rate-limit flag
**Files:** `core/plant_api.py`
**Problem:** Parallel detail fetches (C4) could race on the global rate-limit flag:
one worker's single 429 could disable the whole API for 1 hour, and concurrent 429s
could spam the "API disabled" log.
**Fix:** All read/write of the cooldown flag wrapped in `threading.Lock`;
`_trip_rate_limit()` returns True only when the flag is newly set, preventing
duplicate disable-log lines. Verified in `tests/test_stage4_a5.py`.

### A6 — num_predict via constructor, not .bind()
**Files:** `core/sugi_core.py` (`self._plant_extract_model`)
**Problem:** `chain.bind(num_predict=...)` forwarded the kwarg to
`Client.generate()` which rejects it ("unexpected keyword argument") — the plant
fallback silently died.
**Fix:** Set `num_predict=10` on the `OllamaLLM` constructor (the pattern already
proven by `self.model`/`self.rewrite_model`). Verified in `tests/test_stage4_a6.py`.

### A7 — Mongo ping with retry + backoff
**Files:** `services/daily_insight.py`, `services/farmer_insight_service.py`,
`services/government_insight_service.py`
**Problem:** Atlas transiently returns "No primary found" (ServerSelectionTimeoutError)
during topology election; startup ping once → `sys.exit(1)` on a hiccup, relying only
on external restart.
**Fix:** `_ping_with_retry` — 3 attempts, linear backoff (5s, 10s). Verified in
`tests/test_stage4_a7.py`.

---

## Stage 5 — Correctness Hardening

### A8 — Insight-engine startup stagger + LLM timeout
**Files:** `services/daily_insight.py`, `services/farmer_insight_service.py`,
`services/government_insight_service.py`, `config/.env.example`
**Problem:** All three insight engines shared one `STARTUP_GRACE_SECONDS` (default
180s) and launched together, waking simultaneously and contending for Ollama; LLM
timeout was a tight 90s (daily had none).
**Fix:** Per-service grace vars with staggered defaults — Daily 180s, Government
240s, Farmer 300s (fallback chain: per-service → shared → default) — and the
insight-only LLM timeout raised to 240s in all three engines. Verified in
`tests/test_stage5_a8.py`.

### A9 — Multi-turn history poisoning fix
**Files:** `core/sugi_core.py`
**Problem:** In a session, unrelated follow-on questions ("rekomendasi tanaman
terbaik di tanam bulan desember", "cara menanam salak") answered with the topic of
the PRIOR turn (e.g. apple), because the full past answer was injected into
`{history}` and the template treats history as primary reference. The rewriter +
retrieved context were correct — the model was anchored by the old answer text.
**Fix:** History is injected into the prompt only when the rewriter detects a
referential signal (`rewrite_type != "none"`) via new `_needs_ref_context()` /
`_select_prompt_history()`; self-contained questions get
"Belum ada riwayat percakapan." Template also forbids echoing the `User:`/`Sugi:`
dialog format and claiming missing data when data was given. Verified live + unit
test `tests/test_stage5_a9.py`.

### A10 — Retrieval-level memory gating (+ A4 closure)
**Files:** `core/sugi_core.py`
**Problem:** A9 gated the prompt's `{history}` text, but the retrieval path still
branched on the coarser `has_history` flag in three places, so self-contained new
questions with ANY session history still pulled old-topic memory content into the
retrieved context via the memory store.
**Fix:** All three sites now gate on `needs_ref_context` (rewrite_type != "none"):
the `_build_retriever()` call site, the `mem_future` similarity-search gate, and the
memory-doc merge into `all_docs`.
**A4 closed here:** the shared unfiltered `self.memory_retriever` (no `user_id`
filter — a cross-user privacy leak) is removed; only the correctly scoped
`mem_future` path remains. Verified in `tests/test_stage5_a10.py`.

### A11 — Startup-429 NameError + single-pass rerank
**Files:** `core/plant_api.py`, `core/sugi_core.py`, `core/eval_loop.py`
**Problem (a):** The module-level API-key startup validation calls
`_trip_rate_limit()` on a 429, but that function was defined further down — a real
429 at import time raised `NameError` (which `except ImportError` cannot catch),
killing the whole bot.
**Fix (a):** Moved the rate-limit flag block (flag + lock + `_is_rate_limited` +
`_trip_rate_limit`) above the validation block. Verified in `tests/test_stage5_a11.py`.
**Problem (b):** `_build_retriever` returned a `ContextualCompressionRetriever`, so
RAG docs were cross-encoder-scored twice on weather queries (once in the wrapper,
once at the weather+RAG merge). M1's compression wrapper had introduced this.
**Fix (b):** Builder returns the raw ensemble; ALL candidates are reranked exactly
once at the merge point, with or without weather.
**Also:** removed dead `include_weather` param/branch and `self.weather_retriever`
(vestigial since M1's manual fetch-merge-rerank); eval model `num_predict=10`.

### A12 — Daily-insight concurrency backfill
**Files:** `services/daily_insight.py`
**Problem:** The three daily generators still used
`ThreadPoolExecutor(max_workers=5)` (hardcoded) + `ex.map`, firing up to 5
simultaneous qwen2.5:1.5b calls per group loop — the same burst-concurrency that
caused A8's original timeouts, even in a single `run_once()`.
**Fix:** All three pools replaced with `_run_paced()` — sequential by default
(`DAILY_INSIGHT_MAX_WORKERS`, default 1) with `INSIGHT_LLM_DELAY` pacing between
calls (same pattern as farmer/gov). `>1` is an explicit, env-driven choice that
restores the thread pool. Also backfilled the A7 fix this service skipped: startup
Mongo ping now uses `_ping_with_retry`.

---

## Phase 1 — Security & Correctness (final review)

### R1 — Fail-closed debug access
**Files:** `interfaces/telegram/telegram_bot.py`
**Problem:** `DEBUG_ALLOWED_USERS` gate used `if DEBUG_ALLOWED_USERS and user_id not
in ...` — with the env var unset (empty), the guard never fired and ALL users could
run `/debug` and `!`-commands.
**Fix:** Both gates (`cmd_debug()` and the `!`-command branch in `handle_message()`)
are deny-by-default (`user_id not in DEBUG_ALLOWED_USERS`). A one-time startup
warning prints when the variable is unset. CLI `handle_debug_command` intentionally
left ungated (known gap).

### R2 — Enforce blocked-topics list
**Files:** `core/sugi_core.py`, `config/settings/scope_config.ini`
**Problem:** `self.blocked_kw` (43 keywords) was loaded but never read by
`_is_in_scope()`, so the whole blocked list was dormant.
**Fix:** `_is_in_scope()` checks `blocked_kw` FIRST — blocked overrides allowed (e.g.
"apakah padi bisa mengobati diabetes?" now refuses despite "padi" being in-scope).
The 43-term list was reviewed before enabling; no agriculture false-positives.

### R3 — Global Telegram error handler
**Files:** `interfaces/telegram/telegram_bot.py`
**Problem:** No `add_error_handler(...)` was registered, so exceptions thrown outside
`SugiCore.ask()`'s internal try/except (e.g. failure inside `_send_long`) left the
user with silence.
**Fix:** `_global_error_handler` (logs traceback + best-effort "⚠️ Maaf, terjadi
kesalahan tak terduga…" reply) registered before `run_polling()`. Runtime-verified
with a stub PTB `Application`: forced `_send_long` raise → fallback delivered.

### R4 — Content-hash re-index dedup
**Files:** `services/vectorCSV.py`, `services/vectorpdf.py`
**Problem:** `index_file()` deduped by filename only
(`where={"source": file_name}`), so an edit to an already-indexed file was never
re-indexed.
**Fix:** Each chunk carries `file_hash` metadata (MD5 over 8KiB blocks; per-sheet
`df.to_csv()` hash for XLSX). On mismatch: delete old chunks (`vector_store.delete`)
then re-index. Unchanged files are skipped. Also backfilled BM25 invalidation for
PDFs (`vectorpdf.py` had none).

---

## Phase 2 — Maintainability

### Q1 — Consolidate the tag soup into one changelog
**Files:** `docs/decisions.md` (this file), inline comments across `core/`,
`services/`, `interfaces/`
**Problem:** Every fix across five rounds got a paragraph-length inline comment. Good
for the moment, but there was no single place listing "everything that changed and
why" — a new developer had to reconstruct history from comments scattered across nine
files.
**Fix:** This decision log (one section per tag, organized by phase) + inline
comments trimmed to a one-line pointer (`# A9: ... — see docs/decisions.md#a9`).
Tag IDs unchanged as the anchor between code and log.

### Q2 — Extract shared `services/insight_common.py`
**Files:** `services/insight_common.py` (new), `services/daily_insight.py`,
`services/farmer_insight_service.py`, `services/government_insight_service.py`
**Problem:** `_ping_with_retry`, context helpers (`_get_rag_context`,
`_get_weather_context`, `_get_plant_context`), and the
`OllamaLLM(..., client_kwargs={"timeout": 240})` construction were duplicated across
the three insight services, converging independently on the same patterns.
**Fix:** Extract shared functions into `insight_common.py`. Per-call-site parameters
(`k`, `timeout`, `temperature`, label prefixes, truncation) are passed explicitly so
behavior is preserved, not silently unified. `insight_common.py` never imports from
the three services (one-way dependency only).

---

## Phase 3 — Final Review: live findings & design-level items

### N1 — Province metadata hoist for price insights
**Files:** `services/vectorCSV.py` (`process_dataframe()`, `_file_content_hash`,
`_df_content_hash`), `services/vectorpdf.py` (`_file_content_hash`)
**Problem:** `generate_price_insights` (`services/daily_insight.py`) groups by
`d["meta"].get("province") or d["meta"].get("region") or "Nasional"`, but
`process_dataframe()` never wrote a `province`/`region` key — `base_metadata`
only held `source`/`row_id`/`data_type`/`sheet`. So every one of ~40k price
documents fell into the `"Nasional"` bucket and the service produced ONE insight
instead of one per province — the province value existed in the chunk *text*
("provinsi: Aceh"), just not in queryable metadata. Likely broken since the
function was written.
**Fix:** Hoist the province column into chunk metadata using the same candidate
list as `farmer_insight_service._province_field()` (`provinsi`, `province`,
`nama_provinsi`, `wilayah`, `region`); matched rows get a `province` metadata
key. Also added `_METADATA_SCHEMA_VERSION = "v2"` mixed into the R4 content
hashes (in `vectorCSV.py` and `vectorpdf.py`) so this metadata-logic change
triggers exactly ONE deliberate full reindex via R4's existing delete+reindex
mechanism — replacing the accidental multi-hour migration R4 caused with a
controlled, versioned one. Reindex must be scheduled in a low-traffic window.

### S1 — Stemmed lexical eval overlap
**Files:** `core/eval_loop.py` (`_lexical_faithfulness`, `_lexical_relevance`)
**Problem:** Lexical overlap compared raw tokens, so morphological variants matched
nothing: "cara menanam cabe" versus "Penanaman cabai" shared zero roots and got
flagged LOW/re-hallucinated, or pushed into the inconclusive band where the Qwen
eval was invoked unnecessarily (contending with the chatbot for Ollama).
**Fix:** Tokens are reduced to stems via Sastrawi (`_stem_tokens`), with a graceful
fallback to raw tokenization if the package is absent. Re-validated against the
`data/logs/queries.jsonl` sample before shipping: with stemming, faithfulness
overlap mean rises to ~0.34 (median ~0.30), so the old 0.40/0.20 cutoffs no longer
separate correctly — sweep on LLM-resolved verdicts (n=56) shows false-HIGH dropping
5→3 and correct decisive LOWs rising 12→21 by moving to **0.50/0.25**. Relevance
thresholds (0.50/0.25) held (0 false-HIGH in sample) and were left unchanged.
Also guarded the shared Sastrawi stemmer with a lock: eval runs on a background
thread (C1) and can be concurrent across queries.

### S2 — Scalability roadmap (decision, not a patch)
**Files:** `core/sugi_core.py` (`self._sessions`), service startup topology
**Problem:** Two architectural ceilings were confirmed as design-level rather than
bug-level: (1) `self._sessions` (session/history state) lives in one process's
memory, blocking horizontal scaling behind a load balancer; (2) a single Ollama
instance serves the chatbot plus the three insight engines, and Ollama queues
rather than truly parallelizes — pacing/staggering (M0, A8) reduces collision
frequency but does not remove the ceiling.
**Fix (decision):** Option B on both — explicitly descope for now. The live logs
show a handful of active users, not production-scale concurrent load; both
existing mitigations are working at current scale. Revisit (Redis-backed sessions;
a second Ollama daemon for background insight generation) only when traffic or
availability requirements actually demand it — speculative shared-infra here adds
a new failure mode (Redis availability, two daemons to tune) for a problem that
may not materialize for a long time.
