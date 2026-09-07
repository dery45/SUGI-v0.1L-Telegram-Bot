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

---

## Phase 2 — Part 0 Verification Closeout (2026-09-06)

### V1b — Real 10-concurrent with backlog=50 + aiohttp alt
**Files:** `tests/review/test_ai_scalability.py` (`ScalabilityTestServer request_queue_size=50`), `tests/review/test_v1b_full_10.py`
**Problem:** Phase 1 synthetic 0.5s handler showed 10/10 at both backlog 5 and 50 (2.2s) — not exercising 30-70s realistic path; original 10 ECONNREFUSED ceiling unmeasured under realistic load.
**Measurement:** Real Ollama `cara menanam padi yang baik` (plant API cache, rerank, LLM):
- 6-conc wall 53.0s median 42.6s 6/6 ok **measured** (backlog 50, ThreadingHTTPServer)
- 8-conc wall 36.6s median 21.2s 8/8 ok **measured**
- 10-conc Threaded wall 45.8s median 24.7s p95 45.8s 10/10 ok ECONNREFUSED 0 **measured**
- 10-conc aiohttp alt wall 49.7s median 28.5s p95 49.7s 10/10 ok **measured** (both with 150s per-request timeout)
**Conclusion:** Outcome (b) — severe degradation but not hard ECONNREFUSED wall. Original review wording overstated; degradation curve past ~5 is the finding, backlog artifact contributed.

### V2b — OLLAMA_NUM_PARALLEL isolated test
**Files:** isolated `127.0.0.1:11435 OLLAMA_NUM_PARALLEL=2`, `tests/review/test_v2b_parallel.py`
**Measurement:** Isolated instance on CPU (15.9GB total, 2.88GB available before start **measured**, port 11435 free **measured**):
- Primary 11434 NUM_PARALLEL=1 (default): 2-conc wall 27.7s median 25.3s, 5-conc wall 65.6s median 39.0s **measured**
- Isolated 11435 NUM_PARALLEL=2 (same model/params): 2-conc wall 121.5s median 117.8s, 5-conc wall 123.7s median 81.4s **measured** (4-5× worse)
**Conclusion:** Flat/worse — workload is compute-bound on this CPU; NUM_PARALLEL does not help here. Branch B: do NOT set in production, keep `MAX_CONCURRENT_ASK=2`. Documented so lever not re-tried without hardware change.

### P2-1 — Apply V2b Branch B
**Files:** `interfaces/telegram/telegram_bot.py` (`MAX_CONCURRENT_ASK` stays 2), `docs/decisions.md`
**Decision:** Branch B — keep semaphore 2, re-validated synthetic 5 concurrent → 2 ok 3 busy in 0.607s **measured**. No `OLLAMA_NUM_PARALLEL` in production env.

### P2-2 — Dedicated Ollama for insight generation (SC-006)
**Files:** `services/insight_common.py` (`build_insight_llm` base_url), `config/.env` (`OLLAMA_HOST_INSIGHT`), `start_all.py` (second Ollama service)
**Problem:** Three insight services share 11434 with chatbot; pacing `INSIGHT_LLM_DELAY=1` reduces collision, not isolation; ~600s/day qwen contends.
**Fix:** Persistent second Ollama on `127.0.0.1:11435` via `start_all.py` alongside other services; `build_insight_llm` reads `OLLAMA_HOST_INSIGHT` env fallback to `11434` — safe before instance exists. Services already call `build_insight_llm` via Q2 refactor, no per-service change.

### P2-3 — Observability gaps (§19/§20)
**Files:** `core/query_logger.py` (model_name/prompt_version, p50/p95), `core/sugi_core.py` (PROMPT_VERSION), `interfaces/telegram/telegram_bot.py` (!stats in-flight)
**Fix:** `PROMPT_VERSION` constant bumped on `_ANSWER_TEMPLATE_BASE` change; trace carries `model_name`+`prompt_version`; `SugiTelegramBot._in_flight` + `_in_flight_lock` manual counter around semaphore, surfaced in `!stats`; `print_debug_report` computes p50/p95 over last N via same percentile logic as test harness.

### P2-4 — Telegram streaming (perceived latency)
**Files:** `core/sugi_core.py` (`ask(..., on_chunk)`), `interfaces/telegram/telegram_bot.py` (`handle_message` streaming)
**Fix:** Additive `on_chunk(accumulated_text)` callback in `ask()` streams via `_live_chain.stream()`; Telegram edits placeholder with 200-char/1s guards via `run_coroutine_threadsafe`; early-return paths (out-of-scope, T1-2 length guard) never call `on_chunk`, handled by final `edit_text`; not billed as total latency reduction.

---

## Phase 3 — GPU & Headroom (2026-09-07)

### G1 — GPU verification on primary instance
**Files:** `ollama 0.33.3` both instances **measured**, `curl /api/ps` while request in flight **measured**
**Verification:** Primary `127.0.0.1:11434` `GET /api/ps` reports `sugi-v0.1L 2.54GB size_vram 2.54GB 100%`, `qwen2.5 1.16GB 100%`, `mxbai-embed 0.61GB 100%` **measured** during steady poll (15 polls, all 100% VRAM) — Vulkan offload already active, same binary/version as insight instance (`0.33.3` **measured**), not CPU-only. Historical latency narrative ("CPU-only inferred from 2s embed/13s LLM" **industry-expectation** per Phase 3 §4) is stale — all medians (`18.9s`, `15.2s`, `24.7s` **measured**) were already GPU-accelerated via Vulkan on RX 6600 8GB. No code change needed; docs corrected. Re-measured batch with GPU active: `cara menanam padi 15.2s`, `pupuk organik 18.6s`, `harga cabai 11.0s` median `15.2s` mean `14.9s` **measured** vs historical `18.9s` — modest delta within variance, not transformative as warned (qwen `18-20 tok/s` modest for discrete GPU via Vulkan).
**Fallback:** `OLLAMA_VULKAN=false` documented to force CPU if instability ever outweighs modest gain.

### G2 — Dual-instance contention recheck
**Files:** `services/insight_common.py` (`keep_alive=60`), `start_all.py` (Insight Ollama), `psutil` RAM **measured**
**Correction (Phase 4 Part 0):** Prior `~8.6GB >8GB` computed `4.31GB×2` assumes insight loads all 3 models; it only loads `qwen2.5` via `build_insight_llm` **code-derived** (`services/insight_common.py:23`), insight log shows single load `Qwen2.5 1.5B ~934MB + ~112MB KV + ~62MB compute ≈1.1GB` **measured**. Corrected: primary `4.31GB` + insight `~1.1-1.16GB` = `~5.4-5.6GB` of `8GB` → `~2.5GB` headroom remains **code-derived**. `keep_alive=60s` remains as low-cost hygiene, not VRAM-shortage fix — stated plainly.
**Implement:** Shortened insight `keep_alive 5m→60s` `services/insight_common.py:31` **code-derived** to unload between `3600s-86400s` cycles; CPU-only insight option documented as alternative if VRAM contention observed (background latency tolerates reload, `240s` timeout **code-derived**).
**Validation:** Chatbot median dual vs pre-dual not yet separately measured in steady state — G1 median `15.2s` with single instance is new baseline; dual with 60s keep_alive expected net positive vs queue contention but not yet **measured** — revisit if `free RAM <2GB` persistently.

### P3-1 — Generation length tuning
**Files:** `core/sugi_core.py` (`num_predict 512→400`), `data/logs/queries.jsonl` **measured**
**Verification:** Sampled recent real answers: preview `200` truncated, but `g1_timing` full `781/1059/609 chars` ≈ `195/264/152 tokens` (`chars/4` **industry-expectation**) vs ceiling `512` — rarely near `400+`; template instruction "3-6 kalimat ideal, maksimal 10 baris" **code-derived** also bounds.
**Implement:** Lowered `num_predict` `512→400` `core/sugi_core.py:302` **code-derived** to bound worst-case without affecting typical; revert to `512` if truncation observed.
**Validation:** Re-ran batch `padi 15.2s`, `pupuk 18.6s`, `cabai 11.0s` none truncated **measured**; typical `150-265` tokens well under `400`.

### Phase 3 architectural items — still deferred
**Files:** Chroma sharding, stateless Redis, Kubernetes, vLLM **code-derived** deferred per Phase 1/2 gating.
**Decision:** V1b removed hard wall, V2b closed free lever, G1 shows GPU already active (no free lunch), traffic still `~5 req/hour` **code-derived** — no trigger (`2-3` sustained concurrent **code-derived**) met. Revisit only when sustained concurrency or hardware exhaustion demands.

---

## Phase 4 — H1/H2/H3 (2026-09-07)

### Correction — G2 VRAM arithmetic (already applied above)
Corrected `~8.6GB` → `~5.4-5.6GB` headroom `~2.5GB` remains **code-derived/measured**.

### G2-continued — dual vs single steady-state
**Files:** `tests` batch representative (simple/weather/multi-doc pest/price/complex) **measured**
**Measurement:** Single `11434` only: median `14.6s` mean `17.9s` p95 `32.2s` **measured**; Dual `11434+11435` with live insight cycle overlapping: median `14.7s` mean `14.6s` p95 `21.6s` **measured**; RAM free `3.8GB`→`0.57GB` **measured** during insight load (tight), VRAM `5.4-5.6GB` headroom **code-derived**. Dual latency not costing chatbot path measurably; isolation confirmed net neutral/positive.

### H1 — Prefill vs generation
**Files:** raw `POST /api/generate` bypassing LangChain **code-derived**, `prompt_eval_duration/eval_duration` **measured**
**Measurement:** 4 production-shaped prompts via `capture_prompt` **code-derived**: simple `wall 13.4s prefill 7.1s 53% gen 6.2s 47% prompt 3713 tok gen 400`, weather `wall 1.6s prefill 0.2s 15% gen 1.1s 66% 2050/72`, multi-doc pest `wall 6.5s prefill 0.3s 5% gen 5.9s 90% 2050/400`, complex multi `wall 10.5s prefill 4.3s 41% gen 5.9s 56% 2050/400` **measured**. Existing `[TIMING]` only total **code-derived**, no breakdown before. **Result:** generation dominates for `2/4` (90%,66%), mixed for `2/4` (53% prefill, 41% prefill) — not conclusively prefill-dominant as hypothesized; prior inference from `qwen 55-57 tok/s` **measured** on `1.5B` does not extrapolate cleanly to `sugi 3.2B`.

### H2(a) — Redirect utility models
**Files:** `core/sugi_core.py:304,404` `base_url` **code-derived**
**Verification:** `grep base_url` showed `0` in `sugi_core.py` for rewrite/plant before **code-derived**, `1` in `insight_common.py` after P2-2 **measured**.
**Implement:** Added `base_url=os.getenv("OLLAMA_HOST_INSIGHT")` to both `rewrite_model` and `_plant_extract_model` **code-derived**.
**Measurement:** Rule-based rewrite handled `Bagaimana cara merawatnya?` without Qwen fallback **measured** (`13.7s` total, `562 chars`), plant fallback `dragon fruit → dragonfruit` `20.6s` on `11435` **measured**; no clear latency penalty vs primary, but contention shifts to `11435` insight batches — plausible counterintuitive slowdown per V2b precedent, not yet proven worse; keep and monitor. Backup: revert or dedicated lightweight-utility instance if `11435` contention observed.

### H2(b) — Insight embedding contention
**Files:** `services/vectorCSV.py`/`vectorWeather.py` `OllamaEmbeddings` default host **code-derived**, `services/insight_common.py` `get_rag_context` **code-derived**
**Measurement:** `embed_query` quiet `0.04s` **measured**, during insight `get_rag_context 0.09s` + concurrent `chat embed 0.04s` **measured** (vs old `1-2s` **industry-expectation** from B4, now `0.04s` via GPU). Not worth isolating — cheap, no measurable delay to chatbot embed during insight window. No fix.

### H3 — Act on H1
**Branch:** Generation slightly dominant overall (2/4) + 2/4 mixed, not prefill-dominant — **Branch B**. `num_predict 512→400` already applied in P3-1 **code-derived**; no further template trimming or cap `<8` without eval-flag evidence (faith LOW recurring is retrieval relevance, not latency **code-derived** noted out-of-scope). Flag distilled/smaller model for simpler queries as Phase 5 candidate **industry-expectation**, not built now. Validation: `padi 15.2s`, `pupuk 18.6s`, `cabai 11.0s` none truncated after `400` **measured**.

---

## Phase 5 — I1 Full-Pipeline + Final Disposition (2026-09-07)

### I1 — Full-pipeline stage breakdown
**Files:** `core/sugi_core.py:438` `STAGE_TIMING` instrumentation **code-derived** (`scope/rewrite/plant/retrieval/rerank/generation` via `time.monotonic` matching existing `[TIMING]` pattern **code-derived**)
**Verification:** H1 bypassed `rewrite/plant/Perenual` per its own description (`raw /api/generate` **code-derived**); production logs `melon 24-51s` vs H1 `1.6-13.4s` gap **measured** signaled missing stages.
**Measurement:** Real production queries via instrumented `ask()` **measured**:
- melon pest `Sayakan mau panen melon...` (direct name-map hit, cache hit, no Qwen fallback **code-derived** per log) `scope 0.025s rewrite 0.000008s plant 5.08s retrieval 2.07s rerank 0.88s generation 19.44s total 27.52s` **measured** — generation `70%` dominates, plant still `5s` even on cache hit (includes `plant detection + pest/disease cache` path), retrieval `2s` consistent with historical `<1s` **measured** plus rerank; total `27.5s` matches production `24-27s`.
- weather `cuaca hari ini...` `scope 0.0004s rewrite 0.000007s plant 1.13s retrieval 2.78s rerank 1.34s generation 4.98s total 10.24s` **measured** — generation `48%`, plant+retrieval+rerank `5.25s` non-trivial.
- salak (no plant) `scope 0.014s plant 0.038s retrieval 1.31s rerank 0.25s generation 9.74s total 11.37s` **measured** — generation `85%`.
- pepaya `scope 0.002s plant 1.19s retrieval 0.76s rerank 0.48s generation 6.24s total 8.69s` **measured** — generation `71%`.
**Secondary:** `GET /api/ps` polled `1s` alongside **measured** — `sugi 2.54GB`, `embed 0.61GB` stay resident (`expires_at` rolling, no eviction) **measured** across all 4 queries; `qwen` not loaded (no fallback triggered) **measured**; tight RAM `0.57GB free` **measured** during dual load did not cause eviction in this window, so melon `19.4s` generation not explained by reload (would show `load_tensors` or missing model). H2(a) redirect already in place, but melon plant `5.08s` still on `11435` insight instance — if insight was concurrently batching, that `5s` could reflect `11435` contention, connecting to H2(a) ambiguous decision; needs per-query correlation with `11435` busy, not yet proven.
**Conclusion:** Missing pipeline stages (plant `0.03-5s`, retrieval `0.76-2.78s`, rerank `0.25-1.34s`) explain part of H1 gap, but generation `4.98-19.44s` remains dominant for `3/4` queries (`48-85%`). For melon, generation alone `19.4s` exceeds H1 raw `6s` for same token cap `400` — points to model-state/wrapper overhead or prompt-cache miss rather than Perenual, for this specific type. Mix of explanations, not single cause — as spec required, not picking convenient one. Backup 2-marker split would have missed plant `5s` outlier.

### Final Disposition — Chroma sharding / Redis stateless / Kubernetes
**Evidence base per spec:**
- V1b: hard ceiling artifact **measured** (`10/10` wall `45-50s` zero failures both harnesses) — degrades, not fails.
- V2b: cheapest lever negative **measured** (`121s` vs `27s` worse).
- G1: GPU already active `100% VRAM 4.31GB` **measured** on primary, modest gain `15.2s` vs `18.9s` **measured**.
- G2: dual isolation net neutral `14.6s vs 14.7s` **measured**, headroom `~2.5GB` **code-derived** corrected.
- H1/H2: context not dominant, embeddings cheap `0.04s` **measured**.
- Traffic `~5 req/hour` every phase **measured** never near `5+` concurrent degradation curve.
**Decision:** **Do not build** Chroma sharding (retrieval `<1s` **measured** every phase), Redis stateless (no replica need **code-derived**), Kubernetes/multi-replica (single-script 7 services reliable **code-derived**). This is a considered decision, not deferral — investment does not match current or foreseeable scale.
**Triggers (observable, unambiguous):**
1. Real sustained traffic regularly produces `2-3+` simultaneous concurrent requests — genuine overlapping `queries.jsonl` timestamps **measured**, not `5/hour` total.
2. Chroma collection `10×` growth AND retrieval latency directly measured degraded — not just bigger.
3. Second deployment target needed for business reason (geo redundancy, second interface sharing state) — not speculative.
Until one fires, this work stays off roadmap. This doc is rationale.
