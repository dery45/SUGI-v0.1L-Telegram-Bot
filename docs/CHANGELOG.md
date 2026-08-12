# SUGI AI — Changelog

> **Project:** SUGI v0.1L — Intelligent Agricultural Assistant (Indonesia)
> This file is the **versioned changelog** for all code & system changes.
> Companion overview: [`VERSIONS.md`](VERSIONS.md).

## How versions are assigned

The repository has **no Git tags and no `__version__` markers**. Product
branding is controlled in `README.md` ("SUGI v0.1L"). Every version number
below is a **changelog milestone** reconstructed from `git log`
(commit date + set of changes). See the [Git history index](#git-history-index)
at the bottom for the commit → version mapping.

Legend for source of each entry:
- **[git]** — reconstructed from commit messages + `git show --stat`.
- **[inferred]** — inferred from the current codebase / working tree (not
  verifiable from history).

---

## `v0.2.0` — 2026-08-12 — Optimization, latency & correctness hardening [git]

**Status:** committed & pushed to `origin/main` as `f53db8e`
("v0.2.0 Improve optimization and faster perfomance and responses").

### Summary
Performance, latency, and correctness hardening of the RAG / eval pipeline,
plus a scope-guard fix.

### Detailed changes
- **Background eval loop (C1)** — `core/sugi_core.py`
  Eval + `commit_trace` extracted into a new `_run_eval_and_commit()` method
  invoked on a **daemon thread** from `ask()`, so the user-facing reply no
  longer waits on the LLM evaluation. Error path still commits synchronously.
  Reason: remove LLM-eval latency from the chat response path.
- **Model residency (C2)** — `core/sugi_core.py`, `core/eval_loop.py`
  `keep_alive` unified to `600` seconds (10 min) on `self.model`,
  `self.rewrite_model`, and `_eval_model` to avoid per-query model reload.
- **Eval/stability guard (C3)** — `core/sugi_core.py`, `core/eval_loop.py`
  Main model: `num_ctx=4096` + `num_predict=512` (token-output ceiling).
  Eval model: `num_ctx=4096`, request `timeout=15` via `client_kwargs`
  (a bare `timeout=` kwarg is silently swallowed by langchain-ollama 1.0.1).
- **Parallel plant detail fetch (C4)** — `core/plant_api.py`
  Sequential `species_list[:5]` loop replaced with
  `ThreadPoolExecutor(max_workers=3)` + `as_completed`. Rate limiting is
  preserved by the existing global `_ApiQueue` spacing.
- **Scope guard fix — `penanaman` blocked** — `config/settings/scope_config.ini`
  "rekomendasi penanaman untuk bulan agustus" was wrongly rejected because
  `_word_match` (`core/sugi_core.py`) only handles suffixes, not prefixes.
  Added `penanaman`, `menanam`, `menanamkan`, `ditanam`, `ditanamkan`,
  `bertanam` to `[allowed_pertanian]`.
- **Docs** — `.gitignore` now excludes `docs/` (internal only); `docs/`
  changelog & version files created; `README.md` DB-schema section updated
  (ChromaDB 4 → 6 collections, MongoDB read/write note).
- **Repo hygiene — `tests/` git-ignored** — `.gitignore`
   `tests/` now ignored (all verification scripts incl. `test_stage3_*`,
   `test_stage4_*` are internal-only); previously-tracked
   `tests/test_scope_leak.py` + `tests/verify_fix.py` untracked via
   `git rm --cached`.
- **Repo hygiene — `docs/CHANGELOG.md` tracked** — `.gitignore`
  `docs/*` now ignores all docs **except** `docs/CHANGELOG.md`
  (`!docs/CHANGELOG.md`), so the versioned changelog is committed while
  the rest of `docs/` remains internal-only.
- **A9 — multi-turn history poisoning fix** — `core/sugi_core.py`
  In a session, unrelated follow-on questions ("rekomendasi tanaman terbaik
  di tanam bulan desember", "cara menanam salak") answered with the topic of
  the PRIOR turn (e.g. apple) because the full past answer was injected into
  `{history}` and the template instructs using history as primary reference.
  Repro (live, `SUGI_DEBUG_ASK`): `standalone_query` + retrieved context were
  correct — the 3.2B model was anchored by the old-answer text. Fix: history is
  now injected into the prompt **only** when the rewriter detects a referential
  signal (`rewrite_type != "none"`) via new `_needs_ref_context()` /
  `_select_prompt_history()`; self-contained questions get
  `"Belum ada riwayat percakapan."`. Template also forbids echoing the
  `User:`/`Sugi:` dialog format and claiming missing data when data was given.
  Verified live (December/Salak answers no longer bleed apple) + unit test
  `tests/test_stage5_a9.py` (16 checks).
- **A8 — insight-engine startup stagger + LLM timeout** —
  `services/daily_insight.py`, `services/government_insight_service.py`,
  `services/farmer_insight_service.py`, `config/.env.example`
  All three insight engines shared one `STARTUP_GRACE_SECONDS` env var (default
  180s) and were launched together by `start_all.py`, so they woke at the same
  time and contended for Ollama (plus farmer/gov LLM `client_kwargs` timeout was
  a tight 90s; daily had none). Fix: per-service grace vars with staggered
  defaults — Daily 180s, Government 240s, Farmer 300s (fallback:
  per-service var → shared `STARTUP_GRACE_SECONDS` → per-service default) — and
  insight-only LLM timeout raised to 240s in all three engines. Unit test
  `tests/test_stage5_a8.py` (15 checks).
- **A10 — retrieval-level memory leak + A4 closure** — `core/sugi_core.py`
  A9 gated the prompt's `{history}` text, but the retrieval path still branched
  on the coarser `has_history` flag ("any prior conversation") in three places,
  so self-contained new questions ("cara menanam salak") with ANY session
  history still pulled old-topic memory content (e.g. labu siam from two turns
  earlier) into the retrieved context via the memory store. Fix: all three
  sites now gate on `needs_ref_context` (rewrite_type != "none"):
  `_build_retriever()` call site, the `mem_future` similarity-search gate, and
  the memory-doc merge into `all_docs`. A4 (deferred since Stage 2): the shared
  unfiltered `self.memory_retriever` (no `user_id` filter, cross-user privacy
  leak) is removed from `__init__` and the ensemble — only the correctly scoped
  `mem_future` path (filter={user_id}, gated by `needs_ref_context`) remains.
  Verified with unit test `tests/test_stage5_a10.py` (19 checks): salak/December
  suppress memory injection, referential follow-ups still fire; A9 (16 checks)
  and O3 (14 checks) still pass.
- **A11 — startup 429 NameError + single-pass rerank** — `core/plant_api.py`,
  `core/sugi_core.py`, `core/eval_loop.py`
  (a) `plant_api.py`: the module-level API-key startup validation calls
  `_trip_rate_limit()` on a 429, but that function was defined FURTHER DOWN in
  the file — a real 429 (seen in the melon/watermelon storm) raised `NameError`
  at import time, which `SugiCore.__init__`'s `except ImportError` cannot catch,
  killing the whole bot. Fix: moved the rate-limit flag block (flag + lock +
  `_is_rate_limited` + `_trip_rate_limit`) above the validation block — pure
  reorder, no dependency on `_ApiQueue`. Unit test `tests/test_stage5_a11.py`
  (3 checks) mocks a 429 on startup and asserts a clean import + tripped
  cooldown. (b) Double reranking removed: `_build_retriever` returned a
  `ContextualCompressionRetriever`, so RAG docs were cross-encoder-scored twice
  on weather queries (once in the wrapper, once at the weather+RAG merge). The
  builder now returns the raw ensemble and ALL candidates are reranked exactly
  once at the merge point — with or without weather. (c) Dead code removed:
  `include_weather` param/branch and `self.weather_retriever` (vestigial since
  M1's manual fetch-merge-rerank). (d) `_eval_model` gained `num_predict=10`
  (consistent with the other three utility models).
- **A12 — daily_insight Ollama-concurrency + ping-retry backfill** —
  `services/daily_insight.py`
  The three daily insight generators still used
  `ThreadPoolExecutor(max_workers=MAX_WORKERS)` (hardcoded 5) + `ex.map(...)`,
  firing up to 5 simultaneous qwen2.5:1.5b calls per group loop (province/
  location/commodity) — the same burst-concurrency that caused A8's original
  "[LLM error] timed out / LLM returned empty", even standalone in a single
  `run_once()`. Fix: replaced all three pools with `_run_paced()` — sequential
  by default (`DAILY_INSIGHT_MAX_WORKERS`, default `1`) with `INSIGHT_LLM_DELAY`
  pacing between calls (same pattern as farmer/gov services); a value >1 is an
  explicit, env-driven choice that restores the thread pool. Also backfilled the
  A7 fix this service skipped: startup Mongo ping now goes through
  `_ping_with_retry` (3 attempts, 5s/10s linear backoff) instead of a bare
  unretried `admin.command("ping")` → `sys.exit(1)`.

### Files affected
`.gitignore`, `README.md`, `config/settings/scope_config.ini`,
`config/.env.example`, `core/eval_loop.py`, `core/plant_api.py`,
`core/sugi_core.py`, `services/daily_insight.py`,
`services/farmer_insight_service.py`,
`services/government_insight_service.py`,
`docs/CHANGELOG.md`
(+ runtime-data drift: `.weather_cache.sqlite`, `data/telegram_offset.json`).

---

## `v0.1.9` — 2026-07-13 — Government & Farmer Insight Engines [git]

### Summary
Master-data insight engines added: AI-generated strategic insights learned
across datasets and stored back into ChromaDB / MongoDB.

### Detailed changes
- **Government Insight Engine** — `services/government_insight_service.py`
  (new, ~673 lines). Insight generation over 14 government datasets
  (prices, food security, distribution) with a ChromaDB learning loop;
  `insightVersion` incrementing on regeneration.
- **Farmer Insight Engine** — `services/farmer_insight_service.py`
  (new, ~1288 lines). 10 farmer insights (PPH score, best commodity,
  positive margin, monthly plantation/sell rekomendasi, etc.) + 1 policy
  recommendation; example-driven prompting, multi-stage Bahasa validation,
  retry, RAG/weather/plant/memory enrichment; daily change detection.
- `start_all.py` — both services registered as background loops.
- `README.md` — feature + DB schema docs (ChromaDB collections
  `government_memory`, `insights_memory`; MongoDB output).

### Files affected
`services/government_insight_service.py`, `services/farmer_insight_service.py`,
`start_all.py`, `README.md`.

---

## `v0.1.8` — 2026-06-03 — Documentation refresh [git]

### Summary
README overhaul with live-demo QR and chatbot screenshots.

### Detailed changes
- `README.md` — expanded install/usage docs; image/ previews added
  (`QR-Link-to-try-sugiai.png`, `chatbot-answer-*.png`).

### Files affected
`README.md`, `image/*.png`.

---

## `v0.1.7` — 2026-04-13 — Telegram offline message catch-up [git]

### Summary
Messages sent to the bot while it was offline are now processed on restart.

### Detailed changes
- `interfaces/telegram/telegram_bot.py` — pending-message queue on startup,
  offset persisted crash-safely (`data/telegram_offset.json`).
- `services/daily_insight.py` — large rework (~480 lines changed) as part of
  the offline/restart flow.
- `test_mongo.py` — added for MongoDB connectivity sanity.

### Files affected
`interfaces/telegram/telegram_bot.py`, `services/daily_insight.py`,
`data/telegram_offset.json`, `test_mongo.py`.

---

## `v0.1.6` — 2026-03-27 — Insight pipeline API fixes [git]

### Summary
Bug fixes across insight services and the health check entrypoint.

### Detailed changes
- `start_all.py` — added health-check wiring (+34 lines).
- `services/daily_insight.py`, `services/vectorCSV.py`,
  `services/vectorWeather.py`, `services/vectorpdf.py` — API/endpoint fixes.
- `interfaces/telegram/telegram_bot.py` — minor fixes.

### Files affected
`start_all.py`, `services/*`, `interfaces/telegram/telegram_bot.py`.

---

## `v0.1.5` — 2026-03-26 — Thread safety, streaming, API validation [git]

### Summary
Concurrency hardening for the Telegram interface and parent (Perenual) API
validation.

### Detailed changes
- `interfaces/telegram/telegram_bot.py` — thread-safety fixes for concurrent
  user sessions; streaming response handling.
- `core/sugi_core.py` — thread-safety around eval/logging state.
- `core/plant_api.py` — parent API (Perenual) validation on requests.
- `config/settings/rewriter_config.ini` — rewriter pattern additions.

### Files affected
`interfaces/telegram/telegram_bot.py`, `core/sugi_core.py`,
`core/plant_api.py`, `config/settings/rewriter_config.ini`.

---

## `v0.1.4` — 2026-03-25 — Context-leak fix, chains Q&A, memory & health [git]

### Summary
Cross-commit stabilization: definitional query detection, scope gating,
memory bug fix, and startup health checks.

### Detailed changes
- **Chains Q&A + context leaks** (`ca6b945`) — `core/sugi_core.py` (+188)
  3-layer rewrite logic, definitional-query detection ("apa itu X"),
  rewrite-type scope gating; `config/settings/scope_config.ini` +11 keywords;
  `services/vectorCSV.py|vectorWeather.py|vectorpdf.py` retrieval tweaks;
  tests added (`tests/test_scope_leak.py`, `test_scope_fix.py`,
  `tests/verify_fix.py`); `generate_questions.py` updated.
- **Memory bug + health check** (`802e1c3`) — `core/sugi_core.py` memory
  (`_save_session_memory` path fixes); `start_all.py` adds startup health
  checks (+87).

### Files affected
`core/sugi_core.py`, `config/settings/scope_config.ini`,
`services/vectorCSV.py`, `services/vectorWeather.py`,
`services/vectorpdf.py`, `generate_questions.py`,
`interfaces/telegram/DEPLOYMENT_GUIDE.md`, `start_all.py`,
`tests/test_scope_leak.py`, `test_scope_fix.py`, `tests/verify_fix.py`.

---

## `v0.1.3` — 2026-03-24/25 — Repo hygiene & modular restructure [git]

### Summary
Repository slimmed down, secret/session data ignored, project reorganized
into a modular `core/ services/ interfaces/` layout with `start_all.py`.

### Detailed changes
- Slimmed tracked binary artifacts (removed `bm25_cache.pkl` ~58 MB, PDF
  corpus, large CSVs) and tightened `.gitignore`; added `.env.example`,
  `data/` user files.
- Modular restructure (rename `main.py` → `interfaces/cli/main.py`,
  `telegram_connection/` → `interfaces/telegram/`, `daily_insight.py` →
  `services/`, `eval_loop.py`/`plant_api.py`/`query_logger.py`/
  `sugi_core.py`/`user_store.py` → `core/`, `word_config/` →
  `config/settings/`). New `start_all.py` bootstrap.
- `.gitignore` hygiene for user identity & platform databases
  (`data/users.json`, `data/cli_user.txt`, platform DBs).
- Config now centralised in `config/` (`Modelfile`, `.env.example`).

### Files affected
Move/rename of ~25 files (see `git show 9f79b51`,`16de059`); `.gitignore`,
`config/`, `core/*`, `services/*`, `interfaces/*`, `start_all.py`.
(chore-only commits `2fecbd4`, `cec923a`, `ca89af9`, `c386a3e` also fall into
this version.)

---

## `v0.1.2` — 2026-03-15 — ChromaDB, Qwen rewrite model, daily insights [git]

### Summary
Vector store moved to ChromaDB; utility model changed to `qwen2.5:1.5b`;
daily insight engine + bulk BM25 ingestion added.

### Detailed changes
- **ChromaDB migration** (`e768f29`) — `plant_api.py` (major rework),
  `vectorCSV.py`, `vectorPDF.py`, `vectorWeather.py` repointed to ChromaDB;
  `migrate_to_server.py` added; new food-security CSVs.
- **Model change** (`cf1e3e2`) — eval LLM moved from pip3/Phi-3 class model
  to `qwen2.5:1.5b`; `main.py` (+200), `plant_api.py`, `vectorPDF.py`
  updated.
- **Daily insight + bulk BM25** (`9db4220`, `89c1f8e`) —
  `services/daily_insight.py` (new, ~563 lines), producer-price dataset,
  README, bulk-index path.
- **Separate word config** (`b25d8c2`) — `word_config/plant_keywords.ini`,
  `word_config/rewriter_config.ini`, `word_config/scope_config.ini`
  (later moved to `config/settings/`).

### Files affected
`plant_api.py`, `vectorCSV.py`, `vectorPDF.py`, `vectorWeather.py`,
`migrate_to_server.py`, `daily_insight.py`, `eval_loop.py`, `main.py`,
`word_config/*`, datasets (`data/`).

---

## `v0.1.1` — 2026-03-15 — Rule-based rewriting & eval loop [git]

### Summary
The Phi-3 rewrite model was replaced by a rule-based rewriter with date
logic; answer-quality eval loop (faithfulness/relevance + flagging) plus
query logging introduced.

### Detailed changes
- `main.py` (big rework) — 3-tier rewrite (rule-based → fallback → original);
  date-aware logic.
- `eval_loop.py` (new) — lexical heuristics + LLM faithfulness/relevance
  scoring with LOW → flag.
- `query_logger.py` (new) — per-query trace to `data/logs/queries.jsonl`,
  flagged queries to `data/logs/eval_flags.jsonl`.
- `vectorCSV.py` — CSV ingestion improvements.
- First remote merge (`226b33b`) of `https://github.com/dery45/SUGI-v0.1L`.

### Files affected
`main.py`, `eval_loop.py`, `query_logger.py`, `vectorCSV.py`.

---

## `v0.1.0` — 2026-03-15 — Initial release scaffolding [git]

### Summary
First commits: base RAG chatbot for Indonesian agriculture, custom Ollama
Modelfile, document ingestion and question generation.

### Detailed changes
- Initial commit (`e114ac9`, `c263e70`) — `main.py` (~460 lines), `Modelfile`
  (persona), `.gitignore`, PDF corpus (`pdfsource/`), commodity & food-security
  CSVs, `generate_questions.py` + `generated_questions.txt`.
- README with project details & installation steps (`a6e0784`); contribution
  section removed (`bccfd58`).

### Files affected
`main.py`, `Modelfile`, `generate_questions.py`, `data/`, `pdfsource/`,
`.gitignore`, `README.md`.

---

## Appendix: reconstruction notes

- **Version numbers** are milestone labels assigned for this changelog; the
  repository itself uses only the product name **v0.1L** (README) and has no
  tags/branches other than `main` (tracking `origin/main`).
- **Unverifiable / inferred** items: entries marked **[inferred]** above;
  the exact change-set of `v0.1.2`'s "daily insight" milestone (two commits
  share the same message), and details inside the large binary/artefact
  cleanup in `c376fd5`.

## Git history index (commit → version, reconstructed)

```
2026-08-12  f53db8e                       → v0.2.0
2026-07-13  cd6f42b / 6fb410c / cc94e4c  → v0.1.9
2026-06-03  d8b10fa                       → v0.1.8
2026-04-13  d8245b1                       → v0.1.7
2026-03-27  2cfe39e                       → v0.1.6
2026-03-26  c608d88                       → v0.1.5
2026-03-25  ca6b945 / 802e1c3             → v0.1.4
2026-03-25  16de059 / 9a78adb / 081303e /
            9f79b51 / cec923a / 2fecbd4 /
            60d8bf0 / c386a3e / ca89af9   → v0.1.3
2026-03-24  c376fd5 / 3eee738             → v0.1.3
2026-03-15  9db4220 / 89c1f8e / cf1e3e2 /
            e768f29 / b25d8c2 / 226b33b   → v0.1.2
2026-03-15  3659690 / bccfd58 / a6e0784   → v0.1.1
2026-03-15  e114ac9 / c263e70             → v0.1.0
```