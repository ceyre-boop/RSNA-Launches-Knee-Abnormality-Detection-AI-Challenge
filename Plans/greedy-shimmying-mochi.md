# Plan: Multilingual Report → Label Extractor (Rules v0 + LLM v1)

## Context

Only 58 of 4,407 training studies carry expert labels; 4,349 have only a free-text
radiology report, and the hidden test set is images-only. Pseudo-labels extracted from
reports are the primary training signal for every downstream image model — extraction
quality caps the whole solution (ISA ISC-31/32). Reports span 11+ languages (~40%
English, then Turkish, Greek, Bulgarian, Spanish/Italian/Portuguese, German, French,
Croatian/Bosnian, Dutch, rare Thai). The gold-58 spans most of them → valid eval set.

Approved approach: **Rules v0 + LLM v1** — deterministic multilingual lexicon/negation
engine now, Claude-API engine behind the same interface (disk-cached, resumable,
batched), disagreements exported as an audit queue.

## Reuse (existing code)

- `src/rsna_knee/constants.py` — `TARGET_LABELS`, `STUDY_ID_COLUMN` (key off these; `Baker's` has an apostrophe)
- `src/rsna_knee/metrics.py` — `binary_roc_auc` (raises ValueError on single-class → catch per label, report None), `macro_auc`
- `src/rsna_knee/tracker.py` — `append_run` (keyword-only)
- `src/rsna_knee/cli.py` — argparse subcommand pattern; extend, don't restructure
- train.csv has embedded newlines in reports — always read with pandas

## New modules

```
src/rsna_knee/extraction/
    types.py          # LabelEvidence(score,status,evidence), ExtractionResult, Extractor protocol
    language.py       # script check (Cyrillic/Greek/Thai) + Latin stopword vote (en,es,tr,it,pt,de,fr,hr,nl); unk → union matching
    sections.py       # technique/findings/impression splitter; technique+clinical-question text EXCLUDED from matching
    lexicon.py        # Term entries {label,lang,pattern,weight} ~150 total; NEGATION_CUES + UNCERTAINTY_CUES per language; NFKD accent-normalized matching
    rules_engine.py   # Engine A
    llm_engine.py     # Engine B: anthropic SDK, claude-haiku-4-5, strict JSON schema (status enum per label), cache data/llm_cache/<uid>.json with prompt_version, live + Batches modes, cost estimate + --yes gate
    merge.py          # LLM wins when non-error; disagreements → data/extraction_audit.csv; provenance column
    io.py             # read train.csv, gold-58 mask, write pseudo_labels.csv
src/rsna_knee/cli.py  # + extract-labels, eval-extraction
tests/test_extraction_{language,rules,llm,merge,cli}.py
```

## Key design decisions

- **Both engines emit statuses** (affirmed/negated/uncertain/not_mentioned) mapped through one
  `SCORE_MAP = {affirmed:0.90 (0.95 if in impression), uncertain:0.60, not_mentioned:0.35, negated:0.05}`
  — AUC is rank-based, so bin ordering matters more than calibration; constants exposed for gold-58 sweeps.
- **Sentence-scope negation** (not fixed windows): Turkish negation is suffixal at sentence
  end (`saptanmadı`, `izlenmedi`, `normaldir`); Romance negation is post-posed
  (`sin evidencia de`, `de morfología conservada`). Impression overrides findings on conflict.
- **Normalcy assertions count as negation** ("menisküsler normaldir", "intact", "conservado").
- **Ambiguity rules encoded identically in lexicon and LLM prompt**: chondral terms map to
  compartment OA by location keyword; marrow edema → Contusion only in traumatic context;
  AVN is no label — keeps engine disagreements meaningful.
- **LLM cost**: ~$10 live / ~$5 via Message Batches API for all 4,407 (Haiku); gold-58 smoke ≈ $0.15.
  Resumable via cache; refusal/parse failure → error sentinel + rules fallback for that study.
- Add `anthropic>=0.60` to pyproject (only new dependency; no spaCy/langdetect).

## Implementation sequence

1. `types.py` + `language.py` + `sections.py` + their tests → pytest green.
2. `lexicon.py` seeded for en/es/tr → `rules_engine.py` + tests (positive/negated/not-mentioned × 3 languages, section-weighting, technique-exclusion cases).
3. `io.py` + `merge.py` + CLI wiring → run rules engine on real gold-58 → `eval-extraction` → first tracked baseline (`model=rules-v0`, split=gold58).
4. Expand lexicon to remaining languages guided by per-language coverage stats + false-negative audit; sweep SCORE_MAP not_mentioned ∈ {0.25,0.35,0.45} on gold-58.
5. `llm_engine.py` + mocked tests → live smoke on gold-58 (~$0.15) → compare per-label vs rules.
6. If LLM ≥ rules: full batch run (~$5, needs ANTHROPIC_API_KEY + explicit --yes), merge → `data/pseudo_labels.csv` for all 4,407 + `extraction_audit.csv`; final tracked eval (`model=merged-v1`).

## Verification

- `uv run pytest` green at each step (5 new test files, no network — LLM client mocked).
- `uv run python -m rsna_knee.cli extract-labels --engine rules --only-gold` then
  `eval-extraction` prints per-label accuracy/F1/AUC table and appends a tracker row —
  the gold-58 macro-AUC is the acceptance number for ISA ISC-31.
- Sanity summaries printed at extraction: per-language counts, per-label status
  distribution, count of empty/truncated reports.
- Audit CSV spot-review of 20 disagreement rows before accepting merged labels.

## Risks (accepted for v0)

Gold-58 is tiny/high-variance (report F1+accuracy alongside AUC; only tune 4 constants
against it); multi-clause Turkish sentences may mis-negate (LLM covers); truncated
reports honestly emit 0.35 flat and get counted; cache `prompt_version` must bump on any
prompt/schema edit.
