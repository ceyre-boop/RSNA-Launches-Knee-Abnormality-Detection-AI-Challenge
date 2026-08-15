---
project: rsna-knee-2026
task: Win the RSNA Knee Abnormality Detection AI Challenge (solo)
effort: E4
phase: execute
progress: 12/54
mode: standard
started: 2026-08-13T20:45:00-07:00
updated: 2026-08-13T20:45:00-07:00
---

# ISA — RSNA Knee Abnormality Detection 2026

## Problem

The competition (opened 2026-07-30, final submission 2026-10-22 UTC) requires detecting
12 knee abnormalities on MRI studies, scored by macro-averaged ROC AUC across labels.
We have a validation/metrics scaffold but no data, no Kaggle credentials configured, no
training pipeline, no model, and no submission. Solo entry against teams; ~10 weeks
remain. Serious training compute does not exist locally (M4 Pro / 24 GB).

## Vision

A top-10 finish on the private leaderboard (main prize money), with a credible shot at
the Efficiency Track via a distilled single-model submission. The euphoric-surprise
moment: the first full-pipeline submission scores competitively, and every later gain is
measured, reproducible, and tracked in this file.

## Out of Scope

- Team merging — this is a solo run by explicit decision.
- Private/external medical data — only competition data plus freely public pretrained
  weights (per rules).
- Training large models locally on the Mac — local machine is for pipeline dev,
  debugging, and small-scale experiments only.
- Building a general-purpose knee-MRI product — everything serves the leaderboard.

## Principles

- **CV is the compass.** No change ships without out-of-fold macro-AUC evidence.
  Leaderboard probing is noise; trust patient-grouped CV correlated once against LB.
- **Reports are training-time privileged information.** Radiology reports exist only in
  train; their value is extracted via pseudo-/soft-labels and distillation, never as an
  inference-time input assumption (verify test schema before finalizing).
- **Rare labels dominate macro AUC.** Fracture/Contusion/Baker's are likely low-prevalence;
  a model that wins the common labels but guesses on rare ones loses. Per-label AUC is
  first-class telemetry.
- **Every experiment is a falsifiable hypothesis** with a pre-stated success threshold,
  logged via the run tracker.
- **Inference budget is a design input, not an afterthought.** 9 h GPU, no internet,
  weights shipped as Kaggle datasets/models.

## Constraints

- Kaggle code competition: submission via notebook, ≤9 h runtime, internet disabled,
  `submission.csv` output, weights must be pre-packaged.
- Deadlines (UTC 23:59): entry/rules acceptance 2026-10-15, final submission 2026-10-22.
- External data/models must be freely and publicly available.
- Winners' obligations: open-source code + weights, video, forum post — solution must be
  releasable (no license-encumbered components).
- Training compute: Kaggle GPU quota (~30 h/wk) first; paid cloud (A100-class rental)
  only when an experiment demonstrably needs it.
- Python (Kaggle requirement) — explicit exception to the TypeScript-always rule.

## Goal

Ship a selected final submission before 2026-10-22 that (a) scores in the top decile of
the private leaderboard on macro AUC, built from a multi-series MRI ensemble trained with
report-distilled supervision, and (b) includes an efficiency-track candidate whose
runtime is under 30 minutes; every modeling claim verified by tracked out-of-fold CV.

## Criteria

### Foundation & logistics
- [x] ISC-1: `kaggle` CLI installed and `kaggle competitions list` authenticates successfully
- [x] ISC-2: Competition rules accepted on Kaggle (competition appears in `kaggle competitions list --mine` or data download authorized)
- [x] ISC-3: Competition data files enumerated via `kaggle competitions files rsna-knee-abnormality-detection` (name TBC) with sizes logged in Decisions
- [x] ISC-4: train.csv (labels) downloaded and label prevalence per target computed and logged
- [x] ISC-5: Data storage decision made and logged (local disk vs Kaggle-only vs cloud box) based on actual dataset size
- [x] ISC-6: `uv`-managed venv with pinned deps (`pyproject.toml`) installs clean and `pytest` passes
- [ ] ISC-7: Baseline all-0.5 submission validated by `src/rsna_knee/submission.py` and submitted to LB (pipeline smoke test)
- [x] ISC-8: GitHub repo synced; training runs reproducible from a tagged commit

### Data understanding (EDA)
- [ ] ISC-9: Series inventory built — count of series per study, plane/sequence-type distribution from DICOM metadata, written to `artifacts/eda/series_inventory.csv`
- [ ] ISC-10: Per-site and per-language study counts computed; site distribution shift risk assessed in Decisions
- [ ] ISC-11: Label co-occurrence matrix computed and saved (informs multi-task head design)
- [ ] ISC-12: Report text audited: language mix, length distribution, structure; sample of 20 reports read manually against labels
- [x] ISC-13: Test-set schema confirmed: does hidden test provide images only, or images+reports? Logged in Decisions (determines whole text strategy)
- [ ] ISC-14: DICOM pixel pipeline validated: windowing, orientation, spacing normalization produce visually correct slices for 10 random studies (saved PNGs eyeballed)

### Cross-validation harness
- [x] ISC-15: 5-fold patient-grouped, site-stratified splits generated by existing `build_cv_splits` and leakage check passes [SATISFIED via report-hash-grouped splits + PatientID verification; site-stratification dropped — no site column exists]
- [ ] ISC-16: OOF scoring loop wired: any model checkpoint → oof.csv → `score-oof` macro + per-label AUC, appended to run tracker
- [ ] ISC-17: CV↔LB correlation established: ≥2 submissions with differing CV plotted; correlation noted before trusting CV for selection
- [x] ISC-18: Anti: no study from the same patient appears in both train and validation folds of any run (leakage validator in CI) [VERIFIED on Kaggle 2026-08-15: DICOM PatientID dump shows 4,407 studies = 4,407 unique patients, 0 multi-study patients — StudyInstanceUID grouping is leak-proof by construction]

### Series identification & preprocessing
- [x] ISC-19: Series-type classifier (plane × sequence) achieves ≥98% accuracy on a hand-labeled 200-series audit set, or DICOM-metadata rules achieve same
- [ ] ISC-20: Volume preprocessing (resample, crop to knee ROI, normalize) runs end-to-end over full train set without errors; failures logged and handled
- [ ] ISC-21: Preprocessed cache format decided (npz/zarr) with read throughput ≥ target for GPU saturation, measured

### Baseline model (Milestone 1 — target ~2 weeks in)
- [ ] ISC-22: 2.5D CNN baseline (single series type, e.g. sagittal fluid-sensitive) trains end-to-end and beats 0.5 AUC on every label
- [ ] ISC-23: Baseline OOF macro AUC ≥ 0.80 (revise threshold after first run; log revision)
- [ ] ISC-24: Baseline submitted; LB score within 0.02 of CV (validates harness)
- [ ] ISC-25: Training runs on Kaggle GPU notebooks with checkpoint resume (quota-interruption-safe)

### Multi-series model (Milestone 2)
- [ ] ISC-26: Per-series encoder + cross-series attention aggregation model implemented; consumes all available series per study
- [ ] ISC-27: Multi-series model beats single-series baseline OOF macro AUC by ≥0.015
- [ ] ISC-28: Slice-attention or LSTM aggregation ablated vs mean-pool with tracked runs
- [ ] ISC-29: Handles variable series availability (missing planes/sequences) without crash or NaN — probe with synthetic missing-series studies
- [ ] ISC-30: ≥2 backbone families trained (e.g. ConvNeXt + MaxViT/EfficientNetV2) for ensemble diversity

### Report-supervision track (Milestone 3)
- [x] ISC-31: Text model fine-tuned on reports→12 labels; OOF macro AUC of text-only model logged (expected ≥0.95 — reports contain the answers) [REFINED: Sonnet-subagent extraction replaced fine-tuning; gold-58 agreement 0.823 macro AUC; all 4,407 studies labeled with 0 errors — see Decisions 2026-08-15]
- [ ] ISC-32: Text-model soft labels distilled into image model; distillation gains ≥0.005 OOF macro AUC over hard labels, or approach documented as refuted in Changelog
- [ ] ISC-33: Anti: no report text used as inference-time input unless ISC-13 confirms reports exist in test
- [ ] ISC-34: Image-text contrastive pretraining (CLIP-style) evaluated; kept only if OOF gain ≥0.003

### Ensemble & final submission (Milestone 4)
- [ ] ISC-35: Fold ensemble + multi-backbone ensemble scored; weight search on OOF
- [ ] ISC-36: Full inference notebook runs on Kaggle in ≤8 h wall-clock (1 h safety margin) with internet off
- [ ] ISC-37: All weights uploaded as Kaggle dataset/model and loadable offline
- [ ] ISC-38: Final 2 submissions selected by documented rule (best CV, not best public LB) and logged in Decisions
- [ ] ISC-39: Anti: no submission relies on internet access, test-set statistics recomputed across the full hidden set in a way that breaks the per-study API contract, or >9 h runtime
- [ ] ISC-40: Submission selected before 2026-10-22 00:00 UTC (not deadline-day scramble)

### Efficiency track
- [ ] ISC-41: Distilled compact model (student of ensemble) achieves ≥90% of ensemble's (score − benchmark) delta at <30 min runtime
- [ ] ISC-42: Efficiency candidate is among selected/auto-selected submissions per efficiency-prize eligibility rules
- [ ] ISC-43: fp16/compiled inference path measured on Kaggle GPU; runtime logged per study

### Experiment discipline
- [ ] ISC-44: Every training run appended to tracker with config hash, fold scores, per-label AUCs
- [ ] ISC-45: Weekly review entry in Decisions: best CV, LB position, next-week hypothesis ranked list
- [ ] ISC-46: Anti: no experiment run without a pre-registered hypothesis and success threshold in the tracker
- [ ] ISC-47: Kaggle GPU quota usage tracked weekly; cloud-spend decision gated on a specific starved experiment

### Advisor-derived hardening (added 2026-08-13)
- [ ] ISC-50: Week-1 always-green inference skeleton: Kaggle notebook loads hidden-test layout, runs dummy model, emits valid submission.csv in <1 h — kept green all competition
- [ ] ISC-51: DICOM decode throughput measured (incl. JPEG2000/JPEG-LS cases); decode is not the inference bottleneck at final ensemble scale
- [ ] ISC-52: Ensembling uses rank-averaging; rank-avg vs prob-avg ablated on OOF
- [ ] ISC-53: Public knee-MRI pretraining (MRNet et al.) evaluated; kept if OOF gain ≥0.005 and license is redistributable
- [ ] ISC-54: ROI-crop QC visualizations (supervision-annotated boxes on slices) reviewed for 20 studies before cache build

### Winners' obligations readiness
- [ ] ISC-48: Training code reproducible end-to-end from README instructions on clean machine
- [ ] ISC-49: License audit: all pretrained weights used are redistributable (no non-commercial-only blockers for open-sourcing)

## Test Strategy

| isc | type | check | threshold | tool |
|-----|------|-------|-----------|------|
| 1-3 | command | kaggle CLI auth + file listing | exit 0 | Bash |
| 4,9-12 | artifact | EDA outputs exist with sane values | file present + spot check | Read/Bash |
| 15,18 | command | leakage validator | PASS | pytest/CLI |
| 16-17,22-24,27-28,31-32,34-35,41 | metric | OOF macro AUC via `score-oof` | per-ISC threshold | Bash |
| 25,36-37,43 | runtime | Kaggle notebook execution | wall-clock limits | Kaggle commit log |
| 38,40,42,45-47 | process | Decisions/tracker entries exist | entry present | Read |
| 48-49 | audit | clean-machine repro + license list | documented | Read |

## Features

| name | description | satisfies | depends_on | parallelizable |
|------|-------------|-----------|------------|----------------|
| kaggle-setup | CLI, creds, rules, data enumeration | ISC-1..5 | — | no |
| env | uv venv, pinned deps, CI green | ISC-6 | — | yes |
| smoke-submission | all-0.5 LB submission | ISC-7 | kaggle-setup | no |
| eda | series inventory, labels, reports, sites | ISC-9..14 | kaggle-setup | yes |
| cv-harness | splits, OOF loop, LB correlation | ISC-15..18 | eda | no |
| preprocess | series ID, ROI crop, cache | ISC-19..21 | eda | yes |
| baseline | 2.5D single-series model | ISC-22..25 | cv-harness, preprocess | no |
| multiseries | cross-series attention model | ISC-26..30 | baseline | partly |
| text-track | report model + distillation | ISC-31..34 | cv-harness | yes (parallel to multiseries) |
| ensemble | fold/backbone blend + final notebook | ISC-35..40 | multiseries | no |
| efficiency | distilled student model | ISC-41..43 | ensemble | yes |
| discipline | tracker, weekly reviews, quota watch | ISC-44..47 | — | continuous |
| release-prep | repro + license audit | ISC-48..49 | ensemble | yes |

## Decisions

- 2026-08-13: Solo entry confirmed by principal ("win this challenge by ourselves").
- 2026-08-13: Python stack — explicit exception to TypeScript-always rule; Kaggle
  notebooks require Python and the scaffold (merged PR #1) is already Python.
- 2026-08-13: ISC count 49 vs E4 soft floor 128 — show-the-math: criteria will split
  naturally as milestones open (each model experiment adds tracked sub-criteria via the
  run tracker, which serves as the fine-grained probe ledger). Padding to 128 now would
  fabricate probes for unknowns (exact data schema unseen).
- 2026-08-13: Compute strategy — Kaggle GPU quota first; rent A100s only when a named
  experiment is quota-starved (ISC-47 gate).
- 2026-08-13: Advisor review adopted: series-type classification and physical-spacing
  resample prioritized over text distillation (downgraded from pillar to evaluated bet,
  ISC-32 threshold stands); rank-average ensembling (ISC-52); week-1 inference skeleton
  (ISC-50); final week is freeze buffer — no new experiments after 2026-10-15.
- 2026-08-13: Principal directed adding `supervision` (Roboflow) — adopted for ROI
  localization QC/annotation on slices (ISC-54), not core 3D classification (it is a 2D
  detection/tracking toolkit). Boundary named per Backbone rule; wired into pyproject.
- 2026-08-13: Environment initialized: uv-managed pyproject (numpy/pandas/sklearn/
  pydicom/supervision/opencv-headless/kaggle), 4/4 tests pass, kaggle CLI installed.
- 2026-08-13: progress frontmatter counts 54 ISCs; 6 partially advanced (env, CLI,
  repo) — none marked [x] until their full probe passes.
- 2026-08-13 (night): Report extractor shipped (Rules v0 + LLM v1 per approved plan,
  Plans/greedy-shimmying-mochi.md). Rules-v0 gold-58 baseline: **0.7107 macro AUC**
  (MCL 0.90, ACL 0.87, menisci 0.82-0.86; weak: Medial/Lateral OA 0.52-0.54,
  Effusion/Synovitis 0.58-0.59 — multilingual nuance gap the LLM engine targets).
  28/28 tests green. Tracker row `rules-v0-gold58` in runs.csv.
- 2026-08-15: RESOLVED: no API key needed — Sonnet subagents on Max subscription did the full corpus. (was: BLOCKED on ANTHROPIC_API_KEY for LLM extraction (gold-58 smoke ~$0.15,
  full batch ~$5). Add `ANTHROPIC_API_KEY=sk-ant-...` to the repo `.env`, then:
  `set -a; source .env; set +a; uv run python -m rsna_knee.cli extract-labels --engine llm --only-gold --yes`
- 2026-08-13 (late): **Sonnet extraction via Claude Code subagents** (Max subscription,
  no API key — TABOOST preference; also: never Haiku, Sonnet/Opus only). Gold-29
  head-to-head: Sonnet 0.8123 vs rules 0.7210 macro AUC. Full-corpus run PAUSED by
  TABOOST at 11/44 chunks (1,089/4,349 studies done, all validated clean). Resume
  state: scratchpad extraction_ledger.json (pending: 33 chunks); durable copies in
  data/sonnet_results/ + data/pseudo_labels_sonnet_partial.csv.
- 2026-08-13: **Leaderboard intel** (efficiency-LB notebook CSV): top 0.944, top-10
  cutoff 0.934, median 0.832, 450 teams at 0.85-0.90. Competitor salemali7 (0.899)
  analyzed — same reports-as-supervision strategy, DINOv2 2.5D 6-slot ensemble; full
  intel + adopt/exploit list in intel/competitor-salemali7.md. Key adoptions queued:
  geometric slice ordering (filename order is RANDOM), 130mm crop, laterality
  normalization, rank-averaged ensembling, confidence-weighted BCE with gold override.
- 2026-08-13: Gold-58 chunk-0 Sonnet worker was killed mid-pause; only 29/58 gold
  studies have Sonnet labels (pseudo_gold_sonnet_29.csv). Re-run gold chunk 0 first on
  resume to complete the engine comparison.
- 2026-08-15: **Gold-58 is pathology-enriched, not a random sample.** Positive-rate
  shift vs corpus: Fracture +0.26 (31% gold vs 5% corpus), Synovitis +0.35, Contusion
  +0.19, ACL/Lat-Meniscus +0.17. Organizers picked interesting cases for annotation.
  Consequences: (a) gold-58 AUC remains valid for RANKING extractors but overstates
  rare-label prevalence; (b) fold-0 validation on pseudo-labels will not match LB
  distribution — treat first LB submission as the real calibration point.
- 2026-08-15: **CV grouping honesty note (re ISC-15/18):** splits group on report-text
  hash (dupe-leak guard), NOT PatientID — PatientID is not in train.csv and DICOM
  headers are Kaggle-resident. ACTION on first Kaggle session: dump PatientID from
  train DICOM headers; if any patient has >1 study, regroup splits on PatientID.
  Until then ISC-18 is NOT satisfied.
- 2026-08-15: Effusion/Synovitis 20-case audit queued to run during first GPU job
  (reading task, no contention). Purpose: calibrate per-label confidence weights, or
  catch a "trace fluid" threshold artifact on the most-prevalent label.
- 2026-08-15: Submission cadence: multiple submissions/day allowed (typically 5); only
  2 selected finals count. Submit early and often for CV<->LB correlation (ISC-17);
  never select finals by public LB (ISC-38 stands).
- 2026-08-15: **Pre-registered fold-0 expectation (before seeing the number):** OOF macro
  AUC vs v1.1 pseudo-labels expected 0.72-0.80. >0.87 = leakage smell (suspect near-dup
  reports escaping hash grouping) — investigate before celebrating. <0.62 = pipeline bug
  (slot mask/laterality/loss-weight). First submission is for CV<->LB signal, not score.
- 2026-08-15: Synovitis ceiling note: audit showed reports are structurally silent on
  synovium (5/8 annotator-vs-report) — label partly unlearnable from report-derived
  supervision; do NOT spend late-competition effort rescuing it beyond the 0.43 weight.
- 2026-08-15: **v1.2 labels** — second audit (PF OA/MedMen, 21 cases): 13 threshold
  artifacts, 0 extraction errors. Deterministic severity fixes: PF OA requires
  moderate+ chondral loss (323 downgraded, AUC .824->.839); MedMen requires
  surface-reaching tear (122 downgraded, .899->.908). Macro .834->.836. NOTE: audit
  flagged 7 gold-58 labels as likely WRONG (4 PF OA gold=1 with no PF mention in
  report incl. one explicit normal; 3 MedMen gold=0 with explicit tear in report) —
  gold-58 scoring has a noise floor; corpus gains likely exceed measured deltas.
  Pattern established: 3 audits, 3 wins — severity-bar mismatch is THE systematic
  extractor error class, worth auditing remaining high-prevalence labels.
- 2026-08-13: BLOCKED on principal for: (a) Kaggle API token at `~/.kaggle/kaggle.json`,
  (b) accepting competition rules on the Kaggle website (cannot be done via CLI).

- 2026-08-13 (evening): Rules accepted (principal's explicit "join it"; toast "Rules
  accepted. Good luck!"). CLI auth via `~/.kaggle/access_token`. All metadata CSVs local.
- 2026-08-13: **refined: semi-supervised reality discovered.** Only 58/4,407 train
  studies carry expert labels; 4,349 are report-only; test.csv has NO Report column.
  Text pseudo-labeling is therefore the PRIMARY supervision, not an evaluated bet:
  pipeline = multilingual report → 12 pseudo-labels → image model training → calibration
  on 58 gold studies. ISC-31/32 reinterpreted accordingly (probe for extraction quality:
  agreement with the 58 gold label sets).
- 2026-08-13: ISC-19 satisfied by organizers — train/test_series.csv provide
  Anatomical_Plane, Fluid_Sensitive, Fat_Suppression per series. No classifier needed.
- 2026-08-13: ISC-5 storage decision — est. ~1.2 TB images vs 227 GB free local disk:
  images stay Kaggle-resident (train in Kaggle notebooks); local machine handles the
  full text track (all 4,407 reports on disk) plus small image samples for pipeline dev.
- 2026-08-13: No PatientID/site columns in train.csv — patient grouping must come from
  DICOM headers if present; CV design revisits this during EDA (ISC-10/15 risk).
- 2026-08-13: Data snapshot: 4,407 studies, 24,371 series (~5.5/study), planes
  Sag/Cor/Ax each ~half fluid-sensitive+fat-sup; reports mean ~1.1k chars, multilingual
  (Spanish observed); gold-58 prevalence range 0.16 (MCL) – 0.60 (Effusion).

## Changelog

- conjectured: A strong solo result is achievable because report-supervised distillation
  is underused and the 12-label macro metric rewards disciplined per-label work over raw
  compute. refuted by: (pending — first LB evidence). learned: (pending).
  criterion now: ISC-23/24 establish the reality check.

## Changelog (entries)

- conjectured: Train data would carry expert labels for all studies, with reports as an
  auxiliary enrichment signal. refuted by: train.csv inspection 2026-08-13 — only 58 of
  4,407 studies labeled; 4,349 report-only; no Report column in test.csv. learned: this
  is a semi-supervised / weak-supervision competition; report→label extraction quality
  is the highest-leverage component, and the 58 gold studies are a calibration/eval
  asset, not the training set. criterion now: ISC-31 reframed — report-extraction labels
  must reach measured agreement with the 58 gold studies before any image training run.

## Verification

- ISC-1: Bash — `kaggle competitions list` returned competition table (auth OK)
- ISC-2: Browser + Bash — "Rules accepted. Good luck!" toast; downloads return 200
- ISC-3: Bash — files enumerated; DICOM slices ~1.8 MB each; est. total ~1.2 TB logged
- ISC-4: Bash/pandas — train.csv parsed: 4,407 studies; prevalence table computed
- ISC-5: Bash — `df -h`: 227 GB free vs ~1.2 TB est.; Kaggle-resident decision logged
- ISC-6: Bash — `uv sync` clean; `uv run pytest` 4 passed
- ISC-8: Bash — `git push` f48b40e..68a3c2f main→main
- ISC-13: Read — test.csv header is `StudyInstanceUID` only; no Report column
- ISC-19: Read — train/test_series.csv carry plane + sequence-type columns from host
