---
project: rsna-knee-2026
task: Win the RSNA Knee Abnormality Detection AI Challenge (solo)
effort: E4
phase: observe
progress: 0/64
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
- [ ] ISC-1: `kaggle` CLI installed and `kaggle competitions list` authenticates successfully
- [ ] ISC-2: Competition rules accepted on Kaggle (competition appears in `kaggle competitions list --mine` or data download authorized)
- [ ] ISC-3: Competition data files enumerated via `kaggle competitions files rsna-knee-abnormality-detection` (name TBC) with sizes logged in Decisions
- [ ] ISC-4: train.csv (labels) downloaded and label prevalence per target computed and logged
- [ ] ISC-5: Data storage decision made and logged (local disk vs Kaggle-only vs cloud box) based on actual dataset size
- [ ] ISC-6: `uv`-managed venv with pinned deps (`pyproject.toml`) installs clean and `pytest` passes
- [ ] ISC-7: Baseline all-0.5 submission validated by `src/rsna_knee/submission.py` and submitted to LB (pipeline smoke test)
- [ ] ISC-8: GitHub repo synced; training runs reproducible from a tagged commit

### Data understanding (EDA)
- [ ] ISC-9: Series inventory built — count of series per study, plane/sequence-type distribution from DICOM metadata, written to `artifacts/eda/series_inventory.csv`
- [ ] ISC-10: Per-site and per-language study counts computed; site distribution shift risk assessed in Decisions
- [ ] ISC-11: Label co-occurrence matrix computed and saved (informs multi-task head design)
- [ ] ISC-12: Report text audited: language mix, length distribution, structure; sample of 20 reports read manually against labels
- [ ] ISC-13: Test-set schema confirmed: does hidden test provide images only, or images+reports? Logged in Decisions (determines whole text strategy)
- [ ] ISC-14: DICOM pixel pipeline validated: windowing, orientation, spacing normalization produce visually correct slices for 10 random studies (saved PNGs eyeballed)

### Cross-validation harness
- [ ] ISC-15: 5-fold patient-grouped, site-stratified splits generated by existing `build_cv_splits` and leakage check passes
- [ ] ISC-16: OOF scoring loop wired: any model checkpoint → oof.csv → `score-oof` macro + per-label AUC, appended to run tracker
- [ ] ISC-17: CV↔LB correlation established: ≥2 submissions with differing CV plotted; correlation noted before trusting CV for selection
- [ ] ISC-18: Anti: no study from the same patient appears in both train and validation folds of any run (leakage validator in CI)

### Series identification & preprocessing
- [ ] ISC-19: Series-type classifier (plane × sequence) achieves ≥98% accuracy on a hand-labeled 200-series audit set, or DICOM-metadata rules achieve same
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
- [ ] ISC-31: Text model fine-tuned on reports→12 labels; OOF macro AUC of text-only model logged (expected ≥0.95 — reports contain the answers)
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
- 2026-08-13: BLOCKED on principal for: (a) Kaggle API token at `~/.kaggle/kaggle.json`,
  (b) accepting competition rules on the Kaggle website (cannot be done via CLI).

## Changelog

- conjectured: A strong solo result is achievable because report-supervised distillation
  is underused and the 12-label macro metric rewards disciplined per-label work over raw
  compute. refuted by: (pending — first LB evidence). learned: (pending).
  criterion now: ISC-23/24 establish the reality check.

## Verification

(populated as ISCs pass)
