# Plan: Noisy-Student Loop with Statistically Sound Referee

## Context

Fold-0 trained (0.804 OOF), first submission scoring on LB. Next capability: the
disciplined AlphaZero-analogue — Noisy Student self-training over the 4,407-study
corpus. The referee-design memo (2026-08-16) establishes that gold-58 lacks power for
per-round gating (MDE 0.09–0.20 vs real gains 0.01–0.03) and is adaptively vulnerable
(Dwork/Blum-Hardt). This plan implements the memo's 8-step round structure on our
existing code with CV-primary gating, split gold slices, per-label Confident Learning,
and pre-committed stopping rules.

## Sequencing gate (before round 1 starts)

Wait for: (a) LB score of fold-0/v1.1 submission, (b) v1.3 retrain OOF delta. These
set the baseline label set (winner of v1.1-vs-v1.3) and give the CV↔LB correlation
anchor. Round 1 teacher = that winner's checkpoint.

## Reuse

- `src/rsna_knee/imaging/train.py` — teacher/student training (unchanged; students
  train fresh from scratch per memo §3)
- `src/rsna_knee/metrics.py` — `binary_roc_auc` for all gating math
- `src/rsna_knee/tracker.py` — every round logged as runs
- `artifacts/fold0/oof_fold0.csv` format — OOF comparison basis
- Kernel templates `kaggle_kernel/` (train, T4-flagged push) and `kaggle_kernel_infer/`
- M4 + `data/cache224/` (308 studies) — cheap collapse-triage before GPU spend

## New module: `src/rsna_knee/selftrain/`

1. **`gold_split.py`** — deterministic split of gold-58 into `working40` /
   `locked18`, stratified by per-label positive counts, seed pre-committed (2026).
   Locked slice enforced by code: any evaluation call naming locked studies raises
   unless `--final-unlock` flag passed. Split written once to
   `data/gold_split_v1.json`, committed, never regenerated.
2. **`referee.py`** — the gate:
   - `cv_gate(oof_prev, oof_new)`: primary accept/reject. Per-label AUC deltas on
     the ~877-study fold OOF + macro delta; accept iff macro delta ≥ +0.003 AND no
     label degrades by > 0.01 (floors pre-committed here as constants).
   - `gold_check(preds, every_n=2)`: stratified bootstrap (3,000 draws) of AUC delta
     on working-40; directional signal only if >90% draws agree AND |delta| ≥ 0.02.
     Runs every 2nd round; result is advisory (logged, can veto only on strong
     negative), never a per-round approve.
   - `final_check()`: single use of locked-18 at campaign end.
3. **`corrections.py`** — per-label pseudo-label correction:
   - cleanlab `multilabel_classification` one-vs-rest Confident Learning over
     teacher OOF probabilities vs current label set → candidate flags with
     auto-derived per-class thresholds.
   - Low-count-label backoff: for labels with <10 gold-working positives (Fracture,
     MCL), thresholds pooled with CV-estimated per-label AUC weights (from
     `label_weights.json`) instead of raw empirical precision.
   - Per-label correction caps: accepted corrections limited so post-correction
     prevalence stays within a band around current prevalence; band width scales
     with measured label AUC (0.65-AUC labels get the tightest cap).
   - Correlated-pair co-drift log: ACL↔Contusion, MedMen↔MedOA, LatMen↔LatOA joint
     correction counts vs baseline co-occurrence; over-threshold → flagged for
     TABOOST manual review, round pauses on that label pair.
4. **`round_runner.py`** — CLI orchestrating one round locally (steps 2–4, 6–7 of
   memo): reads teacher OOF, produces corrected label CSV `labels_r{N}.csv` +
   correction report, runs M4 triage (student forward pass sanity on cache224),
   emits the Kaggle dataset payload for the student train. GPU training itself
   stays on Kaggle via existing kernel with `--labels-csv labels_rN`.

## Round structure (per memo §5, mapped)

1. Teacher = current best checkpoint (starts: v1.1/v1.3 winner).
2. Teacher OOF predictions on all 4,407 → already produced by train kernels
   (extend train.py output to full-corpus predictions in the same run — small patch:
   after best epoch, run inference over the training folds too; costs ~30 min GPU).
3–4. `corrections.py` → `labels_r1.csv` + caps + co-drift log.
5. Student kernel trains fresh on `labels_r1.csv` (same fold-0 split, same config).
6. M4 triage on cache224 before pushing (obvious-collapse check only).
7. `referee.cv_gate` on OOF; `gold_check` every 2nd round.
8. Stop: HARD DEFAULT 3 rounds max. Early stop if cv_gate rejects. Rollback = keep
   prior label set + checkpoint. After stop: `final_check()` on locked-18, once.

## Pre-committed constants (written into referee.py, not tuned after peeking)

- cv_gate: macro ≥ +0.003, per-label degradation floor 0.01
- gold_check: 3,000 bootstrap draws, 90% direction, 0.02 effect floor, every 2 rounds
- Max 3 rounds; rollback on first rejection
- Gold split seed 2026, 40/18 stratified

## Dependencies

- Add `cleanlab>=2.6` to pyproject (main deps — CPU-only, used locally).

## Verification

- Unit tests: gold split determinism + locked-slice enforcement; cv_gate accept and
  reject cases on synthetic OOFs; bootstrap gold_check direction logic; correction
  caps respected on synthetic flags; co-drift flag fires on injected correlated flip.
- Dry run: round 0 "identity round" — run corrections with caps=0 (no changes
  accepted) end-to-end to validate plumbing before any real round.
- Every round appends to `runs.csv` with per-label AUCs; correction reports
  committed under `artifacts/selftrain/`.

## Resolved decisions

- TABOOST 2026-08-16: **single-model rounds** — two rounds fit this week's quota;
  escalate to two-model agreement only if round-1 correction reports look suspicious
  on Effusion/Synovitis.
