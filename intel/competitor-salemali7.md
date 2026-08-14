# Competitor Intel — salemali7 "RSNA Knee +90% reports LLM 30 epochs" (public 0.899)

Analyzed 2026-08-13 from the public notebook (fork lineage: pilkwang baseline →
prvsiyan "read the report, then the knee"). Public LB 0.899; efficiency rank 209.
Leaderboard context that day: top 0.944, top-10 cutoff 0.934, median 0.832,
450 teams clustered 0.85–0.90.

## Their pipeline (condensed)
- **Supervision**: same semi-supervised insight as ours — 3 "report teachers"
  (lexicon CSV + two LLM label sets) nanmean-averaged, confidence-weighted BCE
  (weight 0.15–1.0), gold-58 hard override at weight 3.0.
- **Model**: DINOv2 small/base, last 6 blocks unfrozen (LR 8e-6 backbone / 1e-3 head),
  2.5D (3-slice groups), 336px cache, 130mm physical crop, 6 fixed plane×sequence
  "slots" (SAG/COR/AX fluid-FS, SAG fluid-noFS, COR T1, SAG T1), per-target attention
  over slot embeddings with hardcoded anatomical priors (strength 0.55).
- **Critical detail**: slice order from ImagePositionPatient projection onto slice
  normal — filename/InstanceNumber order is essentially random (Spearman ≈ 0.009).
- **Laterality**: normalized to left-knee convention; geometric side inference when
  the Laterality tag is missing (~50% of studies).
- **Augmentation**: rigid-only (rot ±8°, zoom-in ≤8%, translate ±5%, intensity ±10%);
  NO flips (would break laterality/anatomy).
- **Inference**: 20-member ensemble + legacy 4-fold bundle (solo 0.836) + public
  EfficientNet-B3 arm + RadImageNet ResNet-50 arm (α=0.20 rank-blend, gated per
  target); overlapping-window TTA with per-target pooling (max for fracture/menisci,
  top2-mean for ACL/MCL); **percentile-rank averaging** (metric reads only order);
  progressive submission checkpointing; weight fingerprint checks.
- Splits grouped on report-text hash (dupe-leak guard), NOT patient ID.

## Weaknesses to exploit
1. Blend/gate tuning validated on the 58 gold rows — high overfit risk in their
   selection process.
2. Report-teacher noise concentrated in specific languages; our Sonnet extraction
   is measured (0.81 vs 0.72 rules on gold-29 half) and auditable per language.
3. Slot classification by regex on SeriesDescription — vendor quirks silently
   misroute series; we get plane/sequence columns free from the organizers' CSVs.
4. Frozen hash-pinned ensemble arms — they iterate slowly; engineering is defensive
   (protecting 0.899) rather than climbing.
5. Hardcoded anatomical priors instead of learned routing.

## What to adopt
- Geometric slice ordering (ImagePositionPatient projection) — mandatory.
- 130mm physical crop; laterality normalization; rigid-only augmentation (no flips).
- Rank-averaging for ensembles; per-target TTA pooling; progressive submission writes.
- Confidence-weighted BCE with gold-58 override (we already planned this).
- Multi-teacher label averaging — ours: rules v0 + Sonnet; consider Opus as
  adjudicator on disagreements instead of a third full pass.

## Scores referenced in their code
0.836 legacy solo · 0.847 single-device · 0.899 public frontier · 0.906 cited
independent mechanism · gold-58 OOF of RadImageNet arm alone: 0.854.
