# Round-2 Competitor Intel — 2026-08-16

Three public notebooks analysed as untrusted data. Baseline for "what's new" is
`intel/competitor-salemali7.md` (salemali7 / prvsiyan lineage, public 0.899).

Our position at time of writing: **0.840 public LB**, DINOv2-small, 6-slot 2.5D
slot-attention, 224px, 10 epochs, fold-0 only, LLM-extracted pseudo-labels
(0.836 gold agreement).

---

## Headline finding

**All three notebooks are downstream of the *same* pilkwang → prvsiyan lineage we
already documented.** Two of the three are inference-only re-blends of the *same
frozen 20-member DINOv2 checkpoint package*. There is no second independent
solution family in the public space — there is one training run and a growing
pile of rank-blend variations on top of it. That is both good news (nobody public
has a better encoder) and a warning (the public LB cluster at 0.89–0.90 is one
correlated artefact, and the private LB will not reward copying it).

---

## 1. `bend-the-knee-to-dinov3-the-original.ipynb` (108 votes)

### Verdict first: the DINOv3 claim is unsubstantiated

Searched the raw `.ipynb`: the string `dinov3` appears **exactly once, in a
markdown cell**. There is **no DINOv3 weight loading, no `dinov3` config lookup,
no timm/HF DINOv3 model id, and no checkpoint referencing it anywhere in the
executable code.** The only encoder loaders present are:

- `find_dinov2(variant='small')` → walks `/kaggle/input` for a HF dir whose path
  contains `dinov2`; raises `FileNotFoundError('DINOv2 weights not attached')`.
- `_rt_find_dino_base()` → explicitly requires `model_type == 'dinov2'` and
  `'base' in path`.
- `timm.create_model(cfg['backbone'], pretrained=False, ...)` for the
  RadImageNet ResNet-50 arm.

So: **the 108-vote "DINOv3" notebook is a DINOv2 + RadImageNet-ResNet50 rank
blend with a DINOv3 title.** Treat the vote count as a naming artefact, not
evidence. **Answer to "is DINOv3 materially better than DINOv2-small here": there
is no public evidence either way — nobody has actually run it.** (For reference
if we want to test it ourselves: DINOv3 ViT-S/16 is ~21M params, released under
the **DINOv3 License** — a Meta source-available research/commercial licence,
*not* Apache-2.0 like DINOv2. It requires acceptance on HF and imposes
attribution + use restrictions. Kaggle-legal but not as clean as DINOv2, so check
the competition rules on external-data licensing before using it.)

### Architecture + input pipeline

Two model families, blended by rank only:

**Family A — DINOv2 (identical contract to salemali7 intel).** `CROP_MM=130`,
`CACHE_IMG=336`, `GROUP=3` (2.5D triplets), 6 fixed slots
(`SAG_FLUID_FS / COR_FLUID_FS / AX_FLUID_FS / SAG_FLUID_NOFS / COR_T1 / SAG_T1`),
`SLOT_PRIOR_TABLE` with `SLOT_PRIOR_STRENGTH=0.55`, `UNFREEZE_LAST=6`,
`LR_HEAD=1e-3 / LR_BACKBONE=8e-6`, `WEIGHT_DECAY=0.02`, `EPOCHS=10`,
`BATCH_STUDIES=8`, geometric slice ordering by IPP·normal, laterality
normalisation with geometric fallback (`LAT_MIN_OFFSET_MM=20`), rigid-only aug
(rot ±8°, scale +8%, shift ±5%, intensity ±10%), `SLICE_BAND=(0.2,0.8)`.
`RUNS=[{'r224',224},{'r336',336}]` — multi-resolution members.

**Family B — `RTAHMIL`, a genuinely different head (NEW).** Not slot-attention
over 6 pooled slot embeddings; a three-stage transformer:
1. Per-series slice transformer (1 layer, 8 heads, learned `slice_pos_emb`) over
   K slice tokens.
2. Attention pooling to one series token per slot (learned `series_query`).
3. Series tokens get `slot_emb + 0.35*plane_emb + 0.35*sequence_emb`, then a
   **2-layer study-level transformer encoder with key-padding mask** for absent
   slots, then **12 learned target queries cross-attending over all series
   tokens**, each query biased by a `group_emb` over 6 clinical families
   (ligament / meniscus / OA / inflammation / bone / other).
4. Per-target head on `concat(target_context, study_global_mean)`.
Plus `stochastic_slot_mask` — **series dropout** during training (randomly drops
whole slots, guaranteeing at least one survives).

Also present: `SlotDepthMixer` — a learned depth-direction (through-slice)
smoothing operator with a binomial-initialised 5-tap kernel, per-plane and
per-contrast kernel offsets, and a `tanh`-gated mixing strength capped at
`alpha_max=0.25`, applied with the validity mask so padding never bleeds in.
That's a cheap, principled 3D-lite inductive bias on top of a 2D encoder.

### Label source

Same three-teacher scheme, plus a fourth voice:
- **pilkwang** report labels (fork parent's set),
- **stevenleehans** + **lixin73** — two more independent LLM/report label sets,
- an inlined **multilingual regex lexicon** (`extract()`), used as fallback and
  emitting both a score and a `__conf` per target. The lexicon covers EN, ES, FR,
  NL, DE, TR, HR/SR, EL, BG with directional negation (90-char window, adversative
  guard on but/however/ancak/pero/ali/…), uncertainty class, severity scaling,
  ICRS/Outerbridge grade parsing, OA compartment routing (medial/lateral/PF via
  site regexes), global-OA inheritance, and a **Synovitis back-off prior derived
  from Effusion** when no synovitis clause exists.
- Hard-fails (`LabelSourceError`) rather than silently degrading to the lexicon.

### Inference tricks (the actual content of the notebook)

A long chain of pinned rank blends, each a separate submission file:
- Overlapping-window TTA: `window_starts` = all `n-group+1` positions.
- **Per-target hard window pooling** (`max` / `top2` / `top3` / `logit_mean`).
- **NEW: soft (softmax-temperature) window pooling** —
  `weight = softmax(beta * p)` with per-target `beta` (ACL/MCL 6.0, menisci 8.0,
  Baker's/Contusion 8.0, Fracture 10.0), mixed into the hard pooling at per-target
  `alpha` 0.15–0.25. This is a smooth interpolation between mean and max — strictly
  better-behaved than a hard `max` on 10 windows.
- **NEW: fold-rank aggregation** (credited to romantamrazov): raw-average the 4
  members *within* each fold → rank each fold → equal-weight mean over 5 folds.
  This stops one over-confident fold dominating the rank space.
- Per-target member weights `LEGACY_MEMBER_WEIGHT_BY_TARGET` = Lateral Meniscus
  15.0, Lateral OA 15.0, Contusion 5.0, Medial OA 2.5 — i.e. the legacy bundle is
  given essentially all the vote on Lateral Meniscus and Lateral OA and none
  elsewhere. That is per-target arm gating tuned on ~58 gold rows. Fragile.
- Staged RadImageNet injection: **E10** = `0.50*rank(parent) + 0.50*rank(public-v15
  RadImageNet)` on 10 findings, *preserving* Baker's and Fracture; **E11** = a
  pixel-diverse 4-slot / 130mm / 224px / 8-slice RadImageNet family at `alpha=0.20`
  on all 12; **V60** = `0.98*E10E11 + 0.02*legacy-DINO` purely as a tiebreaker.
- SHA-256 pinning of every head file + parameter-count drift checks + prediction
  fingerprint checks. Extremely defensive engineering.
- **NEW: a stacking meta-model ("hybrid") on top of everything.** Features per
  study (1558 dims): PCA-192 of DINO slot features × 6 slots, slot presence mask,
  slot metadata, **hand-crafted radiomics** (56 dims/slot = mean/std/min/max over
  a 14-feature vector: intensity quantiles 1/10/50/90/99, centre-crop mean+std,
  gradient-magnitude mean+std, foreground fraction, foreground mean+std), 13 study
  meta features, and one-hot sex. Trained on **report pseudo-labels with confidence
  weighting** `w = 0.18 + 1.15*conf^1.4`, capped at 2200 pos / 2200 neg by
  confidence, plus the **58 gold rows at weight 7.0**. Four model families
  (LogisticRegression sweep, exact-only LR, ExtraTrees-420, HistGradientBoosting)
  × 8 seeds, combined `0.9*rank + 0.1*prob`. **Applied to only three targets:**
  Lateral Meniscus (0.125 base / 0.500 consensus / 0.375 PCA), Lateral OA
  (0.125 base / 0.875 pilkwang teacher), Synovitis (0.75 base / 0.25 PCA).
- **NEW: report-teacher Synovitis override** — a separate 8-checkpoint
  (2 seeds × 4 folds) `RTAHMIL` model at 336px / 0.42mm target spacing / 7 slices,
  rank-blended into Synovitis at **weight 0.75**. Synovitis is the target they
  clearly consider broken in the image model.
- **NEW: a foreign "Yash" family** (`yashbishnoi98/rsna-knee-infer-v1`) at
  `0.55 Yash / 0.45 public DINO`.

### Stated scores
No LB numbers given anywhere in the notebook body — only structural claims
("preserves 98% of the E10/E11 blend"). Fork parent is the 0.899 lineage.

### New vs salemali7 intel
Soft-window pooling with per-target beta; fold-rank aggregation; series dropout;
`SlotDepthMixer`; the 12-target cross-attention `RTAHMIL` head with group
embeddings; the radiomics+PCA stacking meta-model on pseudo-labels; the
Synovitis report-teacher at 0.75; two extra report-label authors
(stevenleehans, lixin73); the four-teacher label mix; the confirmation that
**DINOv3 is vapourware in the public space**.

---

## 2. `0-899-let-me-cook.ipynb` (claims 0.899)

### Verdict: inference-only, no training, no new weights

This is the **same 20-member DINOv2 manifest package**, re-blended. It contains no
training loop, no optimizer, no epoch count — it reads `train.csv` only for EDA
sanity checks. Its own markdown states plainly: *"the previous hybrid path had no
per-target AUC metadata in the attached 20 checkpoints"* and *"reaching ~0.95
likely requires a stronger training stage… not additional inference heuristics on
the same 20 DINO checkpoints."* Honest, and correct.

### Architecture + input pipeline
Identical checkpoint-compatible contract: `CROP_MM=130`, `CACHE_IMG=336`,
`GROUP=3`, **`CACHE_SLICES=12`** (→ 10 overlapping TTA windows), same 6
`SLOTS_RECOVERED`, same `SLOT_PRIOR_TABLE` at strength 0.55, same
`POOL_PARTS={'cls_mean':2,'cls_mean_focal':3}`. Pixel settings are **overwritten
per-member from the manifest** so each checkpoint is fed exactly what it was
fitted on.

One notable design note worth stealing conceptually: they **refuse the T1 slot
fallback** in the native rule set, with a measured justification — allowing the
T1 slot to fall back to the fluid pool would put one series into two slots for
**2383 of 4407 training studies** and leave **56% of the T1 slot holding PD or
T2**, corrupting the presence mask. Empty slot stays empty.

### Label source
None used — inference only. Labels are baked into the imported checkpoints
(pilkwang report labels upstream).

### Inference tricks
- `TTA_OVERLAP=True` → 10 sliding 3-slice windows over the 12-slice cache.
- **Per-target window pooling table** (this is the clean, explicit version of the
  same idea buried in notebook 1):
  | target | pooling |
  |---|---|
  | Fracture, Contusion, Medial Meniscus, Lateral Meniscus, Baker's | `max` |
  | ACL, MCL | `top2` mean |
  | Medial/Lateral/PF OA, Effusion, Synovitis | `mean` |
  Rationale stated: averaging 10 windows dilutes a tear or fracture visible in
  only one or two positions. Diffuse findings (OA, effusion, synovitis) keep mean.
- **Equal member rank voting**, deliberately replacing scalar holdout weighting —
  their argument is that without per-target OOF metadata, weighting is unfounded.
- Optional **EfficientNet-B3 five-fold** cross-family blend at global `alpha=0.10`
  (rank space), gated behind `ALLOW_UNAUDITED_B3` and a nested-OOF audit file.
  A per-target B3 alpha table exists as a *candidate only*: ACL 0.00, MCL 0.10,
  Medial Meniscus 0.00, **Lateral Meniscus 0.35, Lateral OA 0.35, PF OA 0.35,
  Synovitis 0.35, Baker's 0.35**, Effusion 0.25, Medial OA 0.15, Contusion 0.00,
  Fracture 0.00. Read this as a **map of where DINOv2 is weakest** — it is exactly
  the diffuse/textural findings where an ImageNet CNN adds signal, and exactly
  zero on the focal traumatic ones where DINO already wins.
- Time-budget guard that shrinks the TTA window count from the centre outward.
- Writes `submission_dino_mean_baseline.csv`, `submission_dino_frontier.csv`,
  and promotes to `submission.csv` only through an audit gate.

### Stated scores
`0.891` for the DINO inference path it forks from; title claims `0.899`; explicitly
disclaims 0.95. No OOF numbers of its own.

### New vs salemali7 intel
The explicit per-target pooling table with rationale; the equal-rank-vote argument
against scalar member weighting; the **B3 per-target alpha table as a weakness
map**; the quantified T1-slot-fallback corruption figure (2383/4407, 56%);
`CACHE_SLICES=12` → 10 windows as the concrete TTA budget.

---

## 3. `selecting-4-pos-neg-fracture-series-all-axial.ipynb` (fresh)

### Verdict: an EDA notebook, ~175 lines, no model, no score

A visualisation utility. It selects 4 "high confidence" and 4 "low confidence"
**axial** series and renders a 4×8 slice montage plus the translated report.

### What's in it
- `IMAGE_RESOLUTION=336`, `K=32` slices, ImageNet mean/std, percentile clip at
  `[0.5, 99.5]`.
- `plane_from_iop()` — plane derived from `cross(row, col)` of
  `ImageOrientationPatient`, `argmax|normal|` → sagittal/coronal/axial. Same
  geometry insight as ours, computed rather than regex'd.
- `compute_slice_ordering()` — same IPP-projection-onto-normal ordering, then
  `np.linspace(0, n-1, K)` uniform subsampling (**not** our band-limited
  `SLICE_BAND`).
- `crop_background()` — **NEW and cheap**: threshold `max over slices > 8`,
  bounding box of non-zero, 4px pad. A content-based crop applied *before* the
  physical-mm crop, which removes the black MRI air margin that varies wildly by
  vendor FOV. Notebooks 1 and 2 use only the fixed 130mm physical crop.
- Laterality mirror on coronal/axial only (`vol[:, :, ::-1]`), sagittal left
  alone — matches our convention.
- **Report translation via `deep_translator` / `GoogleTranslator(source='auto')`.**
  Requires network, so this is a *training-time* trick only, not usable in a
  submission kernel. Notable as an alternative to multilingual regex/LLM
  extraction: translate everything to English first, then run a single
  English-only extractor.
- Uses the organizers' `train_series.csv` `Anatomical_Plane` column directly
  (as we do) rather than regexing `SeriesDescription`.

### Label source / training / scores
None. The "high vs low confidence" split is a placeholder (`Fracture.isin([0,1])`
then `.head(4)`) — it does not actually stratify anything. No signal here.

### New vs salemali7 intel
`crop_background()` content-bbox pre-crop; the translate-then-extract-in-English
labelling strategy; confirmation that axial-only fracture work is being explored
publicly but has produced nothing yet.

---

## TOP 5 ACTIONABLE ITEMS

Ranked by expected LB gain per GPU-hour from our 0.840 fold-0-only baseline.

### 1. Per-target TTA window pooling — `max` for focal, `top2` for ligaments, `mean` for diffuse
**~0.005–0.015 macro AUC. Cost: ~0 GPU-hours (inference-side only).**
Both 0.89+ notebooks converge on the same table independently, and notebook 1
goes further with softmax-temperature pooling. The mechanism is concrete:
a meniscal tear or a fracture line appears in 1–2 of 10 sliding windows;
averaging all 10 divides its evidence by five, and macro AUC reads only order.
**Port:** set `CACHE_SLICES=12`, `GROUP=3`, generate all 10 window starts, stack
window probabilities to `[W, B, T]`, then `max` for Fracture / Contusion / Medial
Meniscus / Lateral Meniscus / Baker's, `topk(2).mean()` for ACL / MCL, `mean` for
the three OA targets + Effusion + Synovitis. Then upgrade to the soft version:
`p_soft = (softmax(beta*p, dim=0) * p).sum(0)` with beta = 6 (ACL/MCL), 8 (menisci,
Baker's, Contusion), 10 (Fracture), and blend `0.8*hard + 0.2*soft`. Highest
gain-per-hour item on this list by a wide margin — it costs one inference run.

### 2. Train folds 1–4 and aggregate by fold-rank, not by pooled score
**~0.010–0.020. Cost: ~4× our current single-fold training run.**
We are fold-0 only against ensembles of 20+ members; this is the single largest
structural gap. But copy their *aggregation*: raw-average members within a fold →
rank-normalise each fold → equal-weight mean across folds. Ranking per fold before
averaging stops a single over-confident fold from dominating the percentile space,
which is exactly the failure mode when folds see different label noise. Also adopt
their split key: group on **report-text hash**, not patient ID (duplicate-report
leak guard). This is the highest absolute gain available to us, just not the
cheapest.

### 3. Add one independent non-DINO arm, weighted per target by their measured map
**~0.008–0.015. Cost: ~1× a small CNN training run (B3 or RadImageNet R50).**
Their per-target alpha table is free reconnaissance: EfficientNet-B3 earns
**0.35** weight on Lateral Meniscus, Lateral OA, PF OA, Synovitis, Baker's,
**0.25** on Effusion, and **0.00** on ACL, Medial Meniscus, Contusion, Fracture.
That is a measured statement that DINOv2 self-supervised features are strong on
focal traumatic findings and weak on diffuse textural ones (cartilage, synovium,
fluid) — where an ImageNet/RadImageNet CNN's low-level texture bias helps.
**Port:** train one EfficientNet-B3 or RadImageNet ResNet-50 on the same 6-slot
cache, rank-blend at global alpha 0.10 as the safe default, and only go per-target
if we can validate on more than 58 gold rows. Do **not** replicate their 15.0-vs-0.0
per-target member weights — those are tuned on 58 rows and will not survive the
private split.

### 4. Series dropout + learned depth mixing in our slot-attention head
**~0.005–0.010. Cost: no extra hours — it changes the same training run.**
Two cheap architectural upgrades from `RTAHMIL`/`SlotDepthMixer` that fit our
existing 6-slot design without a rewrite:
(a) **Stochastic slot masking during training** — randomly drop whole slots
(guaranteeing ≥1 survives). Real studies are missing slots constantly; a model
trained only on full studies degrades on the incomplete ones, and the presence
mask makes this trivial to implement. This is regularisation *and* distribution
matching in one line.
(b) **Learned through-slice mixing** — a 5-tap depth-direction convolution with
binomial init, per-plane and per-contrast offsets, `tanh`-gated to a maximum
strength of 0.25, applied under the validity mask. It gives a 2D encoder a small
amount of 3D context at almost no parameter or FLOP cost, and the gate means it
can learn to do nothing if it doesn't help.
Also worth 20 minutes: replace our hardcoded anatomical slot priors with their
**target-group embedding** (`ligament / meniscus / OA / inflammation / bone /
other` added at 0.25 strength to the target queries) — learned routing with a
clinical prior, instead of a fixed 0.55-strength mask.

### 5. Content-bbox background crop before the 130mm physical crop
**~0.002–0.008. Cost: ~0 GPU-hours (cache rebuild only, CPU).**
From notebook 3, and absent from both 0.89 notebooks — a genuine free edge.
`mask = vol.max(axis=0) > 8` → bounding box → 4px pad, applied *before* resizing.
Vendor FOV varies enormously across this multi-centre dataset; a fixed 130mm
physical crop still leaves a variable black air margin, so the knee occupies a
different fraction of the 224px input from study to study. Removing that margin
first makes the effective anatomical scale consistent across vendors, which is
exactly the invariance a ViT at 224px cannot learn for free. Cheap to test:
rebuild the cache, retrain fold 0, compare. If it works it compounds with
everything above.

### Deliberately not recommended
- **Chasing DINOv3.** The 108-vote notebook doesn't contain it. There is zero
  public evidence it beats DINOv2-small here, and its licence is
  source-available rather than Apache-2.0 — burning GPU-hours on it is a bet with
  no prior. Revisit only if items 1–5 are exhausted.
- **Copying their 0.899 blend wholesale.** It is a fingerprinted, SHA-pinned pile
  of coefficients selected on 58 gold rows, and the entire public 0.89–0.90
  cluster is one correlated artefact of a single training run. Our independent
  LLM label set (0.836 gold agreement, auditable per language) is the asset that
  is *uncorrelated* with theirs — improving supervision quality is where our
  private-LB edge lives, and both 0.89 notebooks say so explicitly in their own
  markdown.
- **`deep_translator` at inference.** Network-dependent; Kaggle submission kernels
  run offline. Usable for building better training labels only.
