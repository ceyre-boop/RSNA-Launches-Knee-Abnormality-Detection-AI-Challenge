We're officially in — team #1,412 — and day one produced a strategy-changing discovery.

What happened:

1. Clicked through the rules dialog on your open Chrome tab → "Rules accepted. Good luck!" — downloads unlocked immediately.
2. Pulled all metadata files: train.csv (with full radiology reports), train_series.csv, test schemas, sample submission.
3. Ran the first EDA. Committed and pushed everything (69988c2), 9 of 54 criteria now verified.

The discovery — this competition is not what it looks like:

- Only 58 of 4,407 training studies have expert labels. The other 4,349 have only the written radiology report.
- The hidden test set is images-only (no reports).
- So the winning pipeline is: extract the 12 labels from multilingual reports → train image models on those pseudo-labels → calibrate against the 58 gold studies. Most teams will treat the reports as a bonus; they're actually the entire training signal. Our text track just went from side-bet to main weapon — and it runs locally, since every report is already on this Mac.

Other intel:

- Series-type identification is free — the organizers ship plane + sequence-type columns for every series (train and test). One planned workstream deleted.
- 24,371 series (~5.5 per study), Sag/Cor/Ax roughly balanced, half fluid-sensitive fat-sup.
- Reports are multilingual (Spanish confirmed on row one) — ~1,100 chars average.
- Images estimated ~1.2 TB vs 227 GB free locally → images stay Kaggle-resident; decision logged.
- One risk logged: no PatientID column in train.csv — patient-level CV grouping needs DICOM headers.
