#!/bin/bash
# Download the 308-study local subset series-by-series via kaggle CLI.
# Resumable: skips studies whose directory already exists and is non-empty.
set -u
cd "$(dirname "$0")/.."
DEST=data/train_series
mkdir -p "$DEST"
total=$(tail -n +2 data/local_subset_uids.csv | wc -l | tr -d ' ')
n=0
tail -n +2 data/local_subset_uids.csv | while read -r uid; do
  n=$((n+1))
  if [ -d "$DEST/$uid" ] && [ -n "$(ls -A "$DEST/$uid" 2>/dev/null)" ]; then
    continue
  fi
  kaggle competitions download rsna-knee-abnormality-detection \
    -f "train_series/$uid" -p "$DEST" -q 2>/dev/null
  # files arrive as a zip per path; unzip into place
  z="$DEST/$(basename "$uid").zip"
  [ -f "$z" ] && unzip -qo "$z" -d "$DEST" && rm -f "$z"
  echo "PROGRESS $n/$total $uid"
done
echo "SUBSET_DOWNLOAD_DONE"
