#!/usr/bin/env bash
# Downloads the Androids Corpus (3.69 GB) into <repo>/data/androids (gitignored).
# By running this you accept the terms at https://github.com/androidscorpus/data:
#   academic/non-commercial use only · no redistribution · not to be made public elsewhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/androids"
mkdir -p "$DEST"
URL="https://www.dropbox.com/s/2rurxlh70ihfki4/Androids-Corpus.zip?dl=1"
echo "Downloading to $DEST/Androids-Corpus.zip (resumable) ..."
curl -L -C - -o "$DEST/Androids-Corpus.zip" "$URL"
echo "Unzipping ..."
unzip -q -o "$DEST/Androids-Corpus.zip" -d "$DEST"
echo "Done: $(find "$DEST" -name '*.wav' | wc -l | tr -d ' ') wav files"
