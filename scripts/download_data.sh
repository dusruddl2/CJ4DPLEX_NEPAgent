#!/usr/bin/env bash
# Download FutureOmni test clips and training videos from Hugging Face.
# Usage: scripts/download_data.sh [test|train|all]   Re-running resumes; finished files are skipped.
# Files land under $FUTUREOMNI_ROOT (default data/downloads/FutureOmni inside the repository); the repository's
# data/raw/futureomni and data/raw/futureomni_train are symlinks to its test/ and train_hf/ subfolders.
# Downloads are untrusted data: never run an interpreter from inside these directories.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-all}"
FUTUREOMNI_ROOT="${FUTUREOMNI_ROOT:-$ROOT/data/downloads/FutureOmni}"
TEST_DIR="$FUTUREOMNI_ROOT/test"
TRAIN_DIR="$FUTUREOMNI_ROOT/train_hf"
TEST_REPO="OpenMOSS-Team/FutureOmni"            # official: annotations + split test-video zip
TRAIN_REPO="che111/futureomni"                  # re-upload of the official Google Drive training videos
DONE_MARKER="data/.done_download"
HF="${HF_CLI:-$ROOT/.venv/bin/hf}"

declare -A TEST_PART_BYTES=(
  [test_splitted_videos_part_aa]=21474836480
  [test_splitted_videos_part_ab]=21474836480
  [test_splitted_videos_part_ac]=6117515931
)
declare -A TRAIN_PART_BYTES=(
  [train_videos_part_aa]=21474836480
  [train_videos_part_ab]=21474836480
  [train_videos_part_ac]=21474836480
  [train_videos_part_ad]=19707381109
)

parts_complete() {
  local dir="$1"; shift
  local -n sizes="$1"
  local name
  for name in "${!sizes[@]}"; do
    [[ -f "$dir/$name" ]] || return 1
    [[ "$(stat -c %s "$dir/$name")" == "${sizes[$name]}" ]] || return 1
  done
}

link_repo_dir() {
  local link="$1" target="$2"
  mkdir -p "$(dirname "$link")"
  if [[ -L "$link" ]]; then
    [[ "$(readlink "$link")" == "$target" ]] || echo "warning: $link points elsewhere ($(readlink "$link"))"
  elif [[ -e "$link" ]]; then
    echo "warning: $link exists and is not a symlink; leaving it alone"
  else
    ln -s "$target" "$link"
  fi
}

download_test() {
  mkdir -p "$TEST_DIR"
  link_repo_dir data/raw/futureomni "$TEST_DIR"
  if parts_complete "$TEST_DIR" TEST_PART_BYTES; then
    echo "[test] all parts present in $TEST_DIR; skipping"
    return 0
  fi
  echo "[test] downloading $TEST_REPO -> $TEST_DIR"
  "$HF" download "$TEST_REPO" --repo-type dataset --local-dir "$TEST_DIR" \
    --include 'futureomni_test.json' 'futureomni-7k.json' 'README.md' 'test_splitted_videos_part_*'
}

download_train() {
  mkdir -p "$TRAIN_DIR"
  link_repo_dir data/raw/futureomni_train "$TRAIN_DIR"
  if parts_complete "$TRAIN_DIR" TRAIN_PART_BYTES; then
    echo "[train] all parts present in $TRAIN_DIR; skipping"
    return 0
  fi
  echo "[train] downloading $TRAIN_REPO -> $TRAIN_DIR"
  "$HF" download "$TRAIN_REPO" --repo-type dataset --local-dir "$TRAIN_DIR" --include 'train_videos_part_*'
}

case "$TARGET" in
  test) download_test ;;
  train) download_train ;;
  all) download_test; download_train ;;
  *) echo "usage: $0 [test|train|all]" >&2; exit 2 ;;
esac

if parts_complete "$TEST_DIR" TEST_PART_BYTES && parts_complete "$TRAIN_DIR" TRAIN_PART_BYTES; then
  date -Iseconds > "$DONE_MARKER"
  echo "all downloads complete: $DONE_MARKER"
else
  echo "not all downloads are complete; $DONE_MARKER not written"
fi
