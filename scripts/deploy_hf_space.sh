#!/usr/bin/env bash
# Sync bundle and push to Hugging Face Space kinaar111/oam_flux.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_DIR="${HF_SPACE_DIR:-$ROOT/../oam_flux-hf}"

bash "$ROOT/scripts/sync_hf_space.sh"

if [[ ! -d "$HF_DIR/.git" ]]; then
  echo "Cloning HF Space into $HF_DIR ..."
  rm -rf "$HF_DIR"
  git clone "https://huggingface.co/spaces/kinaar111/oam_flux" "$HF_DIR"
fi

rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$ROOT/space/oam_flux/" "$HF_DIR/"

cd "$HF_DIR"
# Remove stray bytecode from prior deploys
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

git add -A
if git diff --cached --quiet; then
  echo "HF Space already up to date."
  exit 0
fi

MSG="Deploy oam_flux HF Space ($(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo dev))"
git commit -m "$MSG"

push_ok=false
if [[ -n "${HF_TOKEN:-}" ]]; then
  git remote set-url origin "https://kinaar111:${HF_TOKEN}@huggingface.co/spaces/kinaar111/oam_flux"
fi
if git push 2>/dev/null; then
  push_ok=true
  echo "Deployed (git push) → https://huggingface.co/spaces/kinaar111/oam_flux"
fi
if [[ "$push_ok" != true ]]; then
  echo "git push failed — falling back to hf upload ..."
  hf upload kinaar111/oam_flux "$HF_DIR" --repo-type=space --commit-message "$MSG"
  echo "Deployed (hf upload) → https://huggingface.co/spaces/kinaar111/oam_flux"
fi