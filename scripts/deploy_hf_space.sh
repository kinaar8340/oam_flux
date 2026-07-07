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
  "$ROOT/space/oam_flux/" "$HF_DIR/"

cd "$HF_DIR"
git add -A
if git diff --cached --quiet; then
  echo "HF Space already up to date."
  exit 0
fi

MSG="Deploy oam_flux HF Space ($(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo dev))"
git commit -m "$MSG"

if git push 2>/dev/null; then
  echo "Deployed (git push) → https://huggingface.co/spaces/kinaar111/oam_flux"
else
  echo "git push failed — falling back to hf upload ..."
  hf upload kinaar111/oam_flux "$HF_DIR" --repo-type=space --commit-message "$MSG"
  echo "Deployed (hf upload) → https://huggingface.co/spaces/kinaar111/oam_flux"
fi