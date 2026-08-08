#!/usr/bin/env bash
# Pull every model referenced by config/models.yaml into the local Ollama host.
set -euo pipefail

CONFIG="${1:-config/models.yaml}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"

if ! command -v "$OLLAMA_BIN" >/dev/null 2>&1; then
  echo "error: '$OLLAMA_BIN' not found. Install Ollama or run 'docker compose exec ollama ollama pull <model>'." >&2
  exit 1
fi

models=$(grep -E '^\s+model:\s' "$CONFIG" | awk '{print $2}' | tr -d '"'"'" | sort -u)

if [ -z "$models" ]; then
  echo "error: no models found in $CONFIG" >&2
  exit 1
fi

echo "Pulling models from $CONFIG:"
echo "$models" | sed 's/^/  - /'

while IFS= read -r model; do
  [ -z "$model" ] && continue
  echo "==> $model"
  "$OLLAMA_BIN" pull "$model"
done <<< "$models"

echo "Done."
