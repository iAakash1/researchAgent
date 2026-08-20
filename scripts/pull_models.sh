#!/usr/bin/env bash
# Pull every model referenced by config/models.yaml into the local Ollama host.
set -euo pipefail

CONFIG="${1:-config/models.yaml}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"

if ! command -v "$OLLAMA_BIN" >/dev/null 2>&1; then
  echo "error: '$OLLAMA_BIN' not found. Install Ollama or run 'docker compose exec ollama ollama pull <model>'." >&2
  exit 1
fi

# Only aliases whose provider is ollama: a Groq model id is not pullable locally.
models=$(awk '
  /^  [a-z_]+:$/            { provider = "" }
  /^    provider:/          { provider = $2 }
  /^    model:/ && provider == "ollama" { print $2 }
' "$CONFIG" | tr -d '"'"'" | sort -u)

if [ -z "$models" ]; then
  echo "error: no models found in $CONFIG" >&2
  exit 1
fi

# The embedding model lives in retrieval.yaml, not the chat catalogue.
embedding=$(awk '/^embeddings:/{e=1} e && /^  model:/{print $2; exit}' config/retrieval.yaml | tr -d '"'"'")
[ -n "$embedding" ] && models=$(printf '%s\n%s' "$models" "$embedding" | sort -u)

echo "Pulling models from $CONFIG:"
echo "$models" | sed 's/^/  - /'

while IFS= read -r model; do
  [ -z "$model" ] && continue
  echo "==> $model"
  "$OLLAMA_BIN" pull "$model"
done <<< "$models"

echo "Done."
