#!/usr/bin/env bash
# One chat request with curl. The server listens on 127.0.0.1:8000 by default
# and has no API key; if you set API_KEY in .env, export it here too.
set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
curl -s "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -d '{"model": "dsv41-flash",
       "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
       "max_tokens": 2048,
       "reasoning_effort": "low"}' \
  | jq -r '.choices[0].message | "reasoning:\n\(.reasoning // "")\n\nanswer:\n\(.content)"'
