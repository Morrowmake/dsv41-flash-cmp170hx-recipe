#!/usr/bin/env bash
# One-shot sanity check against a running server.
#
# Waits for /health, sends one chat request (thinking off, so the answer comes
# back promptly in .content) and one tool-call request with the model's
# default reasoning, prints the decode rate computed from the usage block of
# each, and echoes the KV cache line from the server log if it can find one.
# Exits 1 if the chat request returns nothing or the tool call is not parsed.
#
# Usage:  ./smoke.sh
#
# Env overrides:
#   HOST        default 127.0.0.1
#   PORT        default 8000
#   MODEL_ID    default $SERVED_MODEL_NAME, else dsv41-flash
#   API_KEY     sent as "Authorization: Bearer <key>" when set
#   WAIT        seconds to wait for /health (default 3600 -- a cold start is
#               model load, the Engram tables and CUDA graph capture)
#   SERVE_LOG   server log to scrape the KV line from (default ./logs/serve.log)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE="http://$HOST:$PORT"
MODEL_ID="${MODEL_ID:-${SERVED_MODEL_NAME:-dsv41-flash}}"
WAIT="${WAIT:-3600}"
SERVE_LOG="${SERVE_LOG:-$REPO_ROOT/logs/serve.log}"

command -v jq >/dev/null || { echo "smoke.sh: needs jq." >&2; exit 1; }

# /v1 needs the key when the server was started with API_KEY; /health never does.
AUTH=()
[ -n "${API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $API_KEY")

# POST a chat request. Prints the response body, then on its own last line the
# wall time curl measured, so no clock arithmetic is needed here.
post_chat() {
  curl -s ${AUTH[@]+"${AUTH[@]}"} -w '\n%{time_total}' "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$1"
}

echo "==> waiting for $BASE/health (up to ${WAIT}s)"
deadline=$(( $(date +%s) + WAIT ))
until [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health" || true)" = "200" ]; do
  [ "$(date +%s)" -lt "$deadline" ] || { echo "smoke.sh: server did not come up." >&2; exit 1; }
  sleep 5
done
echo "    healthy"
echo "    served models: $(curl -s ${AUTH[@]+"${AUTH[@]}"} "$BASE/v1/models" | jq -r '.data[].id' | paste -sd, -)"

# --- 1. plain chat -----------------------------------------------------------
echo
echo "==> chat request"
req_chat=$(jq -n --arg m "$MODEL_ID" '{
  model: $m,
  messages: [{role:"user", content:"In three sentences, explain why PCIe peer-to-peer matters for tensor parallelism."}],
  max_tokens: 200,
  temperature: 0,
  chat_template_kwargs: {thinking: false}
}')
out=$(post_chat "$req_chat")
wall_chat=${out##*$'\n'}
resp_chat=${out%$'\n'*}
# Any reasoning comes back in .reasoning and the answer in .content.
echo "$resp_chat" | jq -e '(.choices[0].message.content // "") | length > 0' >/dev/null \
  || { echo "smoke.sh: no answer in the chat reply:" >&2; echo "$resp_chat" >&2; exit 1; }
echo "$resp_chat" | jq -r '"    reply: " + (.choices[0].message.content | .[0:160] | gsub("\n";" ")) + " ..."'
echo "$resp_chat" | jq -r --arg w "$wall_chat" '
  .usage as $u | "    usage: prompt=\($u.prompt_tokens) completion=\($u.completion_tokens) wall=\($w|tonumber|.*100|round/100)s  ->  \(($u.completion_tokens / ($w|tonumber) * 10 | round) / 10) tok/s"'

# --- 2. tool call ------------------------------------------------------------
echo
echo "==> tool-call request"
req_tool=$(jq -n --arg m "$MODEL_ID" '{
  model: $m,
  messages: [{role:"user", content:"What is the weather in Reykjavik right now? Use the tool."}],
  tools: [{
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather for a city.",
      parameters: {
        type: "object",
        properties: {
          city: {type:"string", description:"City name"},
          unit: {type:"string", enum:["celsius","fahrenheit"]}
        },
        required: ["city"]
      }
    }
  }],
  tool_choice: "auto",
  max_tokens: 4096,
  temperature: 0
}')
out=$(post_chat "$req_tool")
wall_tool=${out##*$'\n'}
resp_tool=${out%$'\n'*}
if [ "$(echo "$resp_tool" | jq -r '(.choices[0].message.tool_calls // []) | length')" -gt 0 ]; then
  echo "$resp_tool" | jq -r '.choices[0].message.tool_calls[0] | "    tool_call: \(.function.name)(\(.function.arguments))"'
else
  echo "    NO tool_call parsed -- check --enable-auto-tool-choice / --tool-call-parser" >&2
  echo "$resp_tool" | jq -r '"    content: " + ((.choices[0].message.content // "") | .[0:200])' >&2
  exit 1
fi
echo "$resp_tool" | jq -r --arg w "$wall_tool" '
  .usage as $u | "    usage: prompt=\($u.prompt_tokens) completion=\($u.completion_tokens) wall=\($w|tonumber|.*100|round/100)s  ->  \(($u.completion_tokens / ($w|tonumber) * 10 | round) / 10) tok/s"'

# --- 3. KV line from the log -------------------------------------------------
echo
echo "==> KV cache"
if [ -r "$SERVE_LOG" ]; then
  kv=$(grep -h "GPU KV cache size" "$SERVE_LOG" | tail -1 || true)
  if [ -n "$kv" ]; then echo "    ${kv#*] }"; else echo "    (no 'GPU KV cache size' line in $SERVE_LOG yet)"; fi
else
  echo "    (no readable log at $SERVE_LOG -- set SERVE_LOG=... to scrape it)"
fi

echo
echo "smoke: ok"
