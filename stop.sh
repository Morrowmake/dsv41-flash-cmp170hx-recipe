#!/usr/bin/env bash
# stop.sh — stop the server this checkout started.
#
# Only ever stops the container recorded in logs/container.id, and only after
# confirming it is ours. It never searches by name, so an unrelated vLLM on
# this machine is never touched. Nothing running? Exits 0.
#
# The image, weights and compile caches stay on disk, so a later start is
# quick.
#
# Equivalent to: ./start.sh stop
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" stop
