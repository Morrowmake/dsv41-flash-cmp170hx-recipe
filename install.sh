#!/usr/bin/env bash
# install.sh — pull the engine image pinned in start.sh by digest. It carries
# the sm_80 EXL3 and Marlin extensions prebuilt and the release's launch.
#
# Already there? Exits 0 without touching it.
# Pull anyway:  FORCE_INSTALL=1 ./install.sh
#
# Equivalent to: ./start.sh install
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" install
