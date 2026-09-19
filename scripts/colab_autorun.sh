#!/usr/bin/env bash
# Free-tier-proof runner: (re)creates a T4 session whenever it is pruned and resumes the config list.
# usage: scripts/colab_autorun.sh <suffix> "<--set args, space separated or empty>" <max_attempts> cfg1.yaml cfg2.yaml ...
set -u
SUFFIX=$1; SETS=$2; MAX=$3; shift 3
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
remaining() { for c in "$@"; do n=$(basename "$c" .yaml); [ -f "runs/${n}${SUFFIX}/metrics.json" ] || echo "$c"; done; }
for attempt in $(seq 1 "$MAX"); do
  left=$(remaining "$@"); [ -z "$left" ] && { echo "=== all done ==="; exit 0; }
  SESSION="thai-$(date +%H%M%S)"
  echo "=== attempt $attempt: session $SESSION, remaining: $(echo $left | wc -w) ==="
  scripts/colab_setup.sh "$SESSION" --with-external 2>&1 | tail -4
  if ! colab --auth=oauth2 status -s "$SESSION" 2>/dev/null | grep -q Hardware; then echo "!!! could not create session; sleeping 10 min"; sleep 600; continue; fi
  scripts/colab_matrix.sh "$SESSION" "$SUFFIX" $SETS -- $left
  colab --auth=oauth2 stop -s "$SESSION" 2>&1 | tail -1
done
left=$(remaining "$@"); [ -z "$left" ] && echo "=== all done ===" || echo "!!! gave up; remaining: $left"
