#!/usr/bin/env bash
# Wait until a T4 can be allocated again (free-tier quota), then run every remaining config (skip-done logic).
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
CFGS="$@"
while true; do
  if timeout 240 colab --auth=oauth2 new -s gpu-probe --gpu T4 2>&1 | grep -q "READY"; then
    colab --auth=oauth2 stop -s gpu-probe >/dev/null 2>&1
    echo "### $(date +%H:%M:%S) GPU available — resuming"
    scripts/colab_autorun.sh "" "num_workers=2" 12 $CFGS
    left=$(for c in $CFGS; do n=$(basename "$c" .yaml); [ -f "runs/$n/metrics.json" ] || echo "$c"; done)
    [ -z "$left" ] && { echo "### $(date +%H:%M:%S) RESUME ALL DONE"; exit 0; }
    echo "### $(date +%H:%M:%S) still remaining: $(echo $left | wc -w) — waiting again"
  else
    echo "$(date +%H:%M:%S) no GPU yet"; sleep 1200
  fi
done
