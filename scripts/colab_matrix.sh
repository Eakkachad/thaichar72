#!/usr/bin/env bash
# Run a list of configs on a Colab session sequentially; pull metrics/log back after each.
# usage: scripts/colab_matrix.sh <session> <suffix> <extra --set args...> -- config1.yaml config2.yaml ...
set -u
SESSION=$1; SUFFIX=$2; shift 2
SETS=()
while [ "$1" != "--" ]; do SETS+=("$1"); shift; done; shift
SCRATCH=$(mktemp -d)
for cfgpath in "$@"; do
  name=$(basename "$cfgpath" .yaml); exp="${name}${SUFFIX}"
  if [ -f "runs/$exp/metrics.json" ]; then echo "=== skip $exp (done) ==="; continue; fi
  # stage-2 configs: make sure the init checkpoint exists on the VM (re-upload after a session loss)
  init=$(grep -E '^init_from:' "$cfgpath" | awk '{print $2}')
  INIT_LINE=""
  if [ -n "$init" ] && [ "$init" != "null" ] && [ -f "$init" ]; then
    echo "uploading $init"; timeout 600 colab --auth=oauth2 upload -s "$SESSION" "$init" "/content/init_upload.pt" 2>&1 | tail -1
    INIT_LINE="os.makedirs(os.path.dirname('$init'), exist_ok=True); shutil.copy('/content/init_upload.pt', '$init') if os.path.exists('/content/init_upload.pt') and not os.path.exists('$init') else None"
  fi
  setargs=""; for s in "${SETS[@]}"; do setargs="$setargs, \"--set\", \"$s\""; done
  cat > "$SCRATCH/run_$exp.py" <<PY
import os, sys, subprocess, time, shutil
os.chdir("/content/thaichar"); t0 = time.time()
$INIT_LINE
cmd = [sys.executable, "scripts/train.py", "--config", "$cfgpath", "--exp-id", "$exp"$setargs]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-2500:]); print("STDERR tail:", r.stderr[-800:]); print("elapsed", round(time.time()-t0,1), "s")
PY
  echo "=== $(date +%H:%M:%S) START $exp ==="
  out=$(timeout 1700 colab --auth=oauth2 exec -s "$SESSION" -f "$SCRATCH/run_$exp.py" --timeout 1600 2>&1 | grep -v -i "warn")
  echo "$out" | tail -14
  if echo "$out" | grep -q "not found\|appears to be lost"; then echo "!!! SESSION LOST at $exp — aborting loop"; exit 2; fi
  mkdir -p "runs/$exp"
  for f in metrics.json log.csv best.pt val_logits.npy val_labels.npy; do
    timeout 300 colab --auth=oauth2 download -s "$SESSION" "/content/thaichar/runs/$exp/$f" "runs/$exp/$f" 2>&1 | tail -1
  done
  echo "=== $(date +%H:%M:%S) DONE $exp ==="
done
