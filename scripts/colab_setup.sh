#!/usr/bin/env bash
# Create (or reuse) a Colab T4 session and push code + data caches + deps.  usage: scripts/colab_setup.sh <session> [--with-external]
set -u
SESSION=$1; WITH_EXT=${2:-}
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
if ! colab --auth=oauth2 status -s "$SESSION" 2>/dev/null | grep -q "Hardware"; then
  colab --auth=oauth2 new -s "$SESSION" --gpu T4 2>&1 | tail -2
fi
tar czf dist/bundle_code_data.tar.gz --exclude='__pycache__' src scripts configs pyproject.toml assets/fonts \
    data/splits/split_seed42.csv data/cache/glyphs.npz data/synth reports/eda/class_stats.csv
colab --auth=oauth2 upload -s "$SESSION" dist/bundle_code_data.tar.gz /content/bundle_code_data.tar.gz 2>&1 | tail -1
if [ "$WITH_EXT" = "--with-external" ]; then
  [ -f dist/external_cache.tar.gz ] || tar czf dist/external_cache.tar.gz data/external/glyphs_external.npz data/external/index.csv
  colab --auth=oauth2 upload -s "$SESSION" dist/external_cache.tar.gz /content/external_cache.tar.gz 2>&1 | tail -1
fi
colab --auth=oauth2 install -s "$SESSION" timm opencv-python-headless pyyaml tabulate imagehash 2>&1 | tail -1
cat > /tmp/colab_setup_$$.py <<PY
import subprocess, os
os.makedirs("/content/thaichar", exist_ok=True)
subprocess.run("cd /content/thaichar && rm -rf src scripts configs && tar xzf /content/bundle_code_data.tar.gz && ([ -f /content/external_cache.tar.gz ] && tar xzf /content/external_cache.tar.gz || true) && mkdir -p runs && ls", shell=True, check=True)
import torch; print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
colab --auth=oauth2 exec -s "$SESSION" -f /tmp/colab_setup_$$.py --timeout 600 2>&1 | tail -3
rm -f /tmp/colab_setup_$$.py
