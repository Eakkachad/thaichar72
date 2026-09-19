# Google Colab CLI — verified usage (2026-09-19)

Install / fix: `uv tool install google-colab-cli --with "jupyter-kernel-client<1.0" --force`
(colab-cli 0.6.0 imports `KernelClient`, removed in jupyter-kernel-client 1.0).
Auth: owner ran `colab --auth=oauth2 sessions` interactively once (PKCE → same terminal for URL + code).
Always pass the global flag `--auth=oauth2`. Account = free tier → `--gpu T4` only.

```bash
colab --auth=oauth2 new -s thai-test --gpu T4           # provision (READY in ~1 min)
colab --auth=oauth2 status -s thai-test
tar czf dist/bundle_code_data.tar.gz src scripts configs pyproject.toml assets/fonts \
    data/splits/split_seed42.csv data/cache/glyphs.npz data/synth reports/eda/class_stats.csv   # ~6 MB
colab --auth=oauth2 upload -s thai-test dist/bundle_code_data.tar.gz /content/bundle_code_data.tar.gz
colab --auth=oauth2 install -s thai-test timm opencv-python-headless pyyaml tabulate imagehash
colab --auth=oauth2 exec -s thai-test -f some_script.py --timeout 1600   # DEFAULT TIMEOUT IS 30 s — always set it
colab --auth=oauth2 download -s thai-test /content/thaichar/runs/X/metrics.json runs/X/metrics.json
colab --auth=oauth2 stop -s thai-test                   # ALWAYS when done
```
Kernel state persists across `exec` calls; scripts should `os.chdir("/content/thaichar")` and call
`scripts/train.py` via subprocess so each run is isolated.
`scripts/colab_matrix.sh <session> <suffix> <--set k=v ...> -- cfg1.yaml cfg2.yaml` runs configs sequentially
and pulls `metrics.json / log.csv / best.pt` back into `runs/<cfg><suffix>/`.
Measured: resnet18 @64 px, 12k train + 12k val images ≈ 11.6 s/epoch on T4 (CPU here: ~4 min).
External cache (177 MB) is NOT in the default bundle — upload separately when needed.
