#!/bin/sh
# Fetch Thai typefaces from the Google Fonts repo (OFL) for Stage-A synthetic pretraining.
# Idempotent: skips files already present. Run from CNN/assets/fonts/.
set -u
BASE="https://raw.githubusercontent.com/google/fonts/main/ofl"
# dir:File.ttf  — chosen for coverage of BOTH looped/traditional Thai (closest to the
# corpus typeface) and loopless/modern Thai (diversity).
SET="
sarabun:Sarabun-Regular.ttf
sarabun:Sarabun-Bold.ttf
maitree:Maitree-Regular.ttf
maitree:Maitree-Bold.ttf
pridi:Pridi-Regular.ttf
taviraj:Taviraj-Regular.ttf
trirong:Trirong-Regular.ttf
charm:Charm-Regular.ttf
charmonman:Charmonman-Regular.ttf
sriracha:Sriracha-Regular.ttf
mali:Mali-Regular.ttf
itim:Itim-Regular.ttf
chonburi:Chonburi-Regular.ttf
srisakdi:Srisakdi-Regular.ttf
athiti:Athiti-Regular.ttf
kanit:Kanit-Regular.ttf
prompt:Prompt-Regular.ttf
mitr:Mitr-Regular.ttf
krub:Krub-Regular.ttf
niramit:Niramit-Regular.ttf
koho:KoHo-Regular.ttf
fahkwang:Fahkwang-Regular.ttf
chakrapetch:ChakraPetch-Regular.ttf
k2d:K2D-Regular.ttf
baijamjuree:BaiJamjuree-Regular.ttf
pattaya:Pattaya-Regular.ttf
"
ok=0; fail=0
for e in $SET; do
  d=$(echo "$e" | cut -d: -f1); f=$(echo "$e" | cut -d: -f2)
  [ -s "$f" ] && { ok=$((ok+1)); continue; }
  if curl -fsSL --max-time 40 -o "$f.part" "$BASE/$d/$f" && [ -s "$f.part" ]; then
    mv "$f.part" "$f"; ok=$((ok+1)); echo "ok   $f"
  else
    rm -f "$f.part"; fail=$((fail+1)); echo "MISS $d/$f"
  fi
done
echo "fetched=$ok missing=$fail"
