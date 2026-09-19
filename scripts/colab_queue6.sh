#!/usr/bin/env bash
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
until grep -q "QUEUE5 DONE" tasks/logs/queue5.log 2>/dev/null; do sleep 120; done
echo "### $(date +%H:%M:%S) queue6 C1-init candidates (20 ep)"
scripts/colab_autorun.sh "" "num_workers=2" 6 configs/final_candidates/F11_r18_c1init_trivial_synth_20.yaml configs/final_candidates/F12_r18_c1init_trivial_20.yaml configs/final_candidates/F13_r18_c1init_morph_synth_20.yaml
echo "### $(date +%H:%M:%S) QUEUE6 DONE"
