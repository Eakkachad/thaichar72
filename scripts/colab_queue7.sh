#!/usr/bin/env bash
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
until grep -q "QUEUE6 DONE" tasks/logs/queue6.log 2>/dev/null; do sleep 120; done
echo "### $(date +%H:%M:%S) queue7 synth-pretrain-init candidates (20 ep)"
scripts/colab_autorun.sh "" "num_workers=2" 6 configs/final_candidates/F14_r18_b6ainit_trivial_20.yaml configs/final_candidates/F15_r18_b6ainit_trivial_synth_20.yaml configs/final_candidates/F16_r18_b6ainit_morph_20.yaml configs/final_candidates/F17_r18_b6ainit_none_20.yaml
echo "### $(date +%H:%M:%S) QUEUE7 DONE"
