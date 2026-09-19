#!/usr/bin/env bash
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
until grep -q "QUEUE4 DONE" tasks/logs/queue4.log 2>/dev/null; do sleep 120; done
echo "### $(date +%H:%M:%S) queue5 trivial candidates (20 ep)"
scripts/colab_autorun.sh "" "num_workers=2" 6 configs/final_candidates/F8_r18_trivial_20.yaml configs/final_candidates/F9_r18_trivial_synth_20.yaml configs/final_candidates/F10_effb0_trivial_synth_20.yaml
echo "### $(date +%H:%M:%S) QUEUE5 DONE"
