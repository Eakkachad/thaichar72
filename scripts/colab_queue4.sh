#!/usr/bin/env bash
# Queue 4 (after queue3): stage-2 fine-tunes (fixed init upload) + 20-epoch final candidates with TTA.
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
until grep -q "QUEUE3 DONE" tasks/logs/queue3.log 2>/dev/null; do sleep 120; done
echo "### $(date +%H:%M:%S) queue4 stage-2"
scripts/colab_autorun.sh "" "num_workers=2" 4 configs/pretrain/C1_ft_resnet18_64.yaml configs/pretrain/B6a_ft_resnet18_64.yaml configs/pretrain/C1B6_ft_resnet18_64.yaml
echo "### $(date +%H:%M:%S) queue4 final candidates (20 ep)"
scripts/colab_autorun.sh "" "num_workers=2" 8 configs/final_candidates/F1_r18_none_20.yaml configs/final_candidates/F2_r18_randaug_20.yaml configs/final_candidates/F3_r18_randaug_synth_20.yaml configs/final_candidates/F4_effb0_randaug_synth_20.yaml configs/final_candidates/F5_r18_trivial_synth_geo_20.yaml configs/final_candidates/F6_r18_randaug_synth_onoff_20.yaml configs/final_candidates/F7_r18_morph_synth_20.yaml
echo "### $(date +%H:%M:%S) QUEUE4 DONE"
