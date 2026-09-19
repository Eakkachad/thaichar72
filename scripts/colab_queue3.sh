#!/usr/bin/env bash
# Queue 3 (after master): doc-disjoint runs for the augmentation ladder + synthetic, and E-tricks on doc split.
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
until grep -q "MASTER DONE" tasks/logs/master_queue.log; do sleep 120; done
echo "### $(date +%H:%M:%S) queue3 doc-split aug ladder"
scripts/colab_autorun.sh _doc "num_workers=2 split_kind=doc" 5 configs/matrix/B_none_resnet18_64.yaml configs/matrix/B_randaug_resnet18_64.yaml configs/matrix/B_trivial_resnet18_64.yaml configs/matrix/B_morph_resnet18_64.yaml configs/matrix/B6_synth_all_resnet18_64.yaml configs/matrix/E5_geometry_resnet18_64.yaml configs/matrix/E6_onoff_resnet18_64.yaml configs/matrix/D3_sampler_sqrt_resnet18_64.yaml
echo "### $(date +%H:%M:%S) QUEUE3 DONE"
