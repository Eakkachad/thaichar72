#!/usr/bin/env bash
# Queue 2: waits for the BDE loop to finish, then runs pretraining stages, size sweep and doc-split runs.
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
until grep -q "LOOP EXIT" tasks/logs/matrix_BDE_T4.log; do sleep 60; done
echo "=== queue2 start $(date +%H:%M:%S) ==="
# stage-1 pretraining (external handwritten, synthetic fonts) → stage-2 fine-tune variants
scripts/colab_matrix.sh thai-test "" num_workers=2 -- configs/pretrain/C1_ext_pretrain_resnet18_64.yaml configs/pretrain/B6a_synth_pretrain_resnet18_64.yaml
scripts/colab_matrix.sh thai-test "" num_workers=2 -- configs/pretrain/C1_ft_resnet18_64.yaml configs/pretrain/B6a_ft_resnet18_64.yaml configs/pretrain/C1B6_ft_resnet18_64.yaml
# input-size sweep (resnet18, aug full)
scripts/colab_matrix.sh thai-test _T4 num_workers=2 -- configs/matrix/S_resnet18_32.yaml configs/matrix/S_resnet18_96.yaml configs/matrix/S_resnet18_128.yaml
# document-disjoint honesty runs of the key recipes
scripts/colab_matrix.sh thai-test _doc num_workers=2 split_kind=doc -- configs/matrix/A1_resnet18_full_64.yaml configs/matrix/B_full_resnet18_64.yaml configs/matrix/A3_effb0_full_64.yaml configs/matrix/A0_smallcnn_64.yaml
# 224 px last (slow)
scripts/colab_matrix.sh thai-test _T4 num_workers=2 -- configs/matrix/S_resnet18_224.yaml
echo "=== queue2 done $(date +%H:%M:%S) ==="
