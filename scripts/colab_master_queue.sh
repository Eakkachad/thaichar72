#!/usr/bin/env bash
cd /home/eggchad/eakject/research/Deep_Man/Deep_CNN
L=/tmp/claude-1000/-home-eggchad-eakject-research-Deep-Man/c3139ea7-78d1-493f-a4ab-87bc06edcd72/scratchpad/bde_rest.txt
echo "### $(date +%H:%M:%S) BDE rest"; scripts/colab_autorun.sh _T4 "num_workers=2" 6 $(cat $L)
echo "### $(date +%H:%M:%S) stage-1 pretrain"; scripts/colab_autorun.sh "" "num_workers=2" 4 configs/pretrain/C1_ext_pretrain_resnet18_64.yaml configs/pretrain/B6a_synth_pretrain_resnet18_64.yaml
echo "### $(date +%H:%M:%S) stage-2 finetune"; scripts/colab_autorun.sh "" "num_workers=2" 4 configs/pretrain/C1_ft_resnet18_64.yaml configs/pretrain/B6a_ft_resnet18_64.yaml configs/pretrain/C1B6_ft_resnet18_64.yaml
echo "### $(date +%H:%M:%S) size sweep"; scripts/colab_autorun.sh _T4 "num_workers=2" 4 configs/matrix/S_resnet18_32.yaml configs/matrix/S_resnet18_96.yaml configs/matrix/S_resnet18_128.yaml
echo "### $(date +%H:%M:%S) doc-disjoint"; scripts/colab_autorun.sh _doc "num_workers=2 split_kind=doc" 4 configs/matrix/A1_resnet18_full_64.yaml configs/matrix/B_full_resnet18_64.yaml configs/matrix/A3_effb0_full_64.yaml configs/matrix/A0_smallcnn_64.yaml
echo "### $(date +%H:%M:%S) 224px"; scripts/colab_autorun.sh _T4 "num_workers=2" 3 configs/matrix/S_resnet18_224.yaml
echo "### $(date +%H:%M:%S) MASTER DONE"
