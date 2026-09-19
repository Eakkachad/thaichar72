# Result Figures & Presentation Visuals

Publication-quality result figures generated from `reports/experiments.csv` and `runs/`.
All figures are saved in both PNG (150 dpi) and SVG formats for presentation slides and technical reports.

| Figure | Image | SVG | Summary Caption |
|---|---|---|---|
| **1. Transfer Modes** | [`fig_A_transfer_modes.png`](fig_A_transfer_modes.png) | [SVG](fig_A_transfer_modes.svg) | Fine-tuning full backbones significantly outperforms frozen feature extraction, with ResNet-18 full reaching 0.9794 balanced accuracy vs 0.5633 when frozen, all beating the previous Rust CNN baseline of 0.967. |
| **2. Aug Ladder** | [`fig_B_aug_ladder.png`](fig_B_aug_ladder.png) | [SVG](fig_B_aug_ladder.svg) | No-aug reaches 0.9815 balanced accuracy on stratified validation, but under document-disjoint shift base-aug drops to 0.9361 while full augmentation maintains 0.9611 balanced accuracy. |
| **3. Mixup & CutMix** | [`fig_B_mix.png`](fig_B_mix.png) | [SVG](fig_B_mix.svg) | Standard full augmentation achieves 0.9644 balanced accuracy, whereas Mixup and CutMix reduce uncalibrated balanced accuracy to 0.9492 and 0.9403 respectively. |
| **4. Imbalance Trade-off** | [`fig_D_imbalance_tradeoff.png`](fig_D_imbalance_tradeoff.png) | [SVG](fig_D_imbalance_tradeoff.svg) | Class-aware sampling achieves the best balanced accuracy at 0.9841 (D3 sampler), whereas naive inverse-frequency loss weighting completely collapses top-1 accuracy to 0.1160 (off-chart: 0.116). |
| **5. Size Sweep** | [`fig_S_size_sweep.png`](fig_S_size_sweep.png) | [SVG](fig_S_size_sweep.svg) | Native 64px delivers 0.9644 balanced accuracy at 15.9 ms, while scaling to 224px only reaches 0.9673 at a 4.5x latency penalty (71.2 ms) as real glyphs are ≤ 44 px. |
| **6. Advanced Tricks** | [`fig_E_tricks.png`](fig_E_tricks.png) | [SVG](fig_E_tricks.svg) | Geometry side-channel features provide a +0.46 pp gain in balanced accuracy over the D0 baseline (0.9644), while LLRD and ON/OFF channels degrade performance on this dataset. |
| **7. Generalization Gap** | [`fig_gap_strat_vs_doc.png`](fig_gap_strat_vs_doc.png) | [SVG](fig_gap_strat_vs_doc.svg) | Generalization to unseen documents shows severe degradation for scratch SmallCNN (-9.03 pp), whereas full augmentation maintains robustness with only a -0.33 pp gap (0.9611 balanced). |
| **8. Training Dynamics** | [`fig_training_curves_best.png`](fig_training_curves_best.png) | [SVG](fig_training_curves_best.svg) | The best stratified model (D3_sampler_inv_cap10_resnet18_64_T4) smoothly converges to 0.9841 balanced accuracy, showing more stable late-epoch gains than B_none. |
| **9. Tau Sweep Grid** | [`fig_tau_sweep_grid.png`](fig_tau_sweep_grid.png) | [SVG](fig_tau_sweep_grid.svg) | Post-hoc logit adjustment at τ=0.75 recovers Mixup balanced accuracy from 0.9492 to 0.9803 (+3.11 pp), while well-calibrated baselines peak near τ=0. |
