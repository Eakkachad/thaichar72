# Weights

| name | model | img | aug | init | top-1 | balanced | MB | md5 |
|---|---|---:|---|---|---:|---:|---:|---|
| thaichar72_resnet18_64_v1labels | resnet18 | 64 | randaug | runs/B6a_synth_pretrain_resnet18_64/best.pt | 0.9839 | 0.9827 | 44.9 | 9605c6f6 |
| thaichar72_r18_synth_pretrain_init | resnet18 | 64 | full | ImageNet | 0.9369 | 0.9188 | 44.9 | 3d667ccd |
| thaichar72_resnet18_64 | resnet18 | 64 | randaug | runs/B6a_synth_pretrain_resnet18_64/best.pt | 0.9903 | 0.9875 | 44.9 | 467bb628 |
| thaichar72_resnet18_64_fp16 | resnet18 | 64 | randaug | runs/B6a_synth_pretrain_resnet18_64/best.pt | 0.9903 | 0.9875 | 22.5 | fce6c261 |
| thaichar72_mnv3small_64_small | mobilenetv3_small_100 | 64 | randaug | ImageNet | 0.9900 | 0.9872 | 6.5 | 8a7aaa6c |
| thaichar72_r18_64_gen | resnet18 | 64 | randaug | runs/G2_synth203du_pretrain_r18_64/best.pt | 0.9902 | 0.9934 | 44.9 | 90ea7a3a |
| thaichar72_r18_64_gen_fp16 | resnet18 | 64 | randaug | runs/G2_synth203du_pretrain_r18_64/best.pt | 0.9902 | 0.9934 | 22.5 | 96f9c8d4 |
| thaichar72_r18_synth203du_pretrain_init | resnet18 | 64 | full | ImageNet | 0.9396 | 0.9354 | 44.9 | 30aa5ef0 |
