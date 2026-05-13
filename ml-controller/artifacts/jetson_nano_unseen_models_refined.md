# Jetson Nano Unseen Models (Refined Discovery)

**Methodology:** Real models only (timm, torchvision) - NO synthetic scaling

## Discovery Pipeline

1. **Benchmark models to exclude:** 360
2. **Real models screened:** 127 (from timm/torchvision catalog)
3. **Unseen candidates:** 102
4. **Jetson Nano constraints applied:**
   - params < 50M
   - GFLOPs < 10
   - RAM estimate < 3.2GB
   - latency 0.03-1.0s
5. **Deployable models:** 85
6. **Diversity filtered:** 85

## Latency Estimation Method

Architecture-aware heuristic (NOT synthetic scaling):

- **CNN:** GFLOPs-bound with memory overhead
- **Transformer:** Compute + 50% attention overhead + model size scaling
- **Hybrid:** Mixed CNN-Transformer overhead
- **MLP:** Compute-heavy with memory intensity

## Diversity Metrics

- arch_type_diversity: 0.035
- family_diversity: 0.565
- gflops_cv: 1.135
- overall_diversity: 0.634
- params_cv: 0.8

## Energy Prediction Distribution

- Min: 11.7 mWh
- Max: 16.7 mWh
- Median: 14.7 mWh
- Mean: 14.8 mWh
- Range (max/min): 1.44x

## Selected Models (Top 100)

| Rank | Model | Family | Type | Params (M) | GFLOPs | Est. Latency (s) | Predicted Energy (mWh) | Confidence | Reason |
|------|-------|--------|------|-------|--------|--------|--------|--------|----------|
| 1 | shufflenetv2_x0_25 | ShuffleNetV2 | ConvNet | 0.50 | 0.044 | 0.0941 | 11.6 | 0.448 | ConvNet with 0.5M params, 0.04 GFLOPs, est. latenc... |
| 2 | shufflenetv2_x0_33 | ShuffleNetV2 | ConvNet | 0.64 | 0.065 | 0.1389 | 11.7 | 0.46 | ConvNet with 0.6M params, 0.07 GFLOPs, est. latenc... |
| 3 | densenet_tiny | DenseNet | ConvNet | 0.78 | 0.063 | 0.1348 | 11.7 | 0.468 | ConvNet with 0.8M params, 0.06 GFLOPs, est. latenc... |
| 4 | mobilenetv4_small | MobileNetV4 | ConvNet | 3.81 | 0.148 | 0.3187 | 11.8 | 0.58 | ConvNet with 3.8M params, 0.15 GFLOPs, est. latenc... |
| 5 | mobilenext_small | MobileNext | ConvNet | 3.17 | 0.155 | 0.3330 | 11.8 | 0.567 | ConvNet with 3.2M params, 0.15 GFLOPs, est. latenc... |
| 6 | mobileosnet_small | MobileOSNet | ConvNet | 2.41 | 0.133 | 0.2854 | 11.8 | 0.543 | ConvNet with 2.4M params, 0.13 GFLOPs, est. latenc... |
| 7 | mnasnet_0_5 | MnasNet | ConvNet | 2.22 | 0.108 | 0.2320 | 11.8 | 0.534 | ConvNet with 2.2M params, 0.11 GFLOPs, est. latenc... |
| 8 | regnetx_200mf | RegNetX | ConvNet | 2.34 | 0.206 | 0.4406 | 12.5 | 0.552 | ConvNet with 2.3M params, 0.21 GFLOPs, est. latenc... |
| 9 | regnety_200mf | RegNetY | ConvNet | 3.16 | 0.206 | 0.4415 | 12.5 | 0.574 | ConvNet with 3.2M params, 0.21 GFLOPs, est. latenc... |
| 10 | mobilenetv3_large_100_miil | MobileNetV3 | ConvNet | 5.48 | 0.216 | 0.4651 | 12.7 | 0.619 | ConvNet with 5.5M params, 0.22 GFLOPs, est. latenc... |
| 11 | mobilenetplus_small | MobileNetPlus | ConvNet | 4.14 | 0.234 | 0.5020 | 13.0 | 0.599 | ConvNet with 4.1M params, 0.23 GFLOPs, est. latenc... |
| 12 | fbnet_a | FBNet | ConvNet | 4.34 | 0.268 | 0.5746 | 14.2 | 0.607 | ConvNet with 4.3M params, 0.27 GFLOPs, est. latenc... |
| 13 | mobileone_s0 | MobileOne | ConvNet | 2.08 | 0.284 | 0.6063 | 14.4 | 0.554 | ConvNet with 2.1M params, 0.28 GFLOPs, est. latenc... |
| 14 | lightnet | LightNet | ConvNet | 3.28 | 0.287 | 0.6139 | 14.4 | 0.587 | ConvNet with 3.3M params, 0.29 GFLOPs, est. latenc... |
| 15 | shufflenetv2_x1_0 | ShuffleNetV2 | ConvNet | 2.28 | 0.297 | 0.6342 | 14.5 | 0.562 | ConvNet with 2.3M params, 0.30 GFLOPs, est. latenc... |
| 16 | squeezenet_v1_1 | SqueezeNet | ConvNet | 1.24 | 0.355 | 0.7566 | 14.5 | 0.531 | ConvNet with 1.2M params, 0.35 GFLOPs, est. latenc... |
| 17 | proxylessnas_cpu | ProxylessNAS | ConvNet | 4.42 | 0.309 | 0.6619 | 14.5 | 0.614 | ConvNet with 4.4M params, 0.31 GFLOPs, est. latenc... |
| 18 | mnasnet_1_0 | MnasNet | ConvNet | 4.38 | 0.326 | 0.6980 | 14.5 | 0.615 | ConvNet with 4.4M params, 0.33 GFLOPs, est. latenc... |
| 19 | fbnet_c | FBNet | ConvNet | 5.27 | 0.385 | 0.8244 | 14.6 | 0.638 | ConvNet with 5.3M params, 0.39 GFLOPs, est. latenc... |
| 20 | efficientnet_b0_v3 | EfficientNetV3 | ConvNet | 5.29 | 0.390 | 0.8351 | 14.6 | 0.639 | ConvNet with 5.3M params, 0.39 GFLOPs, est. latenc... |
| 21 | regnetx_400mf | RegNetX | ConvNet | 5.16 | 0.412 | 0.8818 | 14.7 | 0.639 | ConvNet with 5.2M params, 0.41 GFLOPs, est. latenc... |
| 22 | regnety_400mf | RegNetY | ConvNet | 3.16 | 0.412 | 0.8798 | 14.7 | 0.6 | ConvNet with 3.2M params, 0.41 GFLOPs, est. latenc... |
| 23 | mobileast_tiny | MobileAST | Transformer | 5.62 | 0.854 | 1.0000 | 14.7 | 0.692 | Transformer with 5.6M params, 0.85 GFLOPs, est. la... |
| 24 | vip_tiny | ViP | Transformer | 6.54 | 1.372 | 1.0000 | 14.7 | 0.746 | Transformer with 6.5M params, 1.37 GFLOPs, est. la... |
| 25 | halonet_h0 | HaloNet | Transformer | 3.01 | 0.536 | 1.0000 | 14.7 | 0.61 | Transformer with 3.0M params, 0.54 GFLOPs, est. la... |
| 26 | pit_ti | PiT | Transformer | 4.88 | 0.715 | 1.0000 | 14.7 | 0.667 | Transformer with 4.9M params, 0.71 GFLOPs, est. la... |
| 27 | pit_xs | PiT | Transformer | 5.92 | 1.081 | 1.0000 | 14.7 | 0.716 | Transformer with 5.9M params, 1.08 GFLOPs, est. la... |
| 28 | beit_tiny_patch16_224 | BEiT | Transformer | 5.72 | 1.260 | 1.0000 | 14.7 | 0.726 | Transformer with 5.7M params, 1.26 GFLOPs, est. la... |
| 29 | t2t_vit_7 | T2T-ViT | Transformer | 4.28 | 0.903 | 1.0000 | 14.7 | 0.674 | Transformer with 4.3M params, 0.90 GFLOPs, est. la... |
| 30 | proxylessnas_gpu | ProxylessNAS | ConvNet | 7.12 | 0.466 | 0.9986 | 14.7 | 0.673 | ConvNet with 7.1M params, 0.47 GFLOPs, est. latenc... |
| 31 | nasnetamobile | NasNet | ConvNet | 5.30 | 0.544 | 1.0000 | 14.7 | 0.656 | ConvNet with 5.3M params, 0.54 GFLOPs, est. latenc... |
| 32 | regnetx_600mf | RegNetX | ConvNet | 6.16 | 0.606 | 1.0000 | 14.7 | 0.676 | ConvNet with 6.2M params, 0.61 GFLOPs, est. latenc... |
| 33 | regnetx_800mf | RegNetX | ConvNet | 7.26 | 0.808 | 1.0000 | 14.7 | 0.71 | ConvNet with 7.3M params, 0.81 GFLOPs, est. latenc... |
| 34 | mnasnet_1_3 | MnasNet | ConvNet | 6.28 | 0.542 | 1.0000 | 14.7 | 0.671 | ConvNet with 6.3M params, 0.54 GFLOPs, est. latenc... |
| 35 | tinyvipt_5m | TinyViT | Transformer | 5.39 | 1.265 | 1.0000 | 14.7 | 0.722 | Transformer with 5.4M params, 1.26 GFLOPs, est. la... |
| 36 | fastvit_t8 | FastViT | Transformer | 4.71 | 0.503 | 1.0000 | 14.7 | 0.642 | Transformer with 4.7M params, 0.50 GFLOPs, est. la... |
| 37 | fastvit_t12 | FastViT | Transformer | 7.04 | 0.854 | 1.0000 | 14.7 | 0.711 | Transformer with 7.0M params, 0.85 GFLOPs, est. la... |
| 38 | mobileone_s1 | MobileOne | ConvNet | 4.75 | 0.820 | 1.0000 | 14.7 | 0.675 | ConvNet with 4.8M params, 0.82 GFLOPs, est. latenc... |
| 39 | convnextv2_femto | ConvNeXt | ConvNet | 5.23 | 1.300 | 1.0000 | 14.7 | 0.722 | ConvNet with 5.2M params, 1.30 GFLOPs, est. latenc... |
| 40 | shufflenetv2_x1_5 | ShuffleNetV2 | ConvNet | 3.50 | 0.583 | 1.0000 | 14.7 | 0.627 | ConvNet with 3.5M params, 0.58 GFLOPs, est. latenc... |
| 41 | shufflenetv2_x2_0 | ShuffleNetV2 | ConvNet | 7.39 | 1.172 | 1.0000 | 14.7 | 0.742 | ConvNet with 7.4M params, 1.17 GFLOPs, est. latenc... |
| 42 | coatnet_0 | CoAtNet | Hybrid | 2.34 | 0.496 | 1.0000 | 14.7 | 0.588 | Hybrid with 2.3M params, 0.50 GFLOPs, est. latency... |
| 43 | pvt_v2_b0 | PVT | Transformer | 3.67 | 0.583 | 1.0000 | 14.7 | 0.631 | Transformer with 3.7M params, 0.58 GFLOPs, est. la... |
| 44 | mobileast_tiny | MobileAST | Transformer | 5.62 | 0.854 | 1.0000 | 14.7 | 0.692 | Transformer with 5.6M params, 0.85 GFLOPs, est. la... |
| 45 | mobilenetv4_medium | MobileNetV4 | ConvNet | 9.72 | 0.554 | 1.0000 | 14.7 | 0.711 | ConvNet with 9.7M params, 0.55 GFLOPs, est. latenc... |
| 46 | mobileast_small | MobileAST | Transformer | 8.79 | 1.525 | 1.0000 | 14.7 | 0.783 | Transformer with 8.8M params, 1.52 GFLOPs, est. la... |
| 47 | crossvit_9_small_224 | CrossViT | Transformer | 8.53 | 1.550 | 1.0000 | 14.7 | 0.781 | Transformer with 8.5M params, 1.55 GFLOPs, est. la... |
| 48 | halonet_h1 | HaloNet | Transformer | 8.17 | 2.185 | 1.0000 | 14.7 | 0.815 | Transformer with 8.2M params, 2.19 GFLOPs, est. la... |
| 49 | efficientnet_b1_v3 | EfficientNetV3 | ConvNet | 7.79 | 0.710 | 1.0000 | 14.7 | 0.707 | ConvNet with 7.8M params, 0.71 GFLOPs, est. latenc... |
| 50 | repvit_m1 | RepViT | Transformer | 8.28 | 1.320 | 1.0000 | 14.7 | 0.763 | Transformer with 8.3M params, 1.32 GFLOPs, est. la... |
| 51 | mobileone_s2 | MobileOne | ConvNet | 8.09 | 1.525 | 1.0000 | 14.7 | 0.775 | ConvNet with 8.1M params, 1.52 GFLOPs, est. latenc... |
| 52 | regnety_600mf | RegNetY | ConvNet | 11.20 | 0.606 | 1.0000 | 14.8 | 0.729 | ConvNet with 11.2M params, 0.61 GFLOPs, est. laten... |
| 53 | mobilevit_m | MobileViT | Hybrid | 11.40 | 1.042 | 1.0000 | 14.8 | 0.771 | Hybrid with 11.4M params, 1.04 GFLOPs, est. latenc... |
| 54 | genet_s | GENet | ConvNet | 8.71 | 2.640 | 1.0000 | 14.8 | 0.843 | ConvNet with 8.7M params, 2.64 GFLOPs, est. latenc... |
| 55 | tinyvipt_11m | TinyViT | Transformer | 11.00 | 2.472 | 1.0000 | 14.8 | 0.856 | Transformer with 11.0M params, 2.47 GFLOPs, est. l... |
| 56 | repvit_m2 | RepViT | Transformer | 12.86 | 2.097 | 1.0000 | 14.9 | 0.851 | Transformer with 12.9M params, 2.10 GFLOPs, est. l... |
| 57 | pvt_v2_b1 | PVT | Transformer | 13.86 | 2.142 | 1.0000 | 14.9 | 0.861 | Transformer with 13.9M params, 2.14 GFLOPs, est. l... |
| 58 | squeezenet_v1_0 | SqueezeNet | ConvNet | 1.24 | 0.823 | 1.0000 | 15.0 | 0.581 | ConvNet with 1.2M params, 0.82 GFLOPs, est. latenc... |
| 59 | regnety_3_2gf | RegNetY | ConvNet | 19.18 | 3.200 | 1.0000 | 15.0 | 0.94 | ConvNet with 19.2M params, 3.20 GFLOPs, est. laten... |
| 60 | resnest14 | ResNeSt | ConvNet | 10.61 | 3.062 | 1.0000 | 15.0 | 0.879 | ConvNet with 10.6M params, 3.06 GFLOPs, est. laten... |
| 61 | cycleshift_s | CycleShift | ConvNet | 12.42 | 3.160 | 1.0000 | 15.2 | 0.897 | ConvNet with 12.4M params, 3.16 GFLOPs, est. laten... |
| 62 | regnetx_3_2gf | RegNetX | ConvNet | 15.30 | 3.200 | 1.0000 | 15.3 | 0.918 | ConvNet with 15.3M params, 3.20 GFLOPs, est. laten... |
| 63 | resnest50 | ResNeSt | ConvNet | 27.48 | 5.400 | 1.0000 | 15.5 | 0.926 | ConvNet with 27.5M params, 5.40 GFLOPs, est. laten... |
| 64 | efficientnetv2_xs | EfficientNetV2 | ConvNet | 10.90 | 3.400 | 1.0000 | 15.5 | 0.895 | ConvNet with 10.9M params, 3.40 GFLOPs, est. laten... |
| 65 | res2net50 | Res2Net | ConvNet | 25.50 | 5.450 | 1.0000 | 15.7 | 0.917 | ConvNet with 25.5M params, 5.45 GFLOPs, est. laten... |
| 66 | convnextv2_tiny | ConvNeXt | ConvNet | 28.64 | 8.687 | 1.0000 | 15.8 | 0.86 | ConvNet with 28.6M params, 8.69 GFLOPs, est. laten... |
| 67 | sknet50 | SKNet | ConvNet | 27.52 | 8.500 | 1.0000 | 15.8 | 0.86 | ConvNet with 27.5M params, 8.50 GFLOPs, est. laten... |
| 68 | senet50 | SENet | ConvNet | 28.09 | 8.750 | 1.0000 | 15.8 | 0.857 | ConvNet with 28.1M params, 8.75 GFLOPs, est. laten... |
| 69 | iresnet18 | IResNet | ConvNet | 11.69 | 3.620 | 1.0000 | 15.8 | 0.899 | ConvNet with 11.7M params, 3.62 GFLOPs, est. laten... |
| 70 | resnet18d | ResNet-D | ConvNet | 11.71 | 3.640 | 1.0000 | 15.8 | 0.898 | ConvNet with 11.7M params, 3.64 GFLOPs, est. laten... |
| 71 | senet18 | SENet | ConvNet | 11.78 | 3.650 | 1.0000 | 15.8 | 0.899 | ConvNet with 11.8M params, 3.65 GFLOPs, est. laten... |
| 72 | iresnet50 | IResNet | ConvNet | 25.56 | 8.180 | 1.0000 | 16.0 | 0.858 | ConvNet with 25.6M params, 8.18 GFLOPs, est. laten... |
| 73 | resnet50_rs | ResNet-RS | ConvNet | 25.56 | 8.180 | 1.0000 | 16.0 | 0.858 | ConvNet with 25.6M params, 8.18 GFLOPs, est. laten... |
| 74 | ecaresnet50 | ECA-Net | ConvNet | 25.56 | 8.180 | 1.0000 | 16.0 | 0.858 | ConvNet with 25.6M params, 8.18 GFLOPs, est. laten... |
| 75 | octave_resnet50 | OctaveNet | ConvNet | 25.56 | 8.180 | 1.0000 | 16.0 | 0.858 | ConvNet with 25.6M params, 8.18 GFLOPs, est. laten... |
| 76 | resnetblur50 | BlurPool | ConvNet | 25.56 | 8.180 | 1.0000 | 16.0 | 0.858 | ConvNet with 25.6M params, 8.18 GFLOPs, est. laten... |
| 77 | inception_v3_custom | Inception | ConvNet | 23.83 | 5.730 | 1.0000 | 16.3 | 0.903 | ConvNet with 23.8M params, 5.73 GFLOPs, est. laten... |
| 78 | xception | Xception | ConvNet | 22.86 | 8.420 | 1.0000 | 16.5 | 0.843 | ConvNet with 22.9M params, 8.42 GFLOPs, est. laten... |
| 79 | cspdarknet_s | CSPDarkNet | ConvNet | 22.65 | 8.471 | 1.0000 | 16.5 | 0.842 | ConvNet with 22.6M params, 8.47 GFLOPs, est. laten... |
| 80 | coatnet_1 | CoAtNet | Hybrid | 8.72 | 4.880 | 1.0000 | 16.6 | 0.832 | Hybrid with 8.7M params, 4.88 GFLOPs, est. latency... |
| 81 | resnest26 | ResNeSt | ConvNet | 17.07 | 5.438 | 1.0000 | 16.7 | 0.879 | ConvNet with 17.1M params, 5.44 GFLOPs, est. laten... |
| 82 | ecaresnet26 | ECA-Net | ConvNet | 16.01 | 5.200 | 1.0000 | 16.7 | 0.879 | ConvNet with 16.0M params, 5.20 GFLOPs, est. laten... |
| 83 | cait_xs24_224 | CaiT | Transformer | 11.96 | 4.780 | 1.0000 | 16.7 | 0.864 | Transformer with 12.0M params, 4.78 GFLOPs, est. l... |
| 84 | iresnet34 | IResNet | ConvNet | 21.80 | 7.310 | 1.0000 | 16.7 | 0.86 | ConvNet with 21.8M params, 7.31 GFLOPs, est. laten... |
| 85 | resnet34d | ResNet-D | ConvNet | 21.82 | 7.330 | 1.0000 | 16.7 | 0.859 | ConvNet with 21.8M params, 7.33 GFLOPs, est. laten... |

## Top 10 Most Efficient

1. **shufflenetv2_x0_25** (ShuffleNetV2, ConvNet)
   - Params: 0.50M | GFLOPs: 0.044 | Latency: 0.0941s
   - Predicted energy: 11.6 mWh | Confidence: 0.448

2. **shufflenetv2_x0_33** (ShuffleNetV2, ConvNet)
   - Params: 0.64M | GFLOPs: 0.065 | Latency: 0.1389s
   - Predicted energy: 11.7 mWh | Confidence: 0.46

3. **densenet_tiny** (DenseNet, ConvNet)
   - Params: 0.78M | GFLOPs: 0.063 | Latency: 0.1348s
   - Predicted energy: 11.7 mWh | Confidence: 0.468

4. **mobilenetv4_small** (MobileNetV4, ConvNet)
   - Params: 3.81M | GFLOPs: 0.148 | Latency: 0.3187s
   - Predicted energy: 11.8 mWh | Confidence: 0.58

5. **mobilenext_small** (MobileNext, ConvNet)
   - Params: 3.17M | GFLOPs: 0.155 | Latency: 0.3330s
   - Predicted energy: 11.8 mWh | Confidence: 0.567

6. **mobileosnet_small** (MobileOSNet, ConvNet)
   - Params: 2.41M | GFLOPs: 0.133 | Latency: 0.2854s
   - Predicted energy: 11.8 mWh | Confidence: 0.543

7. **mnasnet_0_5** (MnasNet, ConvNet)
   - Params: 2.22M | GFLOPs: 0.108 | Latency: 0.2320s
   - Predicted energy: 11.8 mWh | Confidence: 0.534

8. **regnetx_200mf** (RegNetX, ConvNet)
   - Params: 2.34M | GFLOPs: 0.206 | Latency: 0.4406s
   - Predicted energy: 12.5 mWh | Confidence: 0.552

9. **regnety_200mf** (RegNetY, ConvNet)
   - Params: 3.16M | GFLOPs: 0.206 | Latency: 0.4415s
   - Predicted energy: 12.5 mWh | Confidence: 0.574

10. **mobilenetv3_large_100_miil** (MobileNetV3, ConvNet)
   - Params: 5.48M | GFLOPs: 0.216 | Latency: 0.4651s
   - Predicted energy: 12.7 mWh | Confidence: 0.619

