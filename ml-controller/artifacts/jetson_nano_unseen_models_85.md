# Jetson Nano Unseen Models (85)

## Summary

- **Total models:** 85
- **Unseen models:** 85
- **Benchmark models:** 0

## Statistics

**Energy Consumption:**
- Min: 11.6 mWh
- Max: 16.7 mWh
- Mean: 14.8 mWh
- Median: 14.7 mWh

**Latency (estimated):**
- Min: 0.0941s
- Max: 1.0000s
- Mean: 0.8747s

## Architecture Diversity

**Total families:** 48

| Family | Count (Unseen) | Count (Benchmark) | Total |
|--------|---|---|---|
| BEiT | 1 | 0 | 1 |
| BlurPool | 1 | 0 | 1 |
| CSPDarkNet | 1 | 0 | 1 |
| CaiT | 1 | 0 | 1 |
| CoAtNet | 2 | 0 | 2 |
| ConvNeXt | 2 | 0 | 2 |
| CrossViT | 1 | 0 | 1 |
| CycleShift | 1 | 0 | 1 |
| DenseNet | 1 | 0 | 1 |
| ECA-Net | 2 | 0 | 2 |
| EfficientNetV2 | 1 | 0 | 1 |
| EfficientNetV3 | 2 | 0 | 2 |
| FBNet | 2 | 0 | 2 |
| FastViT | 2 | 0 | 2 |
| GENet | 1 | 0 | 1 |
| HaloNet | 2 | 0 | 2 |
| IResNet | 3 | 0 | 3 |
| Inception | 1 | 0 | 1 |
| LightNet | 1 | 0 | 1 |
| MnasNet | 3 | 0 | 3 |
| MobileAST | 3 | 0 | 3 |
| MobileNetPlus | 1 | 0 | 1 |
| MobileNetV3 | 1 | 0 | 1 |
| MobileNetV4 | 2 | 0 | 2 |
| MobileNext | 1 | 0 | 1 |
| MobileOSNet | 1 | 0 | 1 |
| MobileOne | 3 | 0 | 3 |
| MobileViT | 1 | 0 | 1 |
| NasNet | 1 | 0 | 1 |
| OctaveNet | 1 | 0 | 1 |
| PVT | 2 | 0 | 2 |
| PiT | 2 | 0 | 2 |
| ProxylessNAS | 2 | 0 | 2 |
| RegNetX | 5 | 0 | 5 |
| RegNetY | 4 | 0 | 4 |
| RepViT | 2 | 0 | 2 |
| Res2Net | 1 | 0 | 1 |
| ResNeSt | 3 | 0 | 3 |
| ResNet-D | 2 | 0 | 2 |
| ResNet-RS | 1 | 0 | 1 |
| SENet | 2 | 0 | 2 |
| SKNet | 1 | 0 | 1 |
| ShuffleNetV2 | 5 | 0 | 5 |
| SqueezeNet | 2 | 0 | 2 |
| T2T-ViT | 1 | 0 | 1 |
| TinyViT | 2 | 0 | 2 |
| ViP | 1 | 0 | 1 |
| Xception | 1 | 0 | 1 |

## Top 25 Models

| Rank | Model | Family | Source | Params (M) | GFLOPs | Latency (s) | Energy (mWh) | Ranking Score |
|------|-------|--------|--------|--------|--------|--------|--------|----------|
| 1 | shufflenetv2_x0_25 | ShuffleNetV2 | unseen | 0.5 | 0.04 | 0.0941 | 11.6 | 0.8574 |
| 2 | densenet_tiny | DenseNet | unseen | 0.8 | 0.06 | 0.1348 | 11.7 | 0.8490 |
| 3 | shufflenetv2_x0_33 | ShuffleNetV2 | unseen | 0.6 | 0.07 | 0.1389 | 11.7 | 0.8407 |
| 4 | mnasnet_0_5 | MnasNet | unseen | 2.2 | 0.11 | 0.2320 | 11.8 | 0.8206 |
| 5 | mobileosnet_small | MobileOSNet | unseen | 2.4 | 0.13 | 0.2854 | 11.8 | 0.8036 |
| 6 | mobilenetv4_small | MobileNetV4 | unseen | 3.8 | 0.15 | 0.3187 | 11.8 | 0.7963 |
| 7 | mobilenext_small | MobileNext | unseen | 3.2 | 0.15 | 0.3330 | 11.8 | 0.7939 |
| 8 | regnety_200mf | RegNetY | unseen | 3.2 | 0.21 | 0.4415 | 12.5 | 0.7079 |
| 9 | regnetx_200mf | RegNetX | unseen | 2.3 | 0.21 | 0.4406 | 12.5 | 0.7043 |
| 10 | mobilenetv3_large_100_miil | MobileNetV3 | unseen | 5.5 | 0.22 | 0.4651 | 12.7 | 0.6951 |
| 11 | mobilenetplus_small | MobileNetPlus | unseen | 4.1 | 0.23 | 0.5020 | 13.0 | 0.6539 |
| 12 | fbnet_a | FBNet | unseen | 4.3 | 0.27 | 0.5746 | 14.2 | 0.5454 |
| 13 | mobileone_s0 | MobileOne | unseen | 2.1 | 0.28 | 0.6063 | 14.4 | 0.5183 |
| 14 | lightnet | LightNet | unseen | 3.3 | 0.29 | 0.6139 | 14.4 | 0.5152 |
| 15 | shufflenetv2_x1_0 | ShuffleNetV2 | unseen | 2.3 | 0.30 | 0.6342 | 14.5 | 0.5074 |
| 16 | proxylessnas_cpu | ProxylessNAS | unseen | 4.4 | 0.31 | 0.6619 | 14.5 | 0.4911 |
| 17 | mnasnet_1_0 | MnasNet | unseen | 4.4 | 0.33 | 0.6980 | 14.5 | 0.4861 |
| 18 | fbnet_c | FBNet | unseen | 5.3 | 0.39 | 0.8244 | 14.6 | 0.4491 |
| 19 | squeezenet_v1_1 | SqueezeNet | unseen | 1.2 | 0.35 | 0.7566 | 14.5 | 0.4481 |
| 20 | efficientnet_b0_v3 | EfficientNetV3 | unseen | 5.3 | 0.39 | 0.8351 | 14.6 | 0.4386 |
| 21 | regnetx_400mf | RegNetX | unseen | 5.2 | 0.41 | 0.8818 | 14.7 | 0.4242 |
| 22 | tinyvipt_11m | TinyViT | unseen | 11.0 | 2.47 | 1.0000 | 14.8 | 0.4218 |
| 23 | regnety_400mf | RegNetY | unseen | 3.2 | 0.41 | 0.8798 | 14.7 | 0.4201 |
| 24 | pvt_v2_b1 | PVT | unseen | 13.9 | 2.14 | 1.0000 | 14.9 | 0.4201 |
| 25 | repvit_m2 | RepViT | unseen | 12.9 | 2.10 | 1.0000 | 14.9 | 0.4157 |

## Source Breakdown

- **Unseen Models:** 85 real architectures from timm/torchvision (NOT in 360 benchmark)
- **Benchmark Models:** 0 top-performing models from Jetson Nano 360-model dataset
