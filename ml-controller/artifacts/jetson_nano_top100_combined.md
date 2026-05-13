# Jetson Nano Top 100 Combined Models (85 Unseen + 15 Benchmark)

## Summary

- **Total models:** 98
- **Unseen models:** 85
- **Benchmark models:** 13

## Statistics

**Energy Consumption:**
- Min: 3.9 mWh
- Max: 44.2 mWh
- Mean: 15.2 mWh
- Median: 14.7 mWh

**Latency (estimated):**
- Min: 0.0240s
- Max: 1.0000s
- Mean: 0.7653s

## Architecture Diversity

**Total families:** 55

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
| EfficientNet | 0 | 2 | 2 |
| EfficientNetV2 | 1 | 0 | 1 |
| EfficientNetV3 | 2 | 0 | 2 |
| FBNet | 2 | 0 | 2 |
| FastViT | 2 | 0 | 2 |
| GENet | 1 | 0 | 1 |
| GhostNet | 0 | 2 | 2 |
| HaloNet | 2 | 0 | 2 |
| IResNet | 3 | 0 | 3 |
| Inception | 1 | 0 | 1 |
| LightNet | 1 | 0 | 1 |
| MnasNet | 3 | 0 | 3 |
| MobileAST | 3 | 0 | 3 |
| MobileNet | 0 | 2 | 2 |
| MobileNetPlus | 1 | 0 | 1 |
| MobileNetV3 | 1 | 0 | 1 |
| MobileNetV4 | 2 | 0 | 2 |
| MobileNext | 1 | 0 | 1 |
| MobileOSNet | 1 | 0 | 1 |
| MobileOne | 3 | 0 | 3 |
| MobileViT | 1 | 0 | 1 |
| NasNet | 1 | 0 | 1 |
| OctaveNet | 1 | 0 | 1 |
| Other | 0 | 2 | 2 |
| PVT | 2 | 0 | 2 |
| PiT | 2 | 0 | 2 |
| ProxylessNAS | 2 | 0 | 2 |
| RegNet | 0 | 2 | 2 |
| RegNetX | 5 | 0 | 5 |
| RegNetY | 4 | 0 | 4 |
| RepViT | 2 | 0 | 2 |
| Res2Net | 1 | 0 | 1 |
| ResNeSt | 3 | 0 | 3 |
| ResNet | 0 | 1 | 1 |
| ResNet-D | 2 | 0 | 2 |
| ResNet-RS | 1 | 0 | 1 |
| SENet | 2 | 0 | 2 |
| SKNet | 1 | 0 | 1 |
| ShuffleNetV2 | 5 | 0 | 5 |
| SqueezeNet | 2 | 0 | 2 |
| T2T-ViT | 1 | 0 | 1 |
| TinyViT | 2 | 0 | 2 |
| ViP | 1 | 0 | 1 |
| ViT | 0 | 2 | 2 |
| Xception | 1 | 0 | 1 |

## Top 25 Models

| Rank | Model | Family | Source | Params (M) | GFLOPs | Latency (s) | Energy (mWh) | Ranking Score |
|------|-------|--------|--------|--------|--------|--------|--------|----------|
| 1 | tf_mobilenetv3_small_minimal_100 | MobileNet | benchmark | 2.0 | 0.06 | 0.0240 | 3.9 | 0.9400 |
| 2 | mobilenetv3_small_050 | MobileNet | benchmark | 1.6 | 0.03 | 0.0321 | 4.1 | 0.9364 |
| 3 | semnasnet_050 | Other | benchmark | 2.1 | 0.09 | 0.0400 | 7.4 | 0.9013 |
| 4 | shufflenetv2_x0_25 | ShuffleNetV2 | unseen | 0.5 | 0.04 | 0.0941 | 11.6 | 0.8574 |
| 5 | densenet_tiny | DenseNet | unseen | 0.8 | 0.06 | 0.1348 | 11.7 | 0.8490 |
| 6 | regnetx_002 | RegNet | benchmark | 2.7 | 0.40 | 0.0649 | 12.3 | 0.8460 |
| 7 | regnety_002 | RegNet | benchmark | 3.2 | 0.40 | 0.0676 | 12.4 | 0.8446 |
| 8 | shufflenetv2_x0_33 | ShuffleNetV2 | unseen | 0.6 | 0.07 | 0.1389 | 11.7 | 0.8407 |
| 9 | spnasnet_100 | Other | benchmark | 4.4 | 0.33 | 0.0409 | 14.0 | 0.8359 |
| 10 | mnasnet_0_5 | MnasNet | unseen | 2.2 | 0.11 | 0.2320 | 11.8 | 0.8206 |
| 11 | tf_efficientnet_lite0 | EfficientNet | benchmark | 4.7 | 0.39 | 0.0344 | 16.0 | 0.8175 |
| 12 | ghostnet_130 | GhostNet | benchmark | 7.4 | 0.23 | 0.0572 | 15.9 | 0.8124 |
| 13 | mobileosnet_small | MobileOSNet | unseen | 2.4 | 0.13 | 0.2854 | 11.8 | 0.8036 |
| 14 | mobilenetv4_small | MobileNetV4 | unseen | 3.8 | 0.15 | 0.3187 | 11.8 | 0.7963 |
| 15 | mobilenext_small | MobileNext | unseen | 3.2 | 0.15 | 0.3330 | 11.8 | 0.7939 |
| 16 | ghostnet_100 | GhostNet | benchmark | 3.9 | 0.31 | 0.0550 | 20.2 | 0.7703 |
| 17 | tf_efficientnet_b0 | EfficientNet | benchmark | 5.3 | 0.39 | 0.0520 | 20.3 | 0.7697 |
| 18 | regnety_200mf | RegNetY | unseen | 3.2 | 0.21 | 0.4415 | 12.5 | 0.7079 |
| 19 | regnetx_200mf | RegNetX | unseen | 2.3 | 0.21 | 0.4406 | 12.5 | 0.7043 |
| 20 | mobilenetv3_large_100_miil | MobileNetV3 | unseen | 5.5 | 0.22 | 0.4651 | 12.7 | 0.6951 |
| 21 | vit_tiny_r_s16_p8_224__b4__224 | ViT | benchmark | 6.3 | 1.08 | 0.0596 | 30.1 | 0.6714 |
| 22 | mobilenetplus_small | MobileNetPlus | unseen | 4.1 | 0.23 | 0.5020 | 13.0 | 0.6539 |
| 23 | resnet10t__b4__224 | ResNet | benchmark | 5.4 | 1.94 | 0.0650 | 34.9 | 0.6222 |
| 24 | fbnet_a | FBNet | unseen | 4.3 | 0.27 | 0.5746 | 14.2 | 0.5454 |
| 25 | levit_128 | ViT | benchmark | 0.4 | 9.21 | 0.0578 | 44.2 | 0.5314 |

## Source Breakdown

- **Unseen Models:** 85 real architectures from timm/torchvision (NOT in 360 benchmark)
- **Benchmark Models:** 13 top-performing models from Jetson Nano 360-model dataset
