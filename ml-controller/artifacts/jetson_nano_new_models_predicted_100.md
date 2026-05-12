# Jetson Nano 100 New Models with Predicted Energy

**Source:** Scaled variants of benchmark model families + popular models outside 360-model benchmark

**Criteria:** All models NOT in Jetson 360 benchmark; latency range p1-p99 of benchmark

**Total available:** 515 generated candidates

**Selected:** 100 by lowest GFLOPs + params_m


| Rank | Model | Params (M) | GFLOPs | GMACs | Size (MB) | Latency (s) | Source | Predicted Energy (mWh) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | edgenext_xx_small_new | 1.330 | 0.266 | 0.133 | 5.20 | 0.0200 | popular_outside_benchmark | 8.5 |
| 2 | regnetv_150_scaled | 30.961 | 1.305 | 0.652 | 118.11 | 0.1217 | scaled_variant | 8.5 |
| 3 | shufflenet_v2_x0_5_new | 1.370 | 0.041 | 0.020 | 5.50 | 0.0320 | popular_outside_benchmark | 8.5 |
| 4 | regnety_030_scaled | 0.949 | 0.120 | 0.060 | 3.71 | 0.0417 | scaled_variant | 8.6 |
| 5 | mobilenetv3_200_scaled | 3.186 | 0.060 | 0.030 | 12.16 | 0.0423 | scaled_variant | 8.6 |
| 6 | regnetx_030_scaled | 0.805 | 0.119 | 0.060 | 3.16 | 0.0401 | scaled_variant | 8.6 |
| 7 | spnasnet_200_scaled | 8.843 | 0.660 | 0.330 | 33.73 | 0.0540 | scaled_variant | 8.6 |
| 8 | semnasnet_200_scaled | 4.164 | 0.170 | 0.086 | 15.88 | 0.0527 | scaled_variant | 8.6 |
| 9 | regnetx_075_scaled | 2.014 | 0.298 | 0.149 | 7.89 | 0.0578 | scaled_variant | 8.6 |
| 10 | semnasnet_125_scaled | 2.603 | 0.106 | 0.054 | 9.93 | 0.0437 | scaled_variant | 8.6 |
| 11 | mobilenetv3_125_scaled | 1.992 | 0.037 | 0.019 | 7.60 | 0.0351 | scaled_variant | 8.6 |
| 12 | mobilenetv3_150_scaled | 2.390 | 0.045 | 0.022 | 9.12 | 0.0377 | scaled_variant | 8.6 |
| 13 | semnasnet_075_scaled | 1.562 | 0.064 | 0.032 | 5.96 | 0.0356 | scaled_variant | 8.6 |
| 14 | spnasnet_075_scaled | 3.316 | 0.247 | 0.124 | 12.65 | 0.0365 | scaled_variant | 8.6 |
| 15 | regnety_075_scaled | 2.372 | 0.299 | 0.150 | 9.29 | 0.0602 | scaled_variant | 8.6 |
| 16 | rexnet_125_scaled | 5.950 | 1.000 | 0.500 | 23.40 | 0.0595 | scaled_variant | 8.6 |
| 17 | rexnet_075_scaled | 3.570 | 0.600 | 0.300 | 14.04 | 0.0485 | scaled_variant | 8.6 |
| 18 | semnasnet_150_scaled | 3.123 | 0.128 | 0.065 | 11.91 | 0.0470 | scaled_variant | 8.6 |
| 19 | spnasnet_125_scaled | 5.527 | 0.413 | 0.206 | 21.08 | 0.0448 | scaled_variant | 8.6 |
| 20 | spnasnet_150_scaled | 6.632 | 0.495 | 0.247 | 25.30 | 0.0482 | scaled_variant | 8.6 |
| 21 | rexnet_150_scaled | 7.140 | 1.200 | 0.600 | 28.08 | 0.0640 | scaled_variant | 8.6 |
| 22 | haloregnetz_030_scaled | 3.504 | 0.591 | 0.295 | 13.37 | 0.0635 | scaled_variant | 8.6 |
| 23 | nf_030_scaled | 2.629 | 0.078 | 0.039 | 10.03 | 0.0655 | scaled_variant | 8.6 |
| 24 | regnetx_125_scaled | 3.356 | 0.498 | 0.249 | 13.15 | 0.0709 | scaled_variant | 8.6 |
| 25 | regnety_150_scaled | 4.744 | 0.599 | 0.300 | 18.57 | 0.0795 | scaled_variant | 8.7 |
| 26 | regnetx_200_scaled | 5.370 | 0.796 | 0.398 | 21.04 | 0.0856 | scaled_variant | 8.7 |
| 27 | regnety_125_scaled | 3.954 | 0.499 | 0.250 | 15.48 | 0.0739 | scaled_variant | 8.7 |
| 28 | regnetx_150_scaled | 4.027 | 0.597 | 0.298 | 15.78 | 0.0763 | scaled_variant | 8.7 |
| 29 | regnety_200_scaled | 6.326 | 0.798 | 0.400 | 24.76 | 0.0891 | scaled_variant | 8.7 |
| 30 | nf_075_scaled | 6.573 | 0.195 | 0.098 | 25.07 | 0.0945 | scaled_variant | 8.7 |
| 31 | nf_200_scaled | 17.529 | 0.520 | 0.260 | 66.87 | 0.1398 | scaled_variant | 8.7 |
| 32 | nf_125_scaled | 10.955 | 0.325 | 0.163 | 41.79 | 0.1159 | scaled_variant | 8.8 |
| 33 | nf_150_scaled | 13.146 | 0.390 | 0.195 | 50.15 | 0.1246 | scaled_variant | 8.8 |
| 34 | skresnet18_030_scaled | 3.587 | 1.092 | 0.546 | 13.68 | 0.0801 | scaled_variant | 8.8 |
| 35 | regnetv_125_scaled | 25.801 | 1.087 | 0.544 | 98.42 | 0.1131 | scaled_variant | 9.2 |
| 36 | resnetblur18_030_scaled | 3.507 | 1.092 | 0.546 | 13.38 | 0.0879 | scaled_variant | 9.2 |
| 37 | resnet14t_030_scaled | 3.024 | 1.092 | 0.546 | 11.54 | 0.0771 | scaled_variant | 9.5 |
| 38 | tf_125_scaled | 2.552 | 0.050 | 0.025 | 9.74 | 0.0373 | scaled_variant | 9.7 |
| 39 | regnetz_030_scaled | 2.137 | 0.171 | 0.085 | 8.15 | 0.0444 | scaled_variant | 9.7 |
| 40 | tf_150_scaled | 3.063 | 0.060 | 0.030 | 11.68 | 0.0401 | scaled_variant | 9.7 |
| 41 | tf_200_scaled | 4.084 | 0.080 | 0.040 | 15.58 | 0.0450 | scaled_variant | 9.7 |
| 42 | regnetv_030_scaled | 6.192 | 0.261 | 0.131 | 23.62 | 0.0639 | scaled_variant | 9.8 |
| 43 | regnetz_075_scaled | 5.342 | 0.427 | 0.214 | 20.38 | 0.0641 | scaled_variant | 9.8 |
| 44 | regnetv_075_scaled | 15.480 | 0.652 | 0.326 | 59.05 | 0.0922 | scaled_variant | 9.8 |
| 45 | regnetz_125_scaled | 8.903 | 0.712 | 0.356 | 33.96 | 0.0786 | scaled_variant | 9.8 |
| 46 | regnetz_150_scaled | 10.684 | 0.855 | 0.427 | 40.76 | 0.0846 | scaled_variant | 9.8 |
| 47 | gc_030_scaled | 4.103 | 0.960 | 0.480 | 15.65 | 0.0789 | scaled_variant | 9.8 |
| 48 | regnetz_200_scaled | 14.245 | 1.140 | 0.570 | 54.34 | 0.0949 | scaled_variant | 9.9 |
| 49 | crossvit_030_scaled | 2.104 | 0.519 | 0.260 | 8.03 | 0.1412 | scaled_variant | 10.0 |
| 50 | legacy_030_scaled | 3.534 | 1.092 | 0.546 | 13.48 | 0.0647 | scaled_variant | 10.1 |
| 51 | seresnet18_030_scaled | 3.534 | 1.092 | 0.546 | 13.48 | 0.0644 | scaled_variant | 10.1 |
| 52 | deit_030_scaled | 0.234 | 0.069 | 0.035 | 25.58 | 0.4047 | scaled_variant | 10.1 |
| 53 | crossvit_075_scaled | 5.261 | 1.297 | 0.649 | 20.07 | 0.2037 | scaled_variant | 10.2 |
| 54 | resnet18d_030_scaled | 3.513 | 1.092 | 0.546 | 13.40 | 0.0673 | scaled_variant | 10.2 |
| 55 | mobilenetv2_030_scaled | 0.094 | 1.051 | 0.090 | 4.11 | 0.0480 | scaled_variant | 10.5 |
| 56 | squeezenet1_0_new | 1.250 | 0.830 | 0.415 | 5.00 | 0.0180 | popular_outside_benchmark | 10.7 |
| 57 | deit_075_scaled | 0.586 | 0.173 | 0.087 | 63.96 | 0.5839 | scaled_variant | 13.3 |
| 58 | deit_125_scaled | 0.976 | 0.289 | 0.145 | 106.60 | 0.7163 | scaled_variant | 13.6 |
| 59 | deit_150_scaled | 1.171 | 0.347 | 0.174 | 127.92 | 0.7705 | scaled_variant | 13.6 |
| 60 | deit_200_scaled | 1.562 | 0.462 | 0.232 | 170.56 | 0.8645 | scaled_variant | 13.7 |
| 61 | vit_030_scaled | 0.117 | 0.035 | 0.017 | 6.75 | 1.7769 | scaled_variant | 14.0 |
| 62 | lcnet_030_scaled | 0.493 | 0.049 | 0.025 | 1.90 | 2.2228 | scaled_variant | 14.0 |
| 63 | vit_075_scaled | 0.293 | 0.087 | 0.044 | 16.88 | 2.5636 | scaled_variant | 14.0 |
| 64 | tinynet_030_scaled | 0.613 | 0.123 | 0.061 | 2.38 | 3.3396 | scaled_variant | 14.1 |
| 65 | lcnet_075_scaled | 1.233 | 0.123 | 0.062 | 4.75 | 3.2068 | scaled_variant | 14.1 |
| 66 | vit_125_scaled | 0.489 | 0.145 | 0.073 | 28.14 | 3.1447 | scaled_variant | 14.1 |
| 67 | vit_150_scaled | 0.587 | 0.174 | 0.087 | 33.77 | 3.3826 | scaled_variant | 14.1 |
| 68 | vit_200_scaled | 0.782 | 0.232 | 0.116 | 45.02 | 3.7952 | scaled_variant | 14.1 |
| 69 | tinynet_075_scaled | 1.532 | 0.307 | 0.153 | 5.94 | 4.8180 | scaled_variant | 14.1 |
| 70 | mixnet_030_scaled | 1.240 | 0.310 | 0.155 | 4.81 | 6.9449 | scaled_variant | 14.2 |
| 71 | lcnet_125_scaled | 2.055 | 0.205 | 0.103 | 7.92 | 3.9338 | scaled_variant | 14.2 |
| 72 | ghostnet_030_scaled | 0.776 | 0.012 | 0.006 | 3.02 | 6.6595 | scaled_variant | 14.2 |
| 73 | lcnet_150_scaled | 2.466 | 0.247 | 0.123 | 9.51 | 4.2314 | scaled_variant | 14.3 |
| 74 | lcnet_200_scaled | 3.288 | 0.329 | 0.164 | 12.68 | 4.7474 | scaled_variant | 14.3 |
| 75 | visformer_030_scaled | 3.096 | 0.619 | 0.310 | 11.85 | 4.8295 | scaled_variant | 14.3 |
| 76 | tinynet_125_scaled | 2.554 | 0.511 | 0.255 | 9.90 | 5.9102 | scaled_variant | 14.4 |
| 77 | ghostnet_075_scaled | 1.940 | 0.030 | 0.015 | 7.56 | 9.6077 | scaled_variant | 14.4 |
| 78 | ghostnet_125_scaled | 3.234 | 0.050 | 0.025 | 12.60 | 11.7858 | scaled_variant | 14.6 |
| 79 | ghostnet_150_scaled | 3.881 | 0.060 | 0.030 | 15.12 | 12.6774 | scaled_variant | 14.6 |
| 80 | tinynet_150_scaled | 3.065 | 0.613 | 0.306 | 11.88 | 6.3574 | scaled_variant | 14.6 |
| 81 | ghostnet_200_scaled | 5.174 | 0.080 | 0.040 | 20.16 | 14.2235 | scaled_variant | 14.8 |
| 82 | resnet10t_030_scaled | 1.630 | 0.571 | 0.285 | 6.23 | 2.3796 | scaled_variant | 14.8 |
| 83 | fbnetv3_030_scaled | 2.579 | 0.645 | 0.322 | 9.95 | 7.9470 | scaled_variant | 15.1 |
| 84 | efficientformer_030_scaled | 3.687 | 0.737 | 0.369 | 14.14 | 6.1593 | scaled_variant | 15.1 |
| 85 | mixnet_075_scaled | 3.101 | 0.776 | 0.388 | 12.02 | 10.0194 | scaled_variant | 15.2 |
| 86 | tinynet_200_scaled | 4.086 | 0.818 | 0.408 | 15.84 | 7.1327 | scaled_variant | 15.3 |
| 87 | mobilevit_030_scaled | 0.382 | 0.153 | 0.076 | 1.49 | 6.2761 | scaled_variant | 15.3 |
| 88 | hrnet_030_scaled | 3.956 | 0.989 | 0.494 | 15.17 | 16.7842 | scaled_variant | 15.4 |
| 89 | mobilevit_075_scaled | 0.954 | 0.382 | 0.191 | 3.74 | 9.0545 | scaled_variant | 15.4 |
| 90 | res2next50_030_scaled | 7.401 | 1.332 | 0.666 | 28.36 | 47.2556 | scaled_variant | 15.4 |
| 91 | mixnet_125_scaled | 5.169 | 1.292 | 0.646 | 20.02 | 12.2909 | scaled_variant | 15.4 |
| 92 | res2net50_030_scaled | 7.586 | 1.214 | 0.607 | 29.03 | 41.1842 | scaled_variant | 15.5 |
| 93 | xcit_030_scaled | 0.915 | 0.366 | 0.183 | 3.53 | 14.7540 | scaled_variant | 15.6 |
| 94 | mobilevit_200_scaled | 2.544 | 1.018 | 0.508 | 9.96 | 13.4045 | scaled_variant | 16.3 |
| 95 | mobilevit_125_scaled | 1.590 | 0.636 | 0.318 | 6.23 | 11.1072 | scaled_variant | 16.4 |
| 96 | xcit_075_scaled | 2.287 | 0.915 | 0.458 | 8.83 | 21.2856 | scaled_variant | 16.4 |
| 97 | pit_030_scaled | 1.454 | 0.582 | 0.291 | 5.57 | 13.4885 | scaled_variant | 16.4 |
| 98 | mobilevit_150_scaled | 1.908 | 0.764 | 0.381 | 7.47 | 11.9475 | scaled_variant | 16.5 |
| 99 | coat_030_scaled | 1.650 | 0.330 | 0.165 | 6.34 | 140.1820 | scaled_variant | 16.8 |
| 100 | coat_075_scaled | 4.124 | 0.825 | 0.413 | 15.85 | 202.2405 | scaled_variant | 17.2 |

## Summary Statistics

- Min predicted energy: 8.5 mWh
- Max predicted energy: 17.2 mWh
- Median: 10.1 mWh
- Mean: 11.6 mWh

## Top 10 Most Efficient (Lowest Predicted Energy)

1. **edgenext_xx_small_new**: 8.5 mWh (1.33M params, 0.266 GFLOPs)
2. **regnetv_150_scaled**: 8.5 mWh (30.96M params, 1.305 GFLOPs)
3. **shufflenet_v2_x0_5_new**: 8.5 mWh (1.37M params, 0.041 GFLOPs)
4. **regnety_030_scaled**: 8.6 mWh (0.95M params, 0.120 GFLOPs)
5. **mobilenetv3_200_scaled**: 8.6 mWh (3.19M params, 0.060 GFLOPs)
6. **regnetx_030_scaled**: 8.6 mWh (0.81M params, 0.119 GFLOPs)
7. **spnasnet_200_scaled**: 8.6 mWh (8.84M params, 0.660 GFLOPs)
8. **semnasnet_200_scaled**: 8.6 mWh (4.16M params, 0.170 GFLOPs)
9. **regnetx_075_scaled**: 8.6 mWh (2.01M params, 0.298 GFLOPs)
10. **semnasnet_125_scaled**: 8.6 mWh (2.60M params, 0.106 GFLOPs)

## Top 10 Most Demanding (Highest Predicted Energy)

1. **coat_075_scaled**: 17.2 mWh (4.12M params, 0.825 GFLOPs)
2. **coat_030_scaled**: 16.8 mWh (1.65M params, 0.330 GFLOPs)
3. **mobilevit_150_scaled**: 16.5 mWh (1.91M params, 0.764 GFLOPs)
4. **pit_030_scaled**: 16.4 mWh (1.45M params, 0.582 GFLOPs)
5. **xcit_075_scaled**: 16.4 mWh (2.29M params, 0.915 GFLOPs)
6. **mobilevit_125_scaled**: 16.4 mWh (1.59M params, 0.636 GFLOPs)
7. **mobilevit_200_scaled**: 16.3 mWh (2.54M params, 1.018 GFLOPs)
8. **xcit_030_scaled**: 15.6 mWh (0.91M params, 0.366 GFLOPs)
9. **res2net50_030_scaled**: 15.5 mWh (7.59M params, 1.214 GFLOPs)
10. **mixnet_125_scaled**: 15.4 mWh (5.17M params, 1.292 GFLOPs)