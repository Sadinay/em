# Six-model V2 fair short-test report

All V2 models use the same frozen 11,700/1,500/1,500 indices, the same training-only target scaler, standardized two-target MSE, effective batch 64, and three seeds. The legacy column is a single historical seed, so its apparent difference has no uncertainty estimate.

## Architecture and result comparison

| Model | Main V2 change | Parameters old -> V2 | Tavg MAE old -> V2 mean +/- std | DeltaT MAE old -> V2 mean +/- std |
|---|---|---:|---:|---:|
| 10 Mini-Inception V2 | Inception blocks gain projection residuals; two independent 128-unit heads; one pool retained. | 911,346 -> 932,146 | 0.0693 -> 0.0588 +/- 0.0100 | 0.0748 -> 0.0672 +/- 0.0014 |
| 10 ResNet20 V2 | Stage 3 uses stride 1, preserving 5x5; two independent 128-unit heads. | 345,938 -> 681,938 | 0.0779 -> 0.0927 +/- 0.0063 | 0.0741 -> 0.0884 +/- 0.0014 |
| 10 SmallCNN V2 | Simple trunk retained; DeltaT gains a dedicated 3x3 Conv128 branch; independent heads. | 503,458 -> 1,060,898 | 0.0645 -> 0.0575 +/- 0.0090 | 0.0776 -> 0.0689 +/- 0.0117 |
| 224 Mini-Inception V2 | BatchNorm replaced by GroupNorm; residual Inception blocks; 4x4 spatial output and independent heads. | 990,706 -> 1,036,466 | 0.1301 -> 0.0514 +/- 0.0020 | 0.1031 -> 0.0633 +/- 0.0027 |
| 224 ResNet18 V2 | Width reduced to 32/64/128/256; GroupNorm; 4x4 spatial output and independent heads. | 11,303,554 -> 3,845,570 | 0.2843 -> 0.0503 +/- 0.0041 | 0.1242 -> 0.0660 +/- 0.0053 |
| 224 VGG16 V2 | All 13 conv layers retained; GroupNorm; 4x4 spatial output; two independent 256-unit heads. | 18,917,634 -> 18,917,122 | 0.1546 -> 0.0546 +/- 0.0058 | 0.1701 -> 0.0605 +/- 0.0030 |

## Validation and runtime

| Model | Best epoch mean +/- std | Time/run (s) | Torch peak MiB | GPU utilization mean | GPU temperature max (C) |
|---|---:|---:|---:|---:|---:|
| 10 Mini-Inception V2 | 22.0 +/- 5.0 | 86.3 +/- 12.5 | 98 | 27.8% | 52.3 |
| 10 ResNet20 V2 | 26.3 +/- 3.2 | 72.9 +/- 0.5 | 82 | 23.3% | 55.0 |
| 10 SmallCNN V2 | 23.7 +/- 1.5 | 27.3 +/- 0.3 | 86 | 22.8% | 56.0 |
| 224 Mini-Inception V2 | 27.3 +/- 2.9 | 2033.1 +/- 31.7 | 6093 | 93.9% | 60.0 |
| 224 ResNet18 V2 | 28.7 +/- 1.5 | 1791.2 +/- 45.6 | 3613 | 94.5% | 61.0 |
| 224 VGG16 V2 | 27.0 +/- 1.0 | 1948.8 +/- 83.6 | 2561 | 92.6% | 61.7 |

## Verification

- Unit/shape tests passed before training.
- Fixed 256-sample memorization diagnostic: status `passed`, all six passed = `True`.
- Test evaluation occurred once per run, only after loading the best validation checkpoint.
- Test predictions, full histories, per-run configs, checkpoints, band counts and band errors are retained.

## Interpretation boundary

A gain can come from several controlled V2 changes together: architecture, per-model learning rate, normalization, longer validation-controlled training, and equalized effective batch. This experiment does not identify a single causal change. Models whose best epoch is at the maximum may still be under-converged. Mean-regression is assessed using calibration slope, bias, residual plots and sparse-band errors rather than overall MAE alone.

## Artifacts

- `aggregate_metrics.json`: all mean/std metrics and band results.
- `overall_metrics_mean_std.csv`: complete target metrics.
- `band_metrics_mean_std.csv`: band count, MAE and bias.
- `runtime_gpu_mean_std.csv`: time, memory, GPU utilization, power and temperature.
- `v2_layer_output_shapes.csv`: actual leaf-layer call order, output size and direct parameter count.
- `training_validation_curves_six_v2.png`: all seed histories.
- `legacy_vs_v2_test_mae.png`: old single-seed versus V2 three-seed comparison.
- `*_prediction_six_v2.png` and `*_residual_six_v2.png`: test behavior using the three-seed mean prediction.
