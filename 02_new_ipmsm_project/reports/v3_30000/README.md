# V3 30,000-sample training report

This report uses two seeds for each selected model. Early stopping uses 5,000 validation samples. The 1,500 core test preserves V2 comparability; the complete Scheme-A test contains 14,655 samples.

| Model | Best epoch | Full-test Tavg MAE (N m) | Full-test DeltaT MAE (N m) |
|---|---:|---:|---:|
| 10x10 Mini-Inception V3 | 39.0 +/- 7.1 | 0.0483 +/- 0.0063 | 0.0451 +/- 0.0024 |
| 10x10 ResNet20 V3 | 43.0 +/- 1.4 | 0.0581 +/- 0.0020 | 0.0558 +/- 0.0011 |
| 10x10 SmallCNN V3 | 52.0 +/- 7.1 | 0.0418 +/- 0.0005 | 0.0425 +/- 0.0011 |
| 224x224 VGG16 V3 | 35.0 +/- 0.0 | 0.0401 +/- 0.0037 | 0.0417 +/- 0.0015 |

Artifacts:

- `aggregate_metrics.json` and `v3_metrics_mean_std.csv`;
- `training_validation_curves_four_v3.png`;
- `tavg_prediction_four_v3.png` and `tavg_residual_four_v3.png`;
- `delta_t_prediction_four_v3.png` and `delta_t_residual_four_v3.png`;
- `v2_vs_v3_core_test_mae.png`: V2/V3 comparison on the shared frozen 1,500-sample core test.
- `femm_error_vs_model_disagreement_v3.png`: FEMM error versus four-architecture disagreement on the complete test population, with the reviewed 1,000 genes highlighted.
