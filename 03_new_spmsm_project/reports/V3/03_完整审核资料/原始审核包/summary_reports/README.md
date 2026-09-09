# V3 40,000-sample training report

This report follows the project-02 V3 report layout. All six models use one seed and the same frozen 6,483-sample test set.

| Model | Best epoch | Test Tavg MAE (N m) | Test DeltaT MAE (N m) |
|---|---:|---:|---:|
| 6x20 SmallCNN V3 | 40 | 0.0093 | 0.0272 |
| 6x20 Mini-Inception V3 | 34 | 0.0197 | 0.0293 |
| XY90 224x224 VGG16 V3 | 39 | 0.0085 | 0.0133 |
| Polar90 224x224 VGG16 V3 | 39 | 0.0065 | 0.0119 |
| XY360 224x224 VGG16 V3 | 40 | 0.0087 | 0.0150 |
| Polar360 224x224 VGG16 V3 | 37 | 0.0158 | 0.0337 |

Best overall model: **Polar90 224x224 VGG16 V3**.

Artifacts:

- `aggregate_metrics.json` and `v3_metrics_mean_std.csv`;
- `training_validation_curves_six_v3.png`;
- `tavg_prediction_six_v3.png` and `tavg_residual_six_v3.png`;
- `delta_t_prediction_six_v3.png` and `delta_t_residual_six_v3.png`;
- `input_representation_test_mae.png`;
- `femm_error_vs_model_disagreement_v3.png` and the associated 1,000-gene JSON.
- `split_novelty_audit.json`: exact-overlap, Hamming-distance and nearest-neighbour baseline audit.

There is no V2-versus-V3 plot because project 03 is a new motor/gene definition and has no same-split project-03 V2 baseline. A cross-motor comparison would not be scientifically valid.
