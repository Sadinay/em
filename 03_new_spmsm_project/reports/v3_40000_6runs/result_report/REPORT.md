# SPMSM V3 six-model test report

All six runs completed successfully using the same audited split: 40,000 training, 6,483 validation and 6,483 held-out test genes. Results below are from seed `20260903`. The test split was accessed only after validation selected the best checkpoint.

## Ranking

| Rank | Model | Best epoch | Test std. MSE | Tavg MAE | Tavg R2 | DeltaT MAE | DeltaT R2 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Polar 90 VGG16 | 39 | 0.006492 | 0.006515 | 0.998857 | 0.011926 | 0.988315 |
| 2 | XY 90 VGG16 | 39 | 0.007254 | 0.008474 | 0.998480 | 0.013302 | 0.987182 |
| 3 | XY 360 VGG16 | 40 | 0.008647 | 0.008710 | 0.998424 | 0.015003 | 0.984489 |
| 4 | Logical Small CNN | 40 | 0.019567 | 0.009256 | 0.998356 | 0.027190 | 0.963012 |
| 5 | Logical Mini-Inception | 34 | 0.028121 | 0.019685 | 0.986071 | 0.029309 | 0.958208 |
| 6 | Polar 360 VGG16 | 37 | 0.029834 | 0.015770 | 0.996408 | 0.033712 | 0.944669 |

Units for both MAE columns are N m.

## Main findings

1. **Polar 90 VGG16 is the best overall model.** It has the lowest test standardized MSE (`0.006492`), Tavg MAE (`0.006515` N m), and DeltaT MAE (`0.011926` N m).
2. Against XY 90 VGG16, Polar 90 reduces Tavg MAE by `23.1%` and DeltaT MAE by `10.4%`.
3. Against the best logical model, Polar 90 reduces Tavg MAE by `29.6%` and DeltaT MAE by `56.1%`.
4. **The 360-degree representation did not improve accuracy at 224x224.** XY 360 is slightly worse than XY 90 (Tavg MAE `+2.8%`, DeltaT MAE `+12.8%`). Polar 360 is substantially worse than Polar 90 (Tavg MAE `+142.1%`, DeltaT MAE `+182.7%`).
5. The likely explanation is effective resolution: at fixed 224 angular columns, Polar 90 has about `0.402` degree/column, whereas Polar 360 has about `1.607` degree/column. The 360-degree representation spends resolution on repeated motor sectors rather than giving the 90-degree design region more detail.
6. XY 360 reached its best validation score at epoch 40, the configured maximum. Therefore the present comparison proves it did not outperform under the common 40-epoch budget; it does not prove that further training could never improve it.

## Audit notes

- Total recorded training time: `11.46` hours.
- Only one seed was retained by design, so these results compare the selected runs but do not quantify random-seed variance.
- Test labels come from FEMM and are identical across all six models.
- Both targets are predicted simultaneously by every model.

## Figures and data

- `model_comparison.png`: accuracy and time across all six models.
- `validation_learning_curves.png`: validation standardized MSE by epoch.
- `best_model_parity.png`: FEMM actual versus Polar 90 predictions.
- `metrics_summary.csv`: machine-readable full metric table.
