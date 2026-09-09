# V3 40,000-sample / simplified 6-run training

The simplified matrix contains six input/architecture configurations and one seed (`20260903`), for six runs. Logical 6x20 retains Small CNN V2 and Mini-Inception V2. Each of the four 224 views retains only VGG16 V2.

## Manual start / resume

Run from any PowerShell directory:

```powershell
python "C:\Users\26096\Desktop\em\03_new_spmsm_project\cnn_zone\scripts\run_v3_40000_6runs.py" --resume
```

`--resume` is safe for both first-time and resumed execution. The earlier two-epoch Mini-Inception timing probe belongs to the superseded 20-run plan and is intentionally not part of this six-run matrix.

To validate the plan without training:

```powershell
python "C:\Users\26096\Desktop\em\03_new_spmsm_project\cnn_zone\scripts\run_v3_40000_6runs.py" --dry-run
```

To repeat one-batch GPU forward/backward probes for all six configurations:

```powershell
python "C:\Users\26096\Desktop\em\03_new_spmsm_project\cnn_zone\scripts\run_v3_40000_6runs.py" --probe-only
```

To train only selected configurations or seeds, use `--models` and `--seeds`, for example:

```powershell
python "C:\Users\26096\Desktop\em\03_new_spmsm_project\cnn_zone\scripts\run_v3_40000_6runs.py" --resume --models logical6x20/small_cnn_v2 xy90_224/vgg16_v2 --seeds 20260903
```

To request a safe wall-time pause after a completed epoch:

```powershell
python "C:\Users\26096\Desktop\em\03_new_spmsm_project\cnn_zone\scripts\run_v3_40000_6runs.py" --resume --time-limit-minutes 60
```

## Epoch policy

- Logical 6x20: maximum 100 epochs, early stopping after a minimum of 20 epochs and 12 non-improving validation epochs.
- All 224 views: minimum 30, maximum 40 epochs; early stopping patience 8.
- The best checkpoint is selected only by validation standardized MSE.
- The held-out test set is evaluated once after a run finishes.
- `polar360_224` uses circular padding only along its angular axis.

## Earlier timing probe

The superseded 20-run matrix tested `polar360_224/mini_inception_v2` on all 40,000 training samples for two complete epochs. Those checkpoints remain archived but are not consumed by this simplified plan.

- Epoch 1: validation standardized MSE `0.296998`; validation R2 `(Tavg=0.97275, DeltaT=0.44701)`.
- Epoch 2: validation standardized MSE `0.127042`; validation R2 `(Tavg=0.98085, DeltaT=0.77066)`.
- Each epoch performed exactly 625 effective optimizer steps.
- Resume loading was verified and identifies epoch 3 as the next epoch.

## Outputs

- Models/checkpoints: `cnn_zone/models/v3_40000_6runs/`
- Plan and run status: `reports/V3/03_完整审核资料/训练过程记录/`
- GPU batch probe: `reports/V3/03_完整审核资料/训练过程记录/batch_probe.json`
