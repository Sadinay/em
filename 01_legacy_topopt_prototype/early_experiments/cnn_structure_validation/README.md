# 材料拓扑图 Conv2D → J

这是修正后的结构 CNN。输入不是六个统计量，也不依赖 MAT 中的 `best_bits`；材料图直接从 FEM 的节点、线段、圆弧和 block label 重建。

## 输入

- 结构图：`[B, 5, 96, 96]`
  - 通道0：模型外背景
  - 通道1：Air
  - 通道2：Iron/Steel
  - 通道3：Copper
  - 通道4：Magnet/N40/NdFeB
- 数值条件：`T_min [B,1]`
- 输出：`J = bestOverall`

所有配对 FEM 的坐标范围均为 `[0,0]–[74,74]`。解析器重建封闭材料区域后，在统一96×96像素中心采样，并转成五通道 one-hot。

## 网络

```text
FEM material image [B,5,96,96]
→ Conv2D(5→16, kernel=3×3, padding=1)
→ ReLU
→ MaxPool2D(2×2, stride=2)
→ AdaptiveAvgPool2D(4×4)
→ Flatten [B,256]
                  ┐
T_min [B,1] ──────┴→ Dense(384) → ReLU → Dense(1)
```

## 数据

42条运行记录经过 FEM 栅格质量检查后，按完整材料栅格和 `T_min` 去重为21个可靠独立结构。使用11个训练、10个验证；验证集不再包含重复结构。多材料标签冲突区域超过10个的旧格式 FEM 会被排除并记录。

## 运行

```powershell
python build_dataset.py --raw-root "..\FP" --audit-output "..\dataset_audit_output" --output "..\cnn_structure_output"
python train.py --dataset "..\cnn_structure_output\structure_dataset.npz" --output "..\cnn_structure_output\overfit" --mode overfit
python train.py --dataset "..\cnn_structure_output\structure_dataset.npz" --output "..\cnn_structure_output\full" --mode full
```

样本极少，结果只用于验证结构图卷积流程。

## 直接从新 FEM 推理

```powershell
python predict_fem.py `
  --fem "C:\path\to\new_model.fem" `
  --t-min 0.8 `
  --checkpoint "..\cnn_structure_output\full\model.pt"
```

此命令不需要 `best_bits`；FEM 提供结构，用户提供 `T_min`。
