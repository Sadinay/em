# SPMSM V3 输入与模型准备审计

状态：输入、数据拆分和模型定义已准备完成，尚未开始训练。

## 数据拆分

- 训练集：40,000。
- 验证集：6,483，与原审计拆分完全相同。
- 测试集：6,483，与原审计拆分完全相同。
- 原训练候选中的未使用样本：11,838，不进入验证集或测试集。
- 分层变量：只使用平均转矩 `Tavg`。
- 两端稀少区间 `<2.0` 和 `>=3.75 N m` 的原训练样本全部保留；中间五个区间以固定随机种子按约 77% 抽取。
- 三个有效集合之间索引交集均为 0。

## 三类输入

1. `logical6x20`：`[2,6,20]`，二通道分别表示 Air 和 N38 永磁体。基因顺序为 `gene = angular_index * 6 + radial_index`。
2. `xy224`：`[8,224,224]`，在真实 x-y 坐标中采样完整 0-90° FEM 电机象限。
3. `polar90_224`：`[8,224,224]`，把同一完整象限按半径 `0-64 mm` 与物理角度 `0-90°` 采样。

两套 224 输入都包含轴心、转子铁心、基因设计带、气隙、定子、绕组和外部空气。它们使用同一个求解后的 FEM 三角网格和同一套材料语义，只改变坐标表达。极坐标输入不是将 xy 图片二次插值或拉伸得到的。

224 输入的八个通道为：设计区空气、径向向内 PM、径向向外 PM、固定空气、固定铁、绕组、其他固定材料、几何掩膜。

查找表检查结果：

- 120 个逻辑基因均被像素覆盖。
- 480 个 FEM 物理设计区域均与 120 基因一一核对。
- `xy224` 设计区每个基因最少 6 个像素，最多 16 个像素。
- `polar90_224` 设计区每个基因最少 8 个像素，最多 24 个像素。

## 模型矩阵

共九个独立实验：

- 三个 `logical6x20`：Mini-Inception V2、ResNet20 V2、Small CNN V2。
- 三个 `xy224`：Mini-Inception V2、ResNet18 V2、VGG16 V2。
- 三个 `polar90_224`：Mini-Inception V2、ResNet18 V2、VGG16 V2。

逻辑模型沿用 V2 的骨干思想，但已把输入从三通道 10x10 改为二通道 6x20，并把最终特征尺寸适配为 3x10。因此这些模型不能直接载入 V2 权重。两套 224 模型的骨干一致，以保证坐标表达对比公平。

九个模型都输出两个数：平均转矩 `Tavg` 与转矩波动 `DeltaT`，并使用相互独立的回归头。九个模型均已完成一次前向计算，输出形状均为 `[batch,2]`。

## 关键文件

- `cnn_zone/configs/v3_input_and_model_matrix.json`：完整实验配置。
- `cnn_zone/outputs/splits/scheme_a_tavg_bands_train40000_val6483_test6483.npz`：新拆分。
- `cnn_zone/outputs/lookups/spmsm_xy224_full_motor.npz`：笛卡尔查找表。
- `cnn_zone/outputs/lookups/spmsm_polar90_224_full_motor.npz`：完整 90° 极坐标查找表。
- `reports/spmsm_inputs/input_representation_preview.png`：三类输入预览。
- `reports/spmsm_inputs/model_matrix_audit.json`：模型参数量与前向检查。
