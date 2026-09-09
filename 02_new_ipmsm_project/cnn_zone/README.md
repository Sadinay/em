# CNN训练区：架构与合规说明

本目录用于新IPMSM项目的多目标回归。正式目标固定为：

```text
output[:,0] = Tavg，单位N·m
output[:,1] = DeltaT，单位N·m
```

`Fitvalue`不作为输入或训练目标。两个目标只使用训练集统计量分别标准化，并在评价时反标准化回物理单位。

当前状态：数据清洗、固定划分、10×10输入、224×224实时语义渲染、六个v2模型结构测试和256样本过拟合均已通过；固定11,700/1,500/1,500子集上的六模型、三随机种子v2公平短测已经全部完成（18/18），最终图表和三seed汇总报告已经生成。尚未进行146,471样本正式全量训练。下面必须区分“结构已实现”“短测已完成”和“正式全量模型”三种状态。

## 统一输入与训练约束

### 10×10逻辑拓扑

```text
原始基因100格
→ [radius=10, angle=10]
→ Air/PM/Iron三通道one-hot
→ [B,3,10,10]
```

不把材料编号当连续灰度，不resize、不插值、不旋转、不翻转。

### 224×224真实FEM语义图

图像由真实FEMM三角网格查找表在batch中实时生成，不写入海量PNG：

```text
[B,100]
→ pixel_gene_id查找
→ [B,8,224,224]
```

8个通道依次为：

```text
design_air
design_pm_inward
design_pm_outward
design_iron
fixed_air
fixed_iron
winding
geometry_mask
```

可选第9个`boundary`通道，只作为消融实验，默认关闭。

### 统一输出头

所有CNN最终都使用共享特征和两个独立线性回归头：

```text
shared feature
├── head_Tavg   → 1
└── head_DeltaT → 1
```

v1短测和本轮v2公平短测的实际损失均为两个标准化目标的平均MSE。README早期版本写成SmoothL1与代码不一致，现已纠正。正式模型必须使用相同的训练/验证/测试索引。

---

# 一、三种10×10 CNN架构

## 1. 10×10低池化Mini-Inception（主逻辑基线，已实现）

代码类：`Logical10MiniInception`。

```text
Input                              [B, 3, 10, 10]
3×3 Conv, 32 + BN + ReLU          [B,32,10,10]
InceptionBlock(32, branch=16)      [B,64,10,10]
InceptionBlock(64, branch=24)      [B,96,10,10]
MaxPool 2×2                        [B,96, 5, 5]
InceptionBlock(96, branch=32)      [B,128,5,5]
Flatten                            [B,3200]
Linear 3200→256 + ReLU             [B,256]
├── Linear 256→1: Tavg
└── Linear 256→1: DeltaT
```

每个InceptionBlock有四个并行分支：

```text
1×1
1×1 → 3×3
1×1 → 3×3 → 3×3（等效较大感受野）
1×1 → 1×5 → 5×1（分别提取角向/径向结构）
```

只进行一次池化，避免10×10输入过早丢失小材料区域。参数量约911,346。已通过256样本双目标过拟合。

## 2. 10×10 CIFAR-style ResNet20（计划正式实现）

不使用ImageNet的7×7卷积和初始max-pool，采用适合小矩阵的CIFAR结构：

```text
Input                                  [B,3,10,10]
3×3 Conv,16 + BN + ReLU               [B,16,10,10]
Stage 1: 3×BasicBlock, channels=16     [B,16,10,10]
Stage 2: 3×BasicBlock, channels=32
         第一个block stride=2          [B,32,5,5]
Stage 3: 3×BasicBlock, channels=64
         第一个block stride=2          [B,64,3,3]
Flatten                                [B,576]
Linear 576→128 + ReLU                  [B,128]
├── Linear 128→1: Tavg
└── Linear 128→1: DeltaT
```

BasicBlock为两个3×3卷积；尺寸或通道变化时，shortcut使用1×1 stride=2卷积和BN。默认不使用Dropout。

## 3. 10×10 Small CNN（计划正式实现）

作为容易训练、参数较少的普通卷积基线：

```text
Input                              [B,3,10,10]
3×3 Conv,32 + BN + ReLU           [B,32,10,10]
3×3 Conv,64 + BN + ReLU           [B,64,10,10]
MaxPool 2×2                        [B,64,5,5]
3×3 Conv,128 + BN + ReLU          [B,128,5,5]
Flatten                            [B,3200]
Linear 3200→128 + ReLU             [B,128]
├── Linear 128→1: Tavg
└── Linear 128→1: DeltaT
```

该模型用于判断复杂的Inception/ResNet是否真的优于普通局部卷积。

### 非CNN辅助基线：MLP

MLP不计入上述“三种CNN”，但建议保留用于判断卷积空间先验是否有效：

```text
Flatten 3×10×10=300
→ Linear 300→256 + ReLU
→ Linear 256→128 + ReLU
→ 两个独立输出头
```

---

# 二、224×224真实FEM语义图架构

## 1. 低池化Mini-Inception（主语义模型，已实现）

代码类：`Semantic90MiniInception`。

```text
Input                              [B,8,224,224]
3×3 Conv,32 + BN + ReLU           [B,32,224,224]
InceptionBlock(branch=16)          [B,64,224,224]
MaxPool 2×2                        [B,64,112,112]
InceptionBlock(branch=24)          [B,96,112,112]
MaxPool 2×2                        [B,96,56,56]
InceptionBlock(branch=32)          [B,128,56,56]
MaxPool 2×2                        [B,128,28,28]
InceptionBlock(branch=48)          [B,192,28,28]
AdaptiveAvgPool 4×4                [B,192,4,4]
Flatten                            [B,3072]
Linear 3072→256 + ReLU             [B,256]
├── Linear 256→1: Tavg
└── Linear 256→1: DeltaT
```

第一层stride=1，第一次池化推迟到首个InceptionBlock之后。参数量约990,706。已通过256样本过拟合，但小batch下BatchNorm统计有明显波动；短训练时应与GroupNorm做受控比较。

## 2. ResNet18语义回归（计划正式实现）

使用ResNet18的`[2,2,2,2]` BasicBlock布局，但针对语义材料图修改入口：

```text
Input                              [B,8,224,224]
3×3 Conv,64, stride=1 + BN+ReLU   [B,64,224,224]
不使用7×7 stride=2和初始max-pool
Stage 1: 2×BasicBlock,64          [B,64,224,224]
Stage 2: 2×BasicBlock,128         [B,128,112,112]
Stage 3: 2×BasicBlock,256         [B,256,56,56]
Stage 4: 2×BasicBlock,512         [B,512,28,28]
AdaptiveAvgPool 1×1               [B,512]
Linear 512→256 + ReLU             [B,256]
├── Linear 256→1: Tavg
└── Linear 256→1: DeltaT
```

通道或尺寸变化时使用1×1投影shortcut。正式运行前需要用RTX 5070实测batch size；如果显存或速度不合理，只允许调整batch size和梯度累积，不改变数据划分。

## 3. VGG16语义回归（计划、资源允许时运行）

保留VGG16的13个3×3卷积层和5次2×2池化：

```text
Input                              [B,8,224,224]
Block1: Conv64 ×2 + Pool           [B,64,112,112]
Block2: Conv128×2 + Pool           [B,128,56,56]
Block3: Conv256×3 + Pool           [B,256,28,28]
Block4: Conv512×3 + Pool           [B,512,14,14]
Block5: Conv512×3 + Pool           [B,512,7,7]
AdaptiveAvgPool 4×4                [B,512,4,4]
Flatten                            [B,8192]
Linear 8192→512 + ReLU             [B,512]
├── Linear 512→1: Tavg
└── Linear 512→1: DeltaT
```

输入层改为8通道，不使用ImageNet RGB预训练权重。回归头不沿用原始VGG的4096→4096分类器，以控制参数量和显存；卷积主干仍保持VGG16层数。

---

# 三、模型比较规则

所有模型必须：

- 使用同一个清洗后的146,471样本目录；
- 使用同一个固定split文件；
- 目标scaler只读取训练集；
- 输出Tavg和DeltaT各自的MAE、RMSE、R²、归一化MAE、绝对误差P50/P90/P95；
- 同时报告性能分层测试和时间外推测试；
- 不使用Fitvalue输入；
- 不进行旋转、翻转、颜色增强或双线性材料插值；
- 保存最佳checkpoint、配置、目标scaler、样本索引、训练曲线和预测表。

主要公平对照为：

```text
10×10 Mini-Inception
vs.
224×224 Mini-Inception
```

它们回答“真实FEM几何展开是否优于逻辑10×10矩阵”。Small CNN、ResNet20、ResNet18和VGG16用于判断结果是否只由某一种网络结构造成。

## 当前验证结果

256样本过拟合不是泛化测试，但两条主链路均已通过：

| 模型 | Tavg MAE | Tavg R² | DeltaT MAE | DeltaT R² |
|---|---:|---:|---:|---:|
| 10×10 Mini-Inception | 0.01634 | 0.99871 | 0.00832 | 0.99827 |
| 224×224 Mini-Inception | 0.02078 | 0.99807 | 0.01478 | 0.99556 |

详细结果见`semantic90/reports/overfit_256_report.md`。上述训练集记忆指标不能写成论文测试集性能；当前泛化比较使用固定11,700/1,500/1,500短测，详见本文件后续v2章节和`../reports/v2_test_11700/`。

---

# 四、V2优化模型

## 1. 版本边界

v1与v2并存，不覆盖旧类、旧checkpoint或旧预测：

```text
cnn_zone/semantic90/src/models.py       # v1，保持可加载
cnn_zone/semantic90/src/models_v2.py    # v2，新类

cnn_zone/models/test_11700/             # v1短测模型
cnn_zone/models/v2_test_11700/          # v2三seed短测模型

reports/test/                            # v1短测报告
reports/v2_test_11700/                   # v2短测报告
```

六个v2类为：

```text
Logical10MiniInceptionV2
Logical10ResNet20V2
Logical10SmallCNNV2
Semantic90MiniInceptionV2
Semantic90ResNet18V2
Semantic90VGG16V2
```

## 2. V2共同修改

### 真正独立的双层回归头

v1先使用一个共享全连接层，再接两个单层输出：

```text
shared convolutional feature
→ shared Linear + ReLU
├── Linear → Tavg
└── Linear → DeltaT
```

v2取消共享全连接层，两个目标分别拥有完整的两层MLP：

```text
shared convolutional feature
├── Linear(feature_dim,128) → ReLU → Dropout(0.1) → Linear(128,1): Tavg
└── Linear(feature_dim,128) → ReLU → Dropout(0.1) → Linear(128,1): DeltaT
```

VGG16 V2由于输入特征为8192维，两个独立头使用256个隐藏神经元和Dropout(0.2)。独立头用于减少两个物理目标在最后非线性映射阶段的相互干扰；卷积主干仍然共享。

### 归一化策略

```text
10×10 v2：继续使用BatchNorm，物理batch=64
224×224 v2：全部改用GroupNorm，不依赖小物理batch的批统计量
```

224模型默认8组GroupNorm；VGG的256/512通道层使用16组。代码会选择能够整除通道数的最大合理组数。

### 公平训练约束

- 所有模型使用相同固定11,700/1,500/1,500索引；
- 两个目标只根据11,700训练样本计算均值和标准差；
- 损失为两个标准化目标的平均MSE；
- AdamW，weight decay为`1e-4`；
- ReduceLROnPlateau：factor=0.5、patience=3、min_lr=`1e-6`；
- 所有模型有效batch固定为64；224模型通过梯度累积实现；
- AMP开启；224模型使用channels-last；
- 只根据验证损失保存最佳checkpoint；
- 测试集只在训练完成并载入最佳checkpoint后评价一次；
- 三个固定seed为`20260822/20260823/20260824`，最终报告mean ± std；
- 输入通道数均参数化，当前公平实验仍固定为3通道逻辑矩阵或8通道语义图。

## 3. 10×10 Mini-Inception V2

代码类：`Logical10MiniInceptionV2`。

```text
Input                                      [B,3,10,10]    # 三种材料的one-hot拓扑输入
3×3 Conv32 + BN + ReLU                    [B,32,10,10]   # 提取相邻格子的基础局部特征，并稳定激活分布
ResidualInception(32→64)                  [B,64,10,10]   # 并行观察不同尺度的材料组合，残差连接帮助训练
ResidualInception(64→96)                  [B,96,10,10]   # 组合更复杂的局部形状，同时保持10×10位置关系
MaxPool 2×2                               [B,96,5,5]     # 压缩空间尺寸，保留每个小区域中最明显的特征
ResidualInception(96→128)                 [B,128,5,5]    # 在较大感受野上提取整体拓扑模式
Flatten                                   [B,3200]       # 把二维特征展开成回归头可使用的一维向量
├── Linear 3200→128 → ReLU → Dropout0.1 → Linear→1: Tavg    # 独立学习平均转矩，并用少量Dropout抑制过拟合
└── Linear 3200→128 → ReLU → Dropout0.1 → Linear→1: DeltaT  # 独立学习转矩波动，避免与Tavg输出互相干扰
Output                                    [B,2]          # 合并得到每个样本的Tavg和DeltaT预测
```

每个ResidualInception仍保留v1的四个分支：`1×1`、`1×1→3×3`、`1×1→3×3→3×3`、`1×1→1×5→5×1`。新增shortcut：通道相同用Identity，通道不同时用`1×1 Conv + BN`投影，分支拼接结果与shortcut相加后ReLU。

相对v1的修改：

- 普通InceptionBlock改为投影残差Inception；
- v1的共享`3200→256`层改为两个独立`3200→128→1`头；
- 增加头部Dropout 0.1；
- 仍只池化一次，不进一步丢失10×10小区域。

参数量：`911,346 → 932,146`。

## 4. 10×10 ResNet20 V2

代码类：`Logical10ResNet20V2`。

```text
Input                                      [B,3,10,10]    # 三种材料的one-hot拓扑输入
3×3 Conv16 + BN + ReLU                    [B,16,10,10]   # 把材料通道转换为16组基础局部特征
Stage1: 3×BasicBlock16, stride1           [B,16,10,10]   # 深化局部特征，保持原始10×10空间位置
Stage2: 3×BasicBlock32, first stride2     [B,32,5,5]     # 下采样并增加通道，提取更大范围的拓扑关系
Stage3: 3×BasicBlock64, first stride1     [B,64,5,5]     # 增强高层特征，但不再缩小以免丢失细小结构
Flatten                                   [B,1600]       # 将64张5×5特征图展开成一维向量
├── Linear 1600→128 → ReLU → Dropout0.1 → Linear→1: Tavg    # 独立回归平均转矩
└── Linear 1600→128 → ReLU → Dropout0.1 → Linear→1: DeltaT  # 独立回归转矩波动
Output                                    [B,2]          # 输出两个物理性能预测值
```

相对v1的修改：

- v1 Stage3首块stride=2，空间尺寸`5×5→3×3`；v2改为stride=1，保留`5×5`；
- Stage3通道变化仍使用`1×1 Conv + BN`投影，但stride为1；
- Flatten从576维增加到1600维；
- 共享`576→128`层改为两个独立`1600→128→1`头，并增加Dropout 0.1。

参数量：`345,938 → 681,938`。

## 5. 10×10 SmallCNN V2

代码类：`Logical10SmallCNNV2`。

```text
Input                                      [B,3,10,10]    # 三种材料的one-hot拓扑输入
3×3 Conv32 + BN + ReLU                    [B,32,10,10]   # 提取最基础的材料边界和邻域特征
3×3 Conv64 + BN + ReLU                    [B,64,10,10]   # 将基础特征组合成更丰富的局部形状
MaxPool 2×2                               [B,64,5,5]     # 减少计算量，并扩大后续卷积的观察范围
3×3 Conv128 + BN + ReLU                   [B,128,5,5]    # 提取共享的高层拓扑特征

Tavg路径：
shared feature → Flatten3200                              # 直接展开共享特征，保留整体结构信息
→ Linear 3200→128 → ReLU → Dropout0.1 → Linear→1         # 将整体拓扑映射为平均转矩

DeltaT路径：
shared feature → 3×3 Conv128 + BN + ReLU  [B,128,5,5]    # 专门强化局部突变和材料边界特征
→ Flatten3200                                             # 展开波动专用特征
→ Linear 3200→128 → ReLU → Dropout0.1 → Linear→1         # 将局部敏感特征映射为转矩波动

Output                                    [B,2]          # 合并Tavg和DeltaT两个预测结果
```

相对v1的修改：

- 保留简单卷积主干，不改成Inception或深残差网络；
- Tavg直接使用共享特征；
- DeltaT增加专用3×3 Conv128，用于捕捉局部材料变化；
- v1共享`3200→128`层改为两个独立头；
- 增加Dropout 0.1。

参数量：`503,458 → 1,060,898`。

## 6. 224×224 Mini-Inception V2

代码类：`Semantic90MiniInceptionV2`。

```text
Input                                      [B,8,224,224]      # FEM几何、材料和绕组组成的8通道语义图
3×3 Conv32 + GN + ReLU                    [B,32,224,224]     # 提取材料边界等基础特征；GN适合较小batch
ResidualInceptionGN(32→64)                [B,64,224,224]     # 多尺度分析细小材料区域，保留完整分辨率
MaxPool                                   [B,64,112,112]     # 尺寸减半，降低计算量并扩大感受野
ResidualInceptionGN(64→96)                [B,96,112,112]     # 组合相邻区域，学习中尺度结构关系
MaxPool                                   [B,96,56,56]       # 再次压缩空间尺寸
ResidualInceptionGN(96→128)               [B,128,56,56]      # 提取更复杂的磁路和材料分布模式
MaxPool                                   [B,128,28,28]      # 将高分辨率特征压缩到可处理尺度
ResidualInceptionGN(128→192)              [B,192,28,28]      # 提取较大范围的整体电机拓扑特征
AdaptiveAvgPool 4×4                       [B,192,4,4]        # 统一末端尺寸，同时保留粗略空间位置
Flatten                                   [B,3072]           # 展开二维语义特征供回归头使用
├── Linear 3072→128 → ReLU → Dropout0.1 → Linear→1: Tavg    # 独立预测平均转矩
└── Linear 3072→128 → ReLU → Dropout0.1 → Linear→1: DeltaT  # 独立预测转矩波动
Output                                    [B,2]              # 输出两个物理性能预测值
```

相对v1的修改：

- 所有BatchNorm改为GroupNorm；
- 四个普通InceptionBlock改为带投影shortcut的ResidualInceptionGN；
- 继续保留4×4最终空间布局，不压缩成1×1；
- v1共享`3072→256`层改为两个独立`3072→128→1`头；
- 增加Dropout 0.1。

参数量：`990,706 → 1,036,466`。

## 7. 224×224 ResNet18 V2

代码类：`Semantic90ResNet18V2`。它保留ResNet18的`[2,2,2,2]`深度，但属于宽度减半版本。

```text
Input                                      [B,8,224,224]      # FEM几何、材料和绕组组成的8通道语义图
3×3 Conv32 stride1 + GN + ReLU            [B,32,224,224]     # 建立32组基础特征，不在入口处损失分辨率
Stage1: 2×BasicBlock32                    [B,32,224,224]     # 学习细小边界和局部结构，保持原图尺寸
Stage2: 2×BasicBlock64, first stride2     [B,64,112,112]     # 尺寸减半、通道增加，学习更大邻域关系
Stage3: 2×BasicBlock128, first stride2    [B,128,56,56]      # 继续扩大感受野，形成中高层拓扑特征
Stage4: 2×BasicBlock256, first stride2    [B,256,28,28]      # 提取覆盖较大物理区域的整体结构特征
AdaptiveAvgPool 4×4                       [B,256,4,4]        # 统一输出大小，并保留4×4空间布局
Flatten                                   [B,4096]           # 展开末端特征供两个回归头使用
├── Linear 4096→128 → ReLU → Dropout0.1 → Linear→1: Tavg    # 独立预测平均转矩
└── Linear 4096→128 → ReLU → Dropout0.1 → Linear→1: DeltaT  # 独立预测转矩波动
Output                                    [B,2]              # 输出两个物理性能预测值
```

相对v1的修改：

- 所有BatchNorm及shortcut归一化改为GroupNorm；
- 主干宽度从`64/128/256/512`减为`32/64/128/256`；
- 最终池化从`1×1`改为`4×4`，保留空间布局；
- Flatten从512维改为4096维；
- v1共享`512→256`层改为两个独立`4096→128→1`头；
- 增加Dropout 0.1。

参数量：`11,303,554 → 3,845,570`。参数显著减少，但最终空间信息更多。

## 8. 224×224 VGG16 V2

代码类：`Semantic90VGG16V2`。

```text
Input                                      [B,8,224,224]      # FEM几何、材料和绕组组成的8通道语义图
Block1: Conv64×2 + GN + ReLU + Pool       [B,64,112,112]     # 提取细小边界和纹理，并首次尺寸减半
Block2: Conv128×2 + GN + ReLU + Pool      [B,128,56,56]      # 组合局部边界，形成较大范围的形状特征
Block3: Conv256×3 + GN + ReLU + Pool      [B,256,28,28]      # 加深特征表达，识别复杂材料区域组合
Block4: Conv512×3 + GN + ReLU + Pool      [B,512,14,14]      # 学习更大尺度的磁路和拓扑关系
Block5: Conv512×3 + GN + ReLU + Pool      [B,512,7,7]        # 提炼高层整体结构特征
AdaptiveAvgPool 4×4                       [B,512,4,4]        # 统一末端尺寸，并保留部分空间位置
Flatten                                   [B,8192]           # 展开高维特征供回归头使用
├── Linear 8192→256 → ReLU → Dropout0.2 → Linear→1: Tavg    # 独立预测平均转矩，Dropout抑制大模型过拟合
└── Linear 8192→256 → ReLU → Dropout0.2 → Linear→1: DeltaT  # 独立预测转矩波动，避免两个目标互相干扰
Output                                    [B,2]              # 输出两个物理性能预测值
```

相对v1的修改：

- 保留VGG16身份所需的13个3×3卷积和五个卷积block；
- 所有BatchNorm改为GroupNorm；
- 继续保留4×4最终空间布局；
- v1共享`8192→512`层改为两个独立`8192→256→1`头；
- Dropout设为0.2。

参数量：`18,917,634 → 18,917,122`，总量基本不变，但输出头从共享改为独立。

## 9. V1与V2汇总表

| 模型 | v1主要结构 | v2主要修改 | 参数量v1→v2 |
|---|---|---|---:|
| 10×10 Mini-Inception | 普通Inception、共享FC | 残差Inception、独立双层头 | 911,346→932,146 |
| 10×10 ResNet20 | Stage3 stride2、3×3输出 | Stage3 stride1、5×5输出、独立头 | 345,938→681,938 |
| 10×10 SmallCNN | 完全共享卷积和FC | DeltaT专用卷积、独立头 | 503,458→1,060,898 |
| 224 Mini-Inception | BN、普通Inception | GN、残差Inception、独立头 | 990,706→1,036,466 |
| 224 ResNet18 | BN、64–512宽度、1×1池化 | GN、32–256宽度、4×4池化、独立头 | 11,303,554→3,845,570 |
| 224 VGG16 | BN、共享512维FC | GN、独立256维双头、Dropout0.2 | 18,917,634→18,917,122 |

## 10. V2训练配置

| 模型 | 学习率 | 最大epoch | 早停patience | 物理batch | 有效batch |
|---|---:|---:|---:|---:|---:|
| 10×10 Mini-Inception V2 | 1e-3 | 30 | 6 | 64 | 64 |
| 10×10 ResNet20 V2 | 3e-4 | 30 | 6 | 64 | 64 |
| 10×10 SmallCNN V2 | 1e-3 | 25 | 6 | 64 | 64 |
| 224 Mini-Inception V2 | 3e-4 | 30 | 6 | 32 | 64 |
| 224 ResNet18 V2 | 3e-4 | 30 | 6 | 24 | 64 |
| 224 VGG16 V2 | 1e-4 | 30 | 5 | 16 | 64 |

物理batch由RTX 5070前向/反向显存探测确定。梯度累积按实际样本数精确归一化，最后不足64的有效batch也会正常更新，不丢弃样本。

## 11. V2验证和运行入口

六个v2模型均已完成：

- 输出shape `[B,2]`验证；
- CPU单元测试和GPU forward/backward；
- GroupNorm整除检查；
- 两个输出头参数独立检查；
- 10×10规定空间尺寸检查；
- 224最终4×4空间尺寸检查；
- 同一固定256样本记忆测试，六个模型两个目标R²均超过0.97；
- 逐epoch可恢复checkpoint，包括模型、优化器、调度器、AMP、随机状态、history和早停状态。

常用命令：

```powershell
cd C:\Users\26096\Desktop\em\02_new_ipmsm_project

# 六模型结构、GPU前后向和物理batch探测
python cnn_zone\semantic90\scripts\validate_v2_models.py

# 固定256样本记忆诊断
python cnn_zone\semantic90\scripts\run_v2_overfit_256.py --resume

# 固定11,700/1,500/1,500三seed公平短测；支持逐epoch恢复
python cnn_zone\semantic90\scripts\run_v2_short_test.py --resume

# 全部18个model/seed任务完成后生成最终报告
python cnn_zone\semantic90\scripts\generate_v2_report.py

# 生成六面板状态图；未完成时留空，全部完成后自动标记complete
python cnn_zone\semantic90\scripts\make_v2_interim_plots.py
```

当前完整结果：

```text
reports/v2_test_11700/run_summary.json
reports/v2_test_11700/README.md
reports/v2_test_11700/training_validation_curves_six_v2.png
reports/v2_test_11700/tavg_prediction_six_v2.png
reports/v2_test_11700/tavg_residual_six_v2.png
reports/v2_test_11700/delta_t_prediction_six_v2.png
reports/v2_test_11700/delta_t_residual_six_v2.png
reports/v2_test_11700/legacy_vs_v2_test_mae.png
reports/v2_test_11700/interim/interim_tavg_prediction_legacy_red_v2_blue.png
```

注意：固定短测用于v1/v2受控比较，不等于146,471样本正式全量训练。224输入的改善也不能只归因于分辨率，因为v2同时修改了归一化、输出头、学习率、有效batch和部分主干结构；后续需要独立消融实验拆分原因。
