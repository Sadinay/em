# v3 八组 CNN 架构与原版差异说明

本文记录 `cnn_comprehensive_v3` 实际运行的八组 CNN。内容以当前源码为准，可直接用于和标准
LeNet式 CNN、CIFAR ResNet20、ImageNet VGG16、ImageNet ResNet18进行对照。

> 结论先行：八组模型均从头训练。224模型保留VGG16或ResNet18的基本骨架，但不是
> torchvision未修改原版；四组基因模型也不是现成分类网络直接套用，而是面向本项目双目标回归的改型。

## 1. 八组配置的关系

### 1.1 基因矩阵输入：两种骨架 × 两种表示

| 配置ID | 骨架 | 实际张量 | 独立设计变量 | 可训练参数 |
|---|---|---:|---:|---:|
| `gene_6x20_smallcnn` | SmallCNN V2 | `B×2×6×20` | 120 | 1,224,450 |
| `gene_6x20_resnet20` | ResNet20 V2 | `B×2×6×20` | 120 | 763,714 |
| `gene_gap90_6x97_smallcnn` | SmallCNN V2 | `B×2×6×97` | 120 | 1,224,450 |
| `gene_gap90_6x97_resnet20` | ResNet20 V2 | `B×2×6×97` | 120 | 763,714 |

两个输入通道是空气和永磁体的one-hot通道。原始位为0时，空气通道为1、永磁体通道为0；
原始位为1时相反。

`6×97`并未新增优化变量。它把同一组120位基因按真实90°转子几何复制到四段设计带，并在
极间位置补入17列固定空气。网络输入变宽，但可优化变量仍为120位。

### 1.2 Polar224输入：两种骨架 × 两种物理视野

| 配置ID | 骨架 | 实际张量 | 可训练参数 | 角向边界处理 |
|---|---|---:|---:|---|
| `polar90_vgg16` | Semantic224 VGG16 V2 | `B×8×224×224` | 18,917,122 | 普通零填充 |
| `polar90_resnet18` | Semantic224 ResNet18 V2 | `B×8×224×224` | 3,845,570 | 普通零填充 |
| `polar360_vgg16` | Semantic224 VGG16 V2 | `B×8×224×224` | 18,917,122 | 角向环形填充 |
| `polar360_resnet18` | Semantic224 ResNet18 V2 | `B×8×224×224` | 3,845,570 | 角向环形填充 |

8个语义通道依次表达：设计区空气、负极性永磁体、正极性永磁体、固定空气、固定铁、绕组、
其他固定材料，以及几何域掩码。
图像由FEM物理区域查找表直接采样，不是把普通图片进行极坐标视觉变换。

## 2. SmallCNN V2：逐层结构

### 2.1 共享主干与两个输出分支

```text
输入 B×2×6×W，W=20 或 97
  │
  ├─ Conv2d(2, 32, kernel=3, stride=1, padding=1, bias=False)
  ├─ BatchNorm2d(32)
  ├─ ReLU(inplace=True)
  │
  ├─ Conv2d(32, 64, kernel=3, stride=1, padding=1, bias=False)
  ├─ BatchNorm2d(64)
  ├─ ReLU(inplace=True)
  ├─ MaxPool2d(kernel=2, stride=2)
  │
  ├─ Conv2d(64, 128, kernel=3, stride=1, padding=1, bias=False)
  ├─ BatchNorm2d(128)
  └─ ReLU(inplace=True)
       │
       ├─ Tavg分支
       │    AdaptiveAvgPool2d(3,10)
       │    Flatten：128×3×10 = 3840
       │    Linear(3840,128) → ReLU → Dropout(0.1) → Linear(128,1)
       │
       └─ DeltaT分支
            Conv2d(128,128,3,padding=1,bias=False)
            BatchNorm2d(128) → ReLU
            AdaptiveAvgPool2d(3,10)
            Flatten：128×3×10 = 3840
            Linear(3840,128) → ReLU → Dropout(0.1) → Linear(128,1)
```

输出层不加Sigmoid、Softmax或ReLU，直接输出两个连续回归值。训练时两个目标分别用训练集均值和
标准差归一化，损失为两个标准化MSE的平均值。

### 2.2 两种输入经过主干时的尺寸

| 位置 | `6×20` | `6×97` |
|---|---:|---:|
| 输入 | `2×6×20` | `2×6×97` |
| 第二层卷积后 | `64×6×20` | `64×6×97` |
| 最大池化后 | `64×3×10` | `64×3×48` |
| 共享主干末端 | `128×3×10` | `128×3×48` |
| 自适应池化后 | `128×3×10` | `128×3×10` |

因此两种输入可以使用完全相同的可训练层。`6×97`在自适应池化前保留了更长的角向物理布局，
但两组模型是分别随机初始化和分别训练的，不共享训练后的参数。

### 2.3 Dropout和归一化

- 有Dropout：Tavg与DeltaT回归头各有一个`Dropout(p=0.1)`。
- 卷积主干没有Dropout。
- 每层卷积后使用BatchNorm和ReLU。
- DeltaT比Tavg多一层专用卷积，用于学习与转矩波动更相关的局部特征。

### 2.4 与经典LeNet式网络的区别

SmallCNN没有唯一的“官方原版”。本项目只借用了LeNet/CNN的局部卷积、权重共享、池化降采样思想：

- 输入是2通道材料矩阵，而不是单通道手写数字图像；
- 使用3层共享卷积，DeltaT另有1层专用卷积；
- 使用BatchNorm和ReLU，而LeNet-5原始实现使用不同激活且没有BatchNorm；
- 使用自适应池化，使同一骨架兼容20列和97列；
- 分类器被两个独立连续值回归头替代；
- 两个回归头均加入`Dropout(0.1)`。

所以准确名称应是“LeNet式SmallCNN改型”，不能写成“原版LeNet-5”。

### 2.5 架构依据与论文中的准确表述

本模型不是任意拼接，也不是由某篇文献逐层规定的唯一结构。其成熟依据是经典
LeNet/CNN范式：局部连接、卷积权重共享、逐层特征提取以及池化降采样。项目在此基础上，
根据SPMSM基因矩阵的尺寸和双目标回归任务进行了适配：

- `6×20`输入的径向尺寸仅为6，因此全网只进行一次`2×2`降采样，避免连续池化过早压缩径向信息；
- `32/64/128`通道和三层共享卷积用于在模型容量与小尺寸输入之间取得工程折中；
- `Tavg`与`DeltaT`共享卷积主干、使用独立输出头，符合多任务学习中“共享表示＋任务专用头”的基本思路；
- `DeltaT`分支增加一层专用卷积，用于继续提取与局部转矩波动相关的特征；
- 自适应池化把不同角向宽度统一为`3×10`，使同一骨架能够分别用于`6×20`和`6×97`输入。

需要限定结论：上述选择具有成熟架构原则和任务尺寸约束作为依据，但本项目没有对卷积层数、
通道数以及`DeltaT`专用卷积逐项进行消融实验，因此不能宣称当前组合是理论最优或唯一合理结构。
其有效性应由本项目中与ResNet20等架构在相同数据划分和评价口径下的实验比较来支持。

论文中可表述为：

> 本研究构建了一种面向`2×6×20`二值材料矩阵的LeNet式SmallCNN回归模型。该模型遵循局部连接、
> 卷积权重共享和池化降采样等经典CNN原则，并针对输入的狭长尺寸仅执行一次空间降采样。
> 网络采用共享卷积主干和两个任务专用回归头，分别预测平均转矩与转矩波动；其中转矩波动分支
> 增加一层专用卷积以进一步提取局部变化特征。具体层数与通道数属于面向本任务的工程设计，
> 不主张其为理论最优结构。

参考依据：

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278–2324. https://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
2. Caruana, R. (1997). *Multitask Learning*. Machine Learning, 28, 41–75. https://doi.org/10.1023/A:1007379606734

## 3. ResNet20 V2：逐层结构

### 3.1 BasicBlock定义

每个残差块为：

```text
主路径：Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm
捷径：尺寸和通道不变时为Identity；发生降采样或通道变化时为Conv1×1 → BatchNorm
输出：ReLU(主路径 + 捷径)
```

### 3.2 完整网络

```text
输入 B×2×6×W，W=20 或 97
  │
  ├─ Stem：Conv3×3(2→16, stride=1) → BN → ReLU
  ├─ Stage 1：3个BasicBlock，16通道，全部stride=1
  ├─ Stage 2：3个BasicBlock，32通道，首块stride=2，其余stride=1
  ├─ Stage 3：3个BasicBlock，64通道，全部stride=1
  ├─ AdaptiveAvgPool2d(3,10)
  ├─ Flatten：64×3×10 = 1920
  ├─ Tavg头：Linear(1920,128) → ReLU → Dropout(0.1) → Linear(128,1)
  └─ DeltaT头：Linear(1920,128) → ReLU → Dropout(0.1) → Linear(128,1)
```

### 3.3 两种输入的尺寸变化

| 位置 | `6×20` | `6×97` |
|---|---:|---:|
| Stem/Stage 1 | `16×6×20` | `16×6×97` |
| Stage 2 | `32×3×10` | `32×3×49` |
| Stage 3 | `64×3×10` | `64×3×49` |
| 自适应池化后 | `64×3×10` | `64×3×10` |

### 3.4 Dropout和归一化

- 有Dropout：两个回归头各有一个`Dropout(p=0.1)`。
- 残差主干没有Dropout。
- 所有卷积路径和需要投影的捷径都使用BatchNorm。
- 所有残差块均使用ReLU。

### 3.5 与标准CIFAR ResNet20的区别

保留部分：`3×3` stem、16/32/64三个stage、每个stage三个BasicBlock，即`6n+2，n=3`的
ResNet20层数组织。

修改部分：

- 输入从RGB三通道改为2通道空气/永磁体；
- 标准CIFAR ResNet20通常在Stage 2和Stage 3各降采样一次，本项目仅在Stage 2降采样；
- Stage 3保持stride=1，避免6格径向尺寸进一步缩小；
- 标准网络通常全局平均池化成`64×1×1`，本项目自适应池化成`64×3×10`；
- 单分类层改为两个`1920→128→1`回归头；
- 两个回归头增加`Dropout(0.1)`；
- 本项目投影捷径使用`1×1`卷积加BatchNorm。

因此它是“CIFAR ResNet20骨架的6×W双回归改型”，不是原始分类器直接复用。

### 3.6 架构依据与论文中的准确表述

本模型直接借鉴He等人提出的残差学习思想，并沿用面向CIFAR小图像的ResNet20层数组织：
`3×3` stem之后设置三个stage，每个stage含三个BasicBlock，每个BasicBlock包含两层`3×3`卷积，
从而形成`6n+2（n=3）`的20层骨架。残差捷径使主路径只需学习相对于输入的变化，并为深层网络
提供更直接的特征和梯度传播通道。

针对本项目的修改包括：

- 将`3×32×32` RGB图像输入改为`2×6×W`空气/永磁体材料矩阵；
- 保留`16/32/64`三阶段通道配置，但只在Stage 2进行一次降采样，避免6格径向尺寸在Stage 3再次减半；
- 以`3×10`自适应池化代替标准全局平均池化，保留一定的径向和角向位置信息；
- 将十分类输出替换为两个独立回归头，分别预测平均转矩和转矩波动，并在回归头中加入`Dropout(0.1)`；
- 两种输入宽度分别训练，均从随机初始化开始，不使用图像预训练权重。

这些修改保留了ResNet20的残差学习机制和总体深度，但输入表示、降采样策略、末端特征聚合与输出任务
均已针对电机代理建模重新设计。因此论文中宜称为“基于CIFAR ResNet20骨架改造的双目标回归网络”。
本项目没有对保留空间尺寸和回归头宽度逐项进行消融，不能宣称该组合是唯一或理论最优选择。

论文中可表述为：

> 本研究以CIFAR ResNet20的三阶段残差骨架为基础，构建适用于`2×6×W`材料矩阵的双目标回归模型。
> 为避免狭窄径向维度被过度压缩，网络仅在第二阶段进行一次空间降采样，并通过`3×10`自适应池化
> 保留部分空间分布信息。原分类层被两个任务专用回归头替代，用于分别预测平均转矩与转矩波动。

参考依据：He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770–778.
https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

## 4. Semantic224 VGG16 V2：逐层结构

### 4.1 卷积块

每个卷积均为`3×3、stride=1、padding=1、bias=False`，每层顺序固定为：

```text
Conv2d → GroupNorm → ReLU
```

五个卷积块为：

| Block | 卷积层数 | 通道 | 末端操作 | 输出空间尺寸 |
|---|---:|---:|---|---:|
| 1 | 2 | 8→64→64 | MaxPool 2×2 | 112×112 |
| 2 | 2 | 64→128→128 | MaxPool 2×2 | 56×56 |
| 3 | 3 | 128→256→256→256 | MaxPool 2×2 | 28×28 |
| 4 | 3 | 256→512→512→512 | MaxPool 2×2 | 14×14 |
| 5 | 3 | 512→512→512→512 | MaxPool 2×2 | 7×7 |

之后执行：

```text
AdaptiveAvgPool2d(4,4)
Flatten：512×4×4 = 8192
Tavg头：Linear(8192,256) → ReLU → Dropout(0.2) → Linear(256,1)
DeltaT头：Linear(8192,256) → ReLU → Dropout(0.2) → Linear(256,1)
```

### 4.2 Dropout和归一化

- 有Dropout：两个回归头各有一个`Dropout(p=0.2)`。
- 卷积块没有Dropout。
- 每个卷积后均使用GroupNorm和ReLU。
- 64和128通道层使用8组GroupNorm；256和512通道层使用16组GroupNorm。

### 4.3 与原始VGG16的区别

保留部分：五个卷积块、每块卷积数`2/2/3/3/3`、通道`64/128/256/512/512`，因此卷积深度
仍对应VGG16骨架。

修改部分：

- 输入由RGB三通道改为8通道电机材料语义图；
- 原始VGG16卷积后没有归一化，本项目每层加入GroupNorm；
- 原始VGG16池化后为`512×7×7=25088`，本项目再自适应到`512×4×4=8192`；
- 原始分类器通常为`25088→4096→4096→1000`并含两次`Dropout(0.5)`；
- 本项目改为两个独立的`8192→256→1`回归头，每个仅一次`Dropout(0.2)`；
- 本轮不加载ImageNet预训练权重，全部从头初始化；
- Polar360版本把有padding的卷积改为径向零填充、角向环形填充。

### 4.4 架构依据与论文中的准确表述

本模型借鉴Simonyan和Zisserman提出的VGG16：保留五个卷积块、每块卷积数`2/2/3/3/3`以及
`64/128/256/512/512`的通道递增规律。它因此保留了VGG通过连续小尺寸`3×3`卷积逐层扩大感受野、
逐级形成高层特征的核心思路。

针对本项目的修改包括：

- 将3通道自然图像输入改为8通道电机材料语义图；
- 在每层卷积后加入GroupNorm，以适应本轮较小的物理batch；
- 通过`4×4`自适应池化将主干输出统一为8192维，而不沿用原版`7×7`池化结果；
- 用两个轻量回归头替代原版三层大规模分类器，分别预测平均转矩和转矩波动；
- 将分类器中的`Dropout(0.5)`调整为每个回归头中的一次`Dropout(0.2)`；
- 不加载ImageNet预训练权重；Polar360输入额外采用角向环形填充以表达圆周连续性。

因此该模型不是原版VGG16分类器的直接迁移，而是“VGG16风格的8通道双目标回归网络”。
VGG16名称描述的是被保留的卷积块组织；归一化、池化、输出头和边界处理均为本项目的任务适配。

论文中可表述为：

> 本研究保留VGG16的五级卷积块及`2/2/3/3/3`卷积层配置，将输入层扩展为8通道材料语义图，
> 并在卷积后引入GroupNorm。网络末端采用`4×4`自适应池化及两个独立回归头，以分别预测平均转矩
> 与转矩波动；对于全圆周输入，卷积在角向采用环形填充以保持周期边界的连续性。

参考依据：Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*.
International Conference on Learning Representations. https://arxiv.org/abs/1409.1556

## 5. Semantic224 ResNet18 V2：逐层结构

### 5.1 残差块

BasicBlock与前述ResNet20相同，但归一化由BatchNorm改为GroupNorm。每个归一化层使用8组。

### 5.2 完整网络

```text
输入 B×8×224×224
  │
  ├─ Stem：Conv3×3(8→32, stride=1, padding=1) → GroupNorm(8组) → ReLU
  │          不使用7×7卷积，不使用初始MaxPool
  ├─ Stage 1：2个BasicBlock，32通道，stride=1       → 32×224×224
  ├─ Stage 2：2个BasicBlock，64通道，首块stride=2   → 64×112×112
  ├─ Stage 3：2个BasicBlock，128通道，首块stride=2  → 128×56×56
  ├─ Stage 4：2个BasicBlock，256通道，首块stride=2  → 256×28×28
  ├─ AdaptiveAvgPool2d(4,4)
  ├─ Flatten：256×4×4 = 4096
  ├─ Tavg头：Linear(4096,128) → ReLU → Dropout(0.1) → Linear(128,1)
  └─ DeltaT头：Linear(4096,128) → ReLU → Dropout(0.1) → Linear(128,1)
```

### 5.3 Dropout和归一化

- 有Dropout：两个回归头各有一个`Dropout(p=0.1)`。
- Stem和残差主干没有Dropout。
- 全部归一化层为8组GroupNorm，不使用BatchNorm。

### 5.4 与标准ImageNet ResNet18的区别

保留部分：四个stage、每个stage两个BasicBlock，即`2/2/2/2`的ResNet18骨架。

修改部分：

- 输入从3通道RGB改为8通道材料语义图；
- 标准stem通常为`7×7、64通道、stride=2`加`3×3 MaxPool`，本项目改为
  `3×3、32通道、stride=1`且取消初始MaxPool；
- stage通道从标准`64/128/256/512`减为`32/64/128/256`；
- BatchNorm改为GroupNorm，以适应较小物理batch；
- 标准全局平均池化输出`512×1×1`，本项目输出`256×4×4`；
- 单分类层改为两个`4096→128→1`回归头；
- 两个回归头各增加`Dropout(0.1)`；
- 本轮不加载ImageNet预训练权重；
- Polar360版本对卷积采用角向环形填充。

### 5.5 架构依据与论文中的准确表述

本模型借鉴的是标准ImageNet ResNet18，而不是CIFAR ResNet20。它保留四个stage以及每个stage含两个
BasicBlock的`2/2/2/2`组织，并保留残差主路径与捷径相加的基本机制。

针对本项目的修改包括：

- 将3通道RGB输入改为8通道电机材料语义图；
- 将原版`7×7、64通道、stride=2`的stem和初始MaxPool，改为`3×3、32通道、stride=1`且不设初始池化，
  以减少输入早期的信息损失；
- 将四阶段通道从`64/128/256/512`缩减为`32/64/128/256`，控制模型容量和计算量；
- 将BatchNorm替换为8组GroupNorm，以降低较小物理batch对归一化统计的影响；
- 以`4×4`自适应池化替代`1×1`全局平均池化，并将单一分类层替换为两个`4096→128→1`回归头；
- 不加载ImageNet预训练权重；Polar360输入额外采用角向环形填充。

因此它应被描述为“基于ImageNet ResNet18骨架的轻量化8通道双目标回归改型”。与ResNet20改型相比，
它保留的是四阶段`2/2/2/2`骨架；二者虽都采用残差块，但来源、输入尺度和阶段组织不同。

论文中可表述为：

> 本研究以ImageNet ResNet18的四阶段残差骨架为基础，将输入层改为8通道，并以较小的`3×3` stem
> 取代原始`7×7`卷积和初始最大池化。各阶段通道数缩减一半，BatchNorm替换为GroupNorm，网络末端
> 使用`4×4`自适应池化与两个独立回归头，从而适配电机材料语义图上的双目标连续值预测。

参考依据：He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770–778.
https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

## 6. Polar90与Polar360的网络差别

同一骨架的90°和360°模型拥有相同的可训练层和参数量，但输入含义与边界处理不同：

- Polar90覆盖一个90°物理扇区，角向两端不是同一个位置，因此使用普通零填充。
- Polar360覆盖完整圆周，0°和360°物理相邻，因此角向使用circular padding。
- 环形处理只发生在角向宽度方向；径向上下边界仍使用零填充。
- 四个Polar模型分别初始化、分别训练，不共享训练后的权重。

## 7. 本轮训练设置

| 模型组 | 最大轮数 | 最小轮数 | 提前停止 | 初始学习率 | 物理batch | 梯度累积 | 有效batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmallCNN，两种输入 | 100 | 20 | 连续12轮验证无改善 | `1e-3` | 64 | 1 | 64 |
| ResNet20，两种输入 | 100 | 20 | 连续12轮验证无改善 | `3e-4` | 64 | 1 | 64 |
| VGG16，两种Polar输入 | 40 | 30 | 连续8轮验证无改善 | `1e-4` | 8 | 8 | 64 |
| ResNet18，两种Polar输入 | 40 | 30 | 连续8轮验证无改善 | `1e-4` | 16 | 4 | 64 |

共同设置：

- 优化器：AdamW；
- 权重衰减：`1e-4`；
- 学习率调度器：ReduceLROnPlateau，验证损失连续3轮不改善时学习率乘0.5，最低`1e-6`；
- 两个目标分别按训练集统计量标准化；
- 损失为Tavg和DeltaT标准化MSE的平均值；
- 使用验证集标准化MSE保存最佳检查点；
- 测试集不参与归一化、调参或检查点选择；
- 八组均从随机初始化开始，不加载f0/f1/f2、SP模型或ImageNet预训练权重。

## 8. 源码对应位置

- 网络逐层定义：`cnn_zone/src/models_v2.py`
- one-hot、gap-aware和8通道Polar输入：`cnn_zone/src/training_inputs.py`
- 训练循环、模型创建和环形填充：`cnn_zone/src/training.py`
- 八组训练超参数：`experiments/cnn_comprehensive_v3/train.py`
- `6×97`物理映射：`experiments/cnn_comprehensive_v3/data/gap90/geometry.json`
