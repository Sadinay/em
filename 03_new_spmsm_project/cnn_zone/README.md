# CNN zone

Project 03 uses its own 120-bit binary SPMSM mapping and FEM geometry.

Prepared inputs:

- `logical6x20`: two channels (Air/PM), six radial by twenty angular cells.
- `xy90_224`: eight semantic channels over the complete 0-90 degree FEM quadrant.
- `polar90_224`: the same complete motor quadrant sampled on radius (0-64 mm) by physical angle (0-90 degrees).
- `xy360_224`: eight semantic channels over the complete 360-degree motor in Cartesian coordinates.
- `polar360_224`: the complete motor sampled on radius by physical angle, with circular angular padding.

The two 224 inputs are produced from the solved FEM triangle mesh. The polar image is not a visual warp of the Cartesian image. Both retain the fixed rotor, air gap, stator, windings and outer air domain; only their coordinate system differs.

The active experiment matrix uses three project-02 logical families on 6x20 (Small CNN, Mini-Inception and ResNet20), and VGG16 on four 224 views: xy90, polar90, xy360 and polar360. Seven configurations with one seed (`20260903`) produce seven runs. Every model has independent `Tavg` and `DeltaT` regression heads.

The active split contains 40,000 training, 6,483 validation and 6,483 test genes. Validation and test are unchanged from the prior audited split.

## 6x20小模型的成熟框架来源与03适配

这里的“理论支持”指成熟网络设计原则和已有论文证据，并不表示原论文能够保证本电机回归任务的精度。三个模型都不是直接照搬分类网络：输入已改为二值材料的`2 x 6 x 20`，分类头改为`Tavg`和`DeltaT`两个独立回归头。

### 1. SmallCNN V2

- **成熟来源**：经典LeNet/CNN范式，即局部连接、卷积权重共享、逐层特征提取和池化降采样。
- **03实现**：三层`3 x 3`卷积（32/64/128通道），一次`2 x 2`最大池化，将特征保持为`3 x 10`后展平回归；`DeltaT`另有一层专用卷积。
- **准确表述**：这是LeNet式的小型CNN设计，不是LeNet-5的逐层复刻。结构简单，较少过早混合6 x 20矩阵中的局部基因位置。

### 2. Mini-Inception V2

- **成熟来源**：GoogLeNet/Inception提出的并行多尺度卷积思想，同时加入ResNet式残差捷径。
- **03实现**：每个模块并行使用`1 x 1`、`3 x 3`、两个`3 x 3`近似`5 x 5`、以及`1 x 5 + 5 x 1`方向卷积分支，随后进行特征拼接和残差相加。
- **适配说明**：输入经一次`2 x 2`池化后由`6 x 20`变为`3 x 10`，最终连接两个回归头。多尺度融合具有成熟依据，但在本项目的细长小矩阵上可能过度平滑局部差异；当前高转矩端预测饱和是实测结果，不是Inception理论必然现象。

### 3. ResNet20 V2

- **成熟来源**：He等人提出的CIFAR ResNet-20。残差块学习`F(x)`并与捷径`x`相加，使较深网络更容易优化并保留恒等信息。
- **03实现**：一个16通道卷积stem，随后三个stage，每个stage包含三个BasicBlock，通道依次为16/32/64，对应ResNet-20的`6n+2, n=3`骨架。
- **适配说明**：第二stage使用步长2，将`6 x 20`降为`3 x 10`；第三stage不再继续降采样，以免径向仅剩1格。末端使用`3 x 10`自适应池化和两个独立回归头。因此它是面向6 x 20拓扑回归的ResNet20改型，而非原始CIFAR分类器。

### 双目标回归依据

三个网络均使用共享特征主干，同时设置`Tavg`和`DeltaT`两个独立输出头。这属于多任务学习的共享表示思想：相关任务可共享结构信息，同时由独立头拟合各自目标。训练时两个目标分别标准化，再对两个标准化MSE取平均，避免数值尺度大的目标支配损失。

### 原始参考文献

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278-2324. https://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
2. Szegedy, C. et al. (2015). *Going Deeper with Convolutions*. CVPR. https://arxiv.org/abs/1409.4842
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR, 770-778. https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
4. Caruana, R. (1997). *Multitask Learning*. Machine Learning, 28, 41-75. https://doi.org/10.1023/A:1007379606734
