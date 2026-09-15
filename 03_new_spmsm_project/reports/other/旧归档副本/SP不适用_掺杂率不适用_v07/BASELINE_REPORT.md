# 冻结旧 CNN 的新增结构基线评估

全量接收核验通过：3299 个基因、19794 个角度；G/F 两套训练数据均完整就绪。旧模型在原验证集上的推理与历史记录一致，但在新增结构上误差明显增加。

本次未训练更新模型、未开发路由、未调用 FEMM、未预测旧 test 或公共新 test。

## 数据和模型身份

f0 为 polar90_224 / vgg16_v2 / seed_20260903，旧 train=40000，最佳 epoch=39。权重 SHA-256 `34cdee2e373e36ecd40e82197de5ea34941397bafea5f586cb00eb87dbfe58ba`。复用原 training.evaluate、8 通道 Polar90 renderer、通道顺序和 checkpoint 的目标均值/标准差；eval 模式，模型状态前后逐张量完全一致。

实际设备：NVIDIA GeForce RTX 5070，Original training.evaluate AMP float16 on CUDA; float32 inverse scaling; original batch size 8。共推理 9382 个唯一基因：旧 validation 6483、新训练并集 2699、公共 dev 200。G/F 的 101 个交集只推理一次。

旧 validation 与 history.json 第39轮的物理误差指标最大差 0 N·m（预设一致性容差1e-5）。比较的是旧验证记录，没有使用旧测试指标代替。

## 关键误差

残差统一为预测−真实；所有误差单位 N·m，DeltaT 是六点峰峰差，非百分比。

| 数据 | N | Tavg MAE | Tavg RMSE | Tavg P95 | DeltaT MAE | DeltaT RMSE | DeltaT P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_validation | 6483 | 0.006412 | 0.011522 | 0.017271 | 0.011520 | 0.018543 | 0.030049 |
| train_G | 1400 | 0.133085 | 0.210818 | 0.483960 | 0.065865 | 0.114425 | 0.220092 |
| train_F | 1400 | 0.165274 | 0.274259 | 0.666358 | 0.170960 | 0.312928 | 0.686871 |
| dev_common | 200 | 0.191545 | 0.332013 | 0.808113 | 0.096163 | 0.175073 | 0.392445 |

公共 dev 与旧 validation 的 MAE 倍数：tavg 29.9 倍, delta_t 8.3 倍。这是本批结构与旧模型之间的初始差距，不是补样训练后的表现。

## 公共 dev 来源差异

| 来源 | N | Tavg MAE | Tavg 最大误差 | Tavg 偏差 | DeltaT MAE | DeltaT 最大误差 | DeltaT 偏差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| U | 50 | 0.300022 | 1.029459 | +0.267239 | 0.071663 | 0.344571 | -0.034499 |
| L | 50 | 0.161894 | 0.876443 | +0.094238 | 0.093518 | 0.688050 | -0.072508 |
| B | 50 | 0.296325 | 0.913720 | +0.188734 | 0.201137 | 1.199012 | -0.169628 |
| P | 50 | 0.007941 | 0.034521 | +0.001794 | 0.018335 | 0.065094 | -0.011006 |

按公共 dev 的 MAE，tavg 最困难来源为 U（0.300022 N·m）。

按公共 dev 的 MAE，delta_t 最困难来源为 B（0.201137 N·m）。

图中可见的具体误差：

- 公共 dev 的 U 来源 `7de26b20fd14`：tavg 真实值 0.28677，预测 1.31623，偏差 +1.02946 N·m。

- 公共 dev 的 B 来源 `209242a91f57`：delta_t 真实值 2.95429，预测 1.75528，偏差 -1.19901 N·m。

这些例子表现为低转矩结构被高估、较高转矩波动被低估，是当前数据上的观察；仅凭本次基线不能确定其因果机制。

U=独立位置配置，L=小尺度相关结构，B=大尺度相关结构，P=旧训练父代的空间扰动。每来源仅50个 dev 样本，尾部与小差异不宜过度解释。P父代仍属于旧train，不能称为完全未见父代。

## G/F 样本组成与就绪情况

G 来源 U/L/B/P=680/617/51/52；F=281/355/567/197。各1400，交集101；各自选样顺序和独立视图保留。磁体面积按492区域参考网格中的设计区域面积求和，再按四扇区换算；不是直接把格数叫面积。最近 Hamming 复用经过原始清单哈希核验的结果，参考集仅为旧train40000。

G-S/G-E 将共享旧40000+train_G；F-S/F-E 将共享旧40000+train_F。数据和固定f0输入链路具备进入后续公平训练对照的条件，无缺失标签或缺失模型权重。当前本机 FEMM 配置的旧版本问题不影响已完成标签或CNN训练，但会阻止安全复用当前求解入口，后续重算前必须处理。

旧模型在G或F上误差较高，只说明对应样本对旧模型更困难，不能据此判定哪种选样方法更有效。G的固定位置WL0/1仅编码局部材料和一圈邻接，没有显式强化桥接断裂、气隙或长程连接；F可能继承旧CNN的盲区。两者均不是已经验证的物理距离，Hamming也不保证精度。

## 图表与大误差案例

![真实值与预测值](<../../experiments/input_distribution_pilot_v1/post_femm_baseline_20260913/figures/truth_prediction.png>)

![公共dev按来源的绝对误差](<../../experiments/input_distribution_pilot_v1/post_femm_baseline_20260913/figures/dev_source_errors.png>)

![G/F组成与结构分布](<../../experiments/input_distribution_pilot_v1/post_femm_baseline_20260913/figures/GF_structure_distributions.png>)

![训练/dev大误差结构](<../../experiments/input_distribution_pilot_v1/post_femm_baseline_20260913/figures/large_error_structures.png>)

例图分别取公共dev及新训练并集的Tavg/DeltaT大误差结构；未从test挑选。逻辑图采用bits.reshape(20,6).T，材料图直接取原8通道renderer输出的材料彩色合成。详细身份与数值见 figures/structure_cases.csv。

## 下一阶段

先执行原定G/F两个更新模型的公平对照：相同架构、同一f0初始化、配对随机种子、相同优化和训练预算，使用共同dev并同时监测旧validation。E组复用同一fG/fF与冻结f0，不额外训练两套有利模型。路由阈值仅在旧validation与新dev开发，最终test继续封存。

每个更新模型计划40000+1400条训练样本，新增占3.38%，这是样本比例，不代表梯度贡献比例。本轮只测出初始差距，不能证明1400足够或不足。若后续检验样本量趋势，可用冻结顺序的嵌套700/1400子集，并保持设置可比；本次未执行。不能仅因旧模型误差大就扩充几千次FEMM，也不能宣称3299个样本覆盖全部结构空间。

完整指标含N、MAE、RMSE、中位数、P90/P95、最大误差、偏差、R²和Spearman，见 baseline_metrics.csv/.json。原始预测及残差在 baseline_predictions.csv；逐条FEMM证据通过完整gene_id可追溯到received归档。
