# Historical March CLONALG 算法规格

状态：编码前冻结规格  
基准入口：`FP/CLONALG/clonalg_N_float.m`  
几何模式：`historical_inset`  
本规格区分“MATLAB源码明确”“历史trace验证”“未确认”。

## 1. 总体模式

| 项目 | 规格 | 证据 |
|---|---|---|
| 优化类型 | 单目标最小化 | `clonalg_N_float.m:282-304` |
| 算法 | 自定义CLONALG | `clonalg_N_float.m:312-445` |
| 交叉 | 不存在，不得加入 | 全入口无交叉调用；`clonalg_N_float.m:312-445` |
| 染色体 | 180位离散三值向量 | `clonalg_N_float.m:22-32,142` |
| 基因集合 | `{0,1,2}` | `clonalg_N_float.m:73-76`; `decode_material_bits.m:10-11` |
| 材料含义 | 0空气、1铁、2铜 | `decode_material_bits.m:2-3` |
| 种群规模 | 10 | `clonalg_N_float.m:127`；trace中的`N=10` |
| 每次运行代数 | 100 | `clonalg_N_float.m:129`；trace中的`gen=100` |
| 停止条件 | 完成100代；第100代评估后立即结束，不再生成下一代 | `clonalg_N_float.m:248-310` |
| 早停/容差 | 不存在 | `clonalg_N_float.m:248-446` |

## 2. 设计域与排列

| 项目 | 规格 | 证据 |
|---|---|---|
| 径向格数 | `nr=18` | `clonalg_N_float.m:22` |
| 角向格数 | `nt=10` | `clonalg_N_float.m:22` |
| 设计域矩阵 | MATLAB逻辑矩阵为`[nt,nr]=[10,18]` | `stator_design_domain_blank.m:55-61` |
| MATLAB线性顺序 | `find`的列优先顺序 | `prepare_design_domain.m:21-23` |
| 可视矩阵 | `reshape(bits,[nt,nr]).'`得到`[nr,nt]=[18,10]` | `detect_floating_iron.m:8`; `detect_floating_copper_small_islands.m:15` |
| Python等价 | `bits.reshape((10,18), order='F').T` | 由上述MATLAB表达式直接等价转换 |

## 3. 初始化

| 项目 | 规格 | 证据 |
|---|---|---|
| 续跑种子 | 加载上一阶段MAT中的`best_bits` | `clonalg_N_float.m:120-123` |
| 全随机底板 | `N×Nd`，每基因独立均匀取0/1/2 | `clonalg_N_float.m:142` |
| 原种子 | 第1个体原样使用 | `clonalg_N_float.m:144` |
| 近邻数 | `round(0.6*(N-1))=5` | `clonalg_N_float.m:146,156-160` |
| 近邻变异率 | 每基因0.05 | `clonalg_N_float.m:150,158` |
| 中扰动数 | `round(0.2*(N-1))=2` | `clonalg_N_float.m:147,162-166` |
| 中扰动率 | 每基因0.12 | `clonalg_N_float.m:151,164` |
| 远扰动数 | 1 | `clonalg_N_float.m:168-171` |
| 远扰动率 | 每基因0.25 | `clonalg_N_float.m:152,170` |
| 剩余随机数 | 1（N=10时） | `clonalg_N_float.m:142,173-176` |

MATLAB `round` 的半数规则在这些默认参数上不影响结果；若未来改变N，MATLAB与Python的半数取整差异需要单独定义。目前兼容配置固定N=10。

## 4. 适应度排序

| 项目 | 规格 | 证据 |
|---|---|---|
| 方向 | J越小越好 | `clonalg_N_float.m:282-299` |
| 调用 | `[J_sorted,ind]=sort(J)` | `clonalg_N_float.m:283` |
| 相同J源码规则 | 未显式指定，源码依赖运行MATLAB版本的默认`sort`行为 | `clonalg_N_float.m:283` |
| 历史观测 | 最新100代的`ind_hist`在全部100代均与稳定升序一致；相同J保留输入顺序 | `trace_all_20260306_174158.mat`回归分析 |
| Python兼容实现 | 稳定升序排序；这是由历史trace确定，不宣称适用于未知MATLAB版本 | 历史观测 |

## 5. 克隆数量

| 项目 | 规格 | 证据 |
|---|---|---|
| 参与克隆父代 | 排名前`ns=max(0,N-2)=8` | `clonalg_N_float.m:315` |
| `beta` | 0.4 | `clonalg_N_float.m:136` |
| 最大克隆数 | `cs_max=beta*N=4` | `clonalg_N_float.m:316` |
| 最小克隆数 | 1 | `clonalg_N_float.m:317` |
| 公式 | `round(cs_max-(cs_max-cs_min)*(rank-1)/max(1,ns-1))` | `clonalg_N_float.m:321-325` |
| 默认结果 | `[4,4,3,3,2,2,1,1]`，总计20 | 按源码公式计算 |

这里必须使用MATLAB正数round语义（`.5`向远离0方向）。

## 6. 超变异

| 项目 | 规格 | 证据 |
|---|---|---|
| 与适应度关系 | 不直接使用J数值，只使用父代适应度排名 | `clonalg_N_float.m:331-340` |
| 最佳父代变异率 | 0.03/基因 | `clonalg_N_float.m:332` |
| 最差被克隆父代变异率 | 0.20/基因 | `clonalg_N_float.m:333` |
| 曲线 | `p(rank)=0.03+(0.20-0.03)*((rank-1)/(ns-1))^2` | `clonalg_N_float.m:334,338-341` |
| 每个克隆改变基因数 | 不是固定数量；180个基因分别独立Bernoulli(p)，所以理论范围0～180 | `clonalg_N_float.m:351-365` |
| 是否允许零变异 | 允许；掩膜可能全false | `clonalg_N_float.m:362,368` |
| 新基因生成 | 对被选基因随机均匀加1或2，再模3；一定变成另一个合法值 | `clonalg_N_float.m:367-373` |

## 7. 父代与克隆择优

| 项目 | 规格 | 证据 |
|---|---|---|
| 每簇候选 | 一个父代与该簇最小J克隆 | `clonalg_N_float.m:396-406` |
| 克隆胜出条件 | `bestCloneJ <= parentJ` | `clonalg_N_float.m:408-413` |
| 相同J | 选择克隆，而不是父代 | `clonalg_N_float.m:409-412` |
| 同簇多个克隆相同最小J | MATLAB `min`返回第一个最小值 | `clonalg_N_float.m:405-406` |
| 后两名 | 不克隆，按当前适应度排名原样继承 | `clonalg_N_float.m:418-421` |

## 8. 随机移民与精英

| 项目 | 规格 | 证据 |
|---|---|---|
| 移民比例 | `d=0.2` | `clonalg_N_float.m:135` |
| 数量 | `max(1,round(d*N))=2` | `clonalg_N_float.m:423-425` |
| 可覆盖位置 | `2:N`中的无放回随机位置 | `clonalg_N_float.m:426-428` |
| 50%分支A | 从全局最优以0.25/基因做三值变异 | `clonalg_N_float.m:429-434` |
| 50%分支B | 180个基因独立均匀随机0/1/2 | `clonalg_N_float.m:431,435-437` |
| 精英数量 | 1 | `clonalg_N_float.m:441-443` |
| 精英语义 | 运行以来全局最优强制写入下一代第1行，覆盖该位置任何此前内容 | `clonalg_N_float.m:301-307,441-443` |

## 9. 缓存与批内去重

| 项目 | 规格 | 证据 |
|---|---|---|
| 运行内缓存 | `containers.Map(char -> double J)` | `clonalg_N_float.m:246` |
| 缓存键 | 将染色体转为uint8，每位加字符`'0'`，形成180字符的0/1/2字符串 | `clonalg_N_float.m:993-997,1018-1020` |
| 批内去重 | `unique(X8,'rows','stable')` | `clonalg_N_float.m:1005-1009` |
| 批内映射 | 用`ic`恢复原始输入顺序 | `clonalg_N_float.m:1039-1040` |
| 缓存内容 | 只保存J，不保存T_avg/T_ripple/错误/耗时 | `clonalg_N_float.m:1011-1037` |

Python本阶段保留相同180字符基因键，同时checkpoint保存缓存。Mock evaluator可以返回结构化结果，但兼容性排序只使用J。

## 10. 预检查和无效目标值

| 项目 | 规格 | 证据 |
|---|---|---|
| 铁连通 | 4邻接；必须连到最外径向行；角向不周期连接 | `detect_floating_iron.m:19-63`; `clonalg_N_float.m:79` |
| 小铜岛 | 4邻接；连通分量格数`<4`时该分量全部单元计为悬浮铜 | `detect_floating_copper_small_islands.m:64-95`; `clonalg_N_float.m:80` |
| 任一悬浮 | `J=1e6+50*nFloatCu+50*nFloatFe` | `eval_design_femm_float.m:3-18` |
| 无铜 | 若匝数全部为0，`J=1e6` | `eval_design_femm_float.m:25-30` |
| 无效返回指标 | `T_avg=0, T_ripple=0` | `eval_design_femm_float.m:15-16,28` |
| FEMM失败值 | 未确认；源码没有异常捕获或统一失败J | 全评估函数 |

`fastReject.AminCu`虽存在于配置，但评估实际使用`ctx.Amin`；两者当前均为4。兼容实现使用明确配置项4，并在报告中保留该历史缺陷说明。

## 11. 匝数分配

| 项目 | 规格 | 证据 |
|---|---|---|
| 六电路总匝数 | 每个均为100 | `clonalg_N_float.m:43,64` |
| 扇区电路ID | `[4,3,6,5,2,1]` | `clonalg_N_float.m:34-40` |
| 铜格计数 | 对每个扇区和每相统计`mat_code==2 && pid==p` | `compute_turns_per_cell.m:17-25` |
| 每铜格匝数 | `N_phase_total[p]/nCell_phase[p]` | `compute_turns_per_cell.m:27-34` |
| 无铜相 | 0 | `compute_turns_per_cell.m:32-33` |

## 12. historical_inset解码规格

纯Python本阶段不输出FEMM命令，而输出可验证的中间拓扑描述：

- 一个0～15°基础设计域的18×10材料矩阵；
- 六个15°扇区，偶数扇区镜像；
- 每个材料单元的扇区、径向/角向索引、材料、电路与匝数；
- 每个铜格的内缩边界，径向和角向两侧均缩进20%；
- 铜格外层背景为空气；相邻铜格仍分别有独立内缩铜区，不执行V5铜连通块合并；
- 空气/铁相邻同材大格的内部边界按旧代码合并语义标记。

证据：`femm_apply_design_bits_rep6_inset.m:5-127,134-187,207-278`。

## 13. 标量目标函数

来源：`scantorque.m:62-65`、`eval_design_femm_float.m:52-60,63-79`。

```text
T_avg = arithmetic_mean(T)
T_ripple = (max(T)-min(T)) / max(abs(T_avg), 1e-6)

torque_penalty = 10*(0.8-T_avg)/0.8,  if T_avg < 0.8
torque_penalty = 0,                     otherwise

J_em = T_ripple + torque_penalty
pFe = 0.10 / (1 + exp(-50*(nFloatFe-0.5)))
pCu = 0.05 / (1 + exp(-30*(nFloatCu-0.5)))
J = J_em + pFe + pCu
```

由于硬拒绝先执行，正悬浮计数不会进入最后的sigmoid公式；无悬浮时两项仍是极小非零数。六角度 `[0,3,6,9,12,15]` 仅是历史兼容定义，科学充分性未确认，本阶段不声称六点能可靠表征真实转矩脉动。

## 14. 随机数兼容目标

### 14.1 要求实现

- 允许显式Python seed；
- 所有算法随机操作只使用一个显式传递/持有的NumPy Generator；
- checkpoint保存完整bit generator状态；
- 相同Python seed、配置和mock evaluator得到完全相同结果；
- 中断恢复与不中断运行完全一致。

### 14.2 不要求且不得声称

- 不要求Python随机数与MATLAB逐次相同；
- 不声称逐代复现历史MATLAB种群；
- 历史入口使用`rng('shuffle')`（`clonalg_N_float.m:125`），且最终trace未保存`rng_hist`，历史随机流不可恢复。

## 15. 历史续跑实际载入状态

源码加载 `best_result_0305_groupF_900gen.mat` 中的 `best_bits` 和 `cfg`（`clonalg_N_float.m:121`），但后续只使用 `S.best_bits`（第122～123行）。

没有载入：

- 原种群；
- 代数计数；
- 全局最优J；
- 历史数组；
- 缓存；
- 随机状态；
- worker/FEMM状态。

因此历史所谓续跑实际是“以上一阶段best作为新种子，重新初始化一次新的100代运行”，不是checkpoint恢复。

## 16. 历史回归发现的源码/输出版本漂移

以 `IA_test_Float/trace_all_20260306_174158.mat` 的100代×10个体为基准：

- 现存 `detect_floating_copper_small_islands.m` 对小铜岛单元数达到1000/1000逐项一致；
- 现存 `detect_floating_iron.m` 对浮铁单元数仅901/1000逐项一致；99个位置（55个唯一染色体）的历史计数不能由当前源码产生；
- 由于这99个位置也都触发小铜岛，Python按当前源码得到的最终拒绝/通过判定仍为1000/1000一致；
- 使用MAT内保存的 `nFloatFe_hist`、`nFloatCu_hist` 代入硬拒绝公式，900/900个无效J完全一致；使用当前源码重新计算计数时只有801/900个无效J完全一致。

该差异不是排列转换问题：相同染色体在历史文件中的浮铁计数始终一致，且多种替代排列、周期边界和邻接定义均不能解释差异。历史MAT未保存运行时源码快照或规则版本号，因此准确历史浮铁计数算法标记为“未确认（疑似源码/输出版本漂移）”。兼容实现保留当前可审计MATLAB源码的规则，不为追逐结果而暗改规则。

另有两类历史回归样本缺失：扫描可读的19个 `trace_all_*.mat` 未发现无铜染色体；这些trace也未保存 `T_avg`、`T_ripple` 或六点转矩数组。因此无铜分支和低平均转矩分支只能用合成数据进行逻辑测试，不能声称已由历史FEMM输出回归。
