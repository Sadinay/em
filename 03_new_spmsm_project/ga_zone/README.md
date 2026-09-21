# 120 位二值基因的 NSGA-II–CNN–FEMM 搜索

主程序和全部配置集中在 `run_nsga2.py`。每代从当前父代生成约 96 个新基因，主代理 Polar90 ResNet18 预测平均转矩与绝对转矩波动；另外三个已选 CNN 与它共同给出四模型分歧。分歧用四个固定模型的总体标准差（分母为 4），**只是选点线索，不是已校准的置信区间**。

## 安全的运行顺序

从项目根目录运行：

```powershell
python ga_zone/run_nsga2.py --self-test
python ga_zone/run_nsga2.py --name first_search
```

默认只生成第一代 FEMM 清单并**暂停**，不会启动 FEMM。要逐代执行真实 FEMM，可以对暂停的运行依次调用：

```powershell
python ga_zone/run_nsga2.py --name first_search --solve-pending
python ga_zone/run_nsga2.py --name first_search --resume
```

第二条命令只求解当前清单中的基因，每个基因必须完成同一工况下六个机械角度，才能产生平均转矩和峰峰转矩波动真值。第三条命令核验六个角度的原始结果、模型文件哈希、基因和工况，然后将真值纳入本代的父代加子代选择，并生成下一代清单后再次暂停。中断的 FEMM 求解保留已有角度结果，可再次执行 `--solve-pending`。另一种明确授权的方式是 `--femm-mode live`，它会连续执行所有代的真实 FEMM，成本很高；**不要把它用于试运行**。

续跑会自动沿用原运行的种群、代数、随机种子、FEMM 比例和计算设备；若显式传入与原记录冲突的参数，程序会拒绝继续。

先做很小的真实闭环试验可用：

```powershell
python ga_zone/run_nsga2.py --name tiny_check --population 4 --generations 1 --device cpu
python ga_zone/run_nsga2.py --name tiny_check --solve-pending
python ga_zone/run_nsga2.py --name tiny_check --resume --population 4 --generations 1 --device cpu
```

这只会安排一个新基因进入 FEMM，共六个角度。运行名不可重复，避免覆盖证据。

## 每代的选择

默认种群 96，FEMM 名额是**当代新出现且未验证的唯一基因的 10%**，每代最多 10 个。约一半覆盖预测 Pareto 前沿的两端与中段；约三成从仍有竞争力的候选中按四模型分歧挑选；其余用于结构差异和随机抽查。重复出现、已有 FEMM 真值的基因直接复用，不重复求解。

有真值的基因用 FEMM 真值计算 NSGA-II 的双目标排序；未验证的才暂用主 CNN 预测。父代与子代共同参加非支配排序和拥挤距离选择，保留 96 个。所有已验证基因还形成独立的真值 Pareto 档案，默认最多预留约 10% 的父代席位给其非支配代表，防止乐观预测完全挤掉已验证结构。预测值与真值混排依然不是完全公平的同精度比较；最终性能结论必须基于真实 FEMM。

每次运行的数据集中存放在 `ga_zone/data/<运行名>/`：`run.json` 是配置和模型/物理来源，`state.json` 是可续跑检查点和已验证 Pareto 档案，`seeds.csv` 是初始种子，`candidates.csv` 是所有唯一基因的预测、分歧与真值，`femm_queue.csv` 是逐代 FEMM 选择原因与状态，`femm_labels.csv` 是完整验证的指标，`femm_results/` 存放每个基因的六角度原始 FEMM 文件。所有表由一致的基因哈希和 120 位串关联。选点比例、交叉、位翻转与种群规模都在主程序的 `SearchConfig` 中。

`plots/` 只在第 1、20、40、60、80 代及实际最终代保存快照；若总代数不足，就只保留已达到的节点。横轴是平均转矩、纵轴是绝对转矩波动。灰点是所有代理预测，红点是**尚未做 FEMM 的预测非支配候选**，蓝色空心点是**已完成 FEMM 的真值非支配候选**。图只显示离散样本，不画或暗示连续、已验证的前沿。

CNN 权重在此版保持冻结。配置中预留每 100 代检查一次更新的接口，但**目前不会重训或更换模型**；默认 80 代也不会到达该检查点。磁体格数 12–108 和孤立单格修复是目前搜索约束，不代表机械制造可行性已经验证。

依据：[Deb 等人的 NSGA-II 原论文](https://doi.org/10.1109/4235.996017)；FEMM 工况、六角度转矩定义沿用项目现有求解配置。
