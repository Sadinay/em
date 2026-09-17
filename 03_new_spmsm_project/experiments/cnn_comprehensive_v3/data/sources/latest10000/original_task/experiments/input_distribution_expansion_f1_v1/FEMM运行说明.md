# FEMM任务包运行说明

本包是可移动的最小03项目子树。解压到任意目录，在包含 `03_new_spmsm_project` 的目录打开 PowerShell。计算设备需Windows、已安装并注册FEMM、Python 3.12。无需CNN、PyTorch或原模型checkpoint。

```powershell
python -m pip install -r .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\requirements-femm.txt
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py status
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py prepare --scope pilot
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py solve --scope pilot --workers 6
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py status
```

20个检查基因全部成功并人工查看结果后，再执行完整训练/验证队列（9000个，包含这20个，不会重复成功角度）：

```powershell
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py solve --scope all --workers 6
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py resume --scope all --workers 6
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py export --scope all
```

新测试1000个单独求解、结果封存，需显式指定test；不会被默认训练/验证导出读取：

```powershell
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py prepare --scope test
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py solve --scope test --workers 6
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py resume --scope test --workers 6
python .\03_new_spmsm_project\experiments\input_distribution_expansion_f1_v1\femm_entry.py status --include-test-status
```

`prepare`只写输入，`status`只查身份与状态，不打开FEMM；`solve/resume`会启动FEMM。默认6个独立进程，可改workers。每个基因内部六角度依次运行。全部10000基因、60000正常角度任务；20个检查样本计入预算。重试另计，每角度最多2次已记录尝试；连续3个失败或最近50个完成尝试中5个失败则暂停。失败不填零，不用CNN代替。

train/dev结果：`femm_zone/workspaces/f1e1/td/femm_runs/<运行指纹>/`；test结果：`femm_zone/workspaces/f1e1/te/femm_runs/<运行指纹>/`。每个完整基因才产生label.json。成功角度的基因、工况、输入、结果和FEM/ANS哈希全部验证后复用。不要提前删除FEM/ANS，否则恢复完整性检查会拒绝复用；完成接收验证后再按项目的清理流程归档。

Ctrl+C允许正在求解的角度保存后暂停。崩溃留下active.lock时，先确认其中PID已退出，再仅移走该锁文件；不要删角度状态或重置已耗尽次数来绕过失败阈值。达到失败阈值时先调查，不能将异常当正常完成。恢复会校验源码/清单/物理指纹；修改参数不会悄悄复用旧结果。

物理固定：内角29/32/35/38/41/44，3.5A正向电流随机械行程×4更新，初电流角0，MinAngle15，转矩倍率1，Tavg六点均值、DeltaT六点最大减最小。便携物理指纹忽略电脑安装路径但锁定所有物理值和文件字节；更换FEMM二进制版本将形成不同运行指纹，不能误用旧缓存。

本机本轮未启动FEMM。包内不携带旧标签数据集、模型或特征缓存；原始03 MAT为保持物理来源指纹完整随包携带，执行器只读取inp及MaterialPosition（不会读取MAT性能标签）。不要手改清单或用未来结果替换基因。

Windows工作目录采用短名 f1e1/td（训练验证）和f1e1/te（封存测试），避免完整实验名加两层哈希超出FEMM路径长度限制。建议在较短目录解压；入口会检查路径长度。所有.fem输入本身合计约8.4GB，求解生成的.ans另需磁盘空间，可先用20基因检查实际用量。
