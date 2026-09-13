# 2026-09-09 本机 FEMM 测试

最终版本以 **6 个独立 worker 完成 20 个基因、120 个角度，全部首次尝试成功**，求解批次墙钟耗时 143.17 秒。全量队列尚未启动，剩余 3279 个基因。机器与逐项校验证据见 [machine_validation.json](machine_validation.json)。

## 本次修复和验证

- 新建项目内 `.venv-femm`，Python 3.13.14；依赖版本见 [requirements-femm.txt](requirements-femm.txt)。系统默认 Python 缺少运行依赖，后续应使用此环境。
- 初次实跑一个 worker 的 pyfemm 进入文件通信分支，出现 `NameError: ifile`，达到三个基因失败阈值后自动暂停。该批次有五个成功角度，失败及暂停记录完整保留。
- 修复为先显式导入 Windows COM，创建 `DispatchEx` 独立实例后明确使用 COM 通信。修复版先完成了同一组 20 个基因的 120 个角度。
- 按用户追加要求实现成功后只保留结果和校验记录；以最终版本再次验证同一组 20 个基因。与清理改动前的 120 个原始转矩逐项比较，最大差异为 **0 N·m**。
- 9 项运行回归测试通过，覆盖错误通信分支、结果清理、损坏拒绝、缺失凭据、清理中断恢复及目录边界；另有六进程模拟调度、自检和冻结输入哈希检查通过。
- G2–G5 历史结果经哈希、工况、参考值检查后复用，本次没有重新求解历史四例。
- 独立复算六点均值及峰峰差，与所有 20 个标签一致；角度、数量及基因归属一致。
- 成功角度的 FEM/ANS 和网格工作文件已全部清理，结果和清理凭据哈希通过检查。随后再次执行 `solve --scope pilot --workers 6`，**复用 20 个基因、提交 0 个任务**，验证只保留结果后的续跑。
- 测试结束后无遗留 FEMM/求解器进程，无批次锁。当前工况目录约 1.57 MiB，包含结果、状态、清理凭据及全队列表格。

初次失败批次目录以 `92e8fb2e` 开头，COM 修复验证批次以 `470589a6` 开头；两者的成功角度也已按校验流程清理工作文件，结果 JSON 保持原样，失败工作文件保留。此前迁移来的待求解目录未清理。

## 当前结果

工况指纹：`97cd342fe5d68919dffee97fe5ff2d37bcfdbfd461eb357e9ca92f75ee05c712`。

- [labels.csv](femm_runs/97cd342fe5d68919dffee97fe5ff2d37bcfdbfd461eb357e9ca92f75ee05c712/labels.csv)：3299 行队列记录，其中 20 行成功，3279 行尚未计算，未计算标签为空。
- [waveforms.csv](femm_runs/97cd342fe5d68919dffee97fe5ff2d37bcfdbfd461eb357e9ca92f75ee05c712/waveforms.csv)：120 行原始转矩及角度、电流、文件哈希。
- [report.json](femm_runs/97cd342fe5d68919dffee97fe5ff2d37bcfdbfd461eb357e9ca92f75ee05c712/report.json)：完成数和耗时统计。小批吞吐约 503 基因/小时，对剩余队列估计约 6.5 小时；这是小批实测外推，实际速度会随结构和机器负载变化。

## 手动启动全量

在 PowerShell 中执行下面两行；一个命令会创建六个 worker，无需开六个终端：

```powershell
cd C:\Users\Alexander\Desktop\03_new_spmsm_project
.\.venv-femm\Scripts\python.exe experiments/input_distribution_pilot_v1/pilot.py solve --scope all --workers 6
```

已完成的测试基因会通过校验后复用，其余成功角度自动清理工作文件。运行期间另开终端，从同一项目根目录查看状态：

```powershell
.\.venv-femm\Scripts\python.exe experiments/input_distribution_pilot_v1/pilot.py status
```

全量结束后生成报告：

```powershell
.\.venv-femm\Scripts\python.exe experiments/input_distribution_pilot_v1/pilot.py report
```
