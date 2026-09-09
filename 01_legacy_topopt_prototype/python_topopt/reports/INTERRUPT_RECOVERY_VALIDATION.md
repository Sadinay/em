# 中断恢复验证

## Mock 验证

自动化测试覆盖样本间停止、角度级停止、running 角度恢复为 interrupted、超时后有限重试、损坏 checkpoint 回退、连续运行与多次恢复的种群/RNG/候选顺序一致性。全套 67 项测试通过。

## 真实 FEMM 角度级验证

Run：`runs/reference_validation_20260806_001`，候选 `C000004` / 样本 `S000005`。

第一次 Python 进程：

- 计算并事务提交 0°、3°、6°；
- 9°、12°、15° 保持 pending；
- 进程以 paused 状态退出。

第二次独立 Python 进程：

- 从 SQLite 恢复同一物理样本；
- 只计算 9°、12°、15°；
- 前三个角度没有重新求解；
- 最终六个角度的 `attempt` 全部为 1；
- 得到 `torque_ratio=0.987074779316`、`historical_J=0.102971846230`。

恢复后 `PRAGMA integrity_check=ok`，任务校验无错误，FEMM/fkn 无残留。

两级 Ctrl+C 行为已经实现：第一次只设置安全暂停请求，当前角度完成后停；第二次抛出强制中断，并只终止当前 worker 拥有的进程树。第二次 Ctrl+C 的真实人工按键尚未做破坏性实机演练，自动化测试覆盖了等价清理路径。
