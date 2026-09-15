# 固定SmallCNN测试评估

只评估原版冻结f0和预先选定的F-S、α=.80、β=.01、12.5%新样本、9500步模型。两者使用同一400个新测试样本及6483个旧测试样本。模型状态在推理前后完全不变，无反向传播或优化器更新。

- `evaluation_plan.json`：读测试标签前固定的模型、哈希和评估方案。
- `evaluate.py`：复用原输入编码及推理流程；测试推理单独入口，未放宽训练代码的数据隔离。
- `data_audit.json`：测试身份、版本和隔离核验。
- `predictions/`：两模型的新旧测试预测。
- `metrics.json`、`inference_verification.json`：完整指标和模型不变性核验。
- [报告与图](report/REPORT.md)：并列比较原f0与SP模型。

已经完成推理。只重绘报告：

```powershell
python ./03_new_spmsm_project/experiments/cnn_test_smallcnn_sp080_v1/evaluate.py --report-only
```

默认入口拒绝覆盖已完成的测试结果。没有使用这些测试结果重新训练或选择模型。400新测试集从本次开始已用于观察效果，今后需要保留这一使用记录。
