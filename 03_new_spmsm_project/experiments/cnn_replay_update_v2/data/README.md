# 训练数据与回放索引

本目录按模型保存 audit：数据清单核验、旧/新回放索引、预检和运行记录。它们是本轮实际使用的数据审计和抽样材料。

三轮共享的原始数据只保留一份，继续从原位置读取，未重新划分，也没有复制测试标签：

- 旧数据：[统一数据目录](../../../data_zone/processed/spmsm_topology_dataset/training_corrected_binary/)
- 旧训练/验证/测试划分：[固定划分](../../../cnn_zone/outputs/splits/scheme_a_tavg_bands_train40000_val6483_test6483.npz)
- 新训练 G：[已验收数据](../../input_distribution_pilot_v1/post_femm_baseline_20260913/data/train_G/)
- 新训练 F：[已验收数据](../../input_distribution_pilot_v1/post_femm_baseline_20260913/data/train_F/)
- 新验证：[已验收数据](../../input_distribution_pilot_v1/post_femm_baseline_20260913/data/dev_common/)

layout_migration.json 只记录本次目录迁移及源码对应关系，不是训练参数。刚导入的一万个新基因属于后续扩样，未混入这三轮历史实验。
