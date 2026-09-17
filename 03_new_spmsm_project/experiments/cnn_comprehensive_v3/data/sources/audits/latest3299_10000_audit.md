# 最新 3299 / 10000 数据身份与标签审计

审计状态：**passed**。范围仅包含 FEMM_results_20260913（3299）与 FEMM_10000_results_20260917（train_dev 9000 + sealed_test 1000）。

| 检查 | 结果 | 关键结果 |
|---|---|---|
| 批次规模与 split | passed | 3299；9000 + 1000 |
| 身份、bit 编码、修正和面积 | passed | old/new 均无异常 |
| 批内重复 | passed | gene_id、bits、raw_bits、raw_gene_id 均 0 |
| 标签表一致性 | passed | 三个标签表逐字段一致 |
| 标签定义、来源和分组 | passed | source/group 配额与成功状态一致 |
| 6 角度完整性与 Tavg/DeltaT | passed | 19794 + 54000 + 6000 波形行 |
| 3299 memberships | passed | 8000 对 group membership 一致 |
| 10000 队列身份 | passed | root/train_dev/sealed_test 一致 |
| train_dev 子表 | passed | train 8000、dev 1000 |
| 批间交集 | passed | 修正 bit/raw bit/gene_id 交集均 0 |
| 工况与物理指纹 | passed | canonical digest b3676fdbfd636c5eeb8f5b692675d3bbafefbc063b0387b7203309d61d447d0a |

## 工况定义

两批都使用 3.5 A、4 极对、静磁 Frequency=0、MinAngle=15、Depth=36 mm、倍率 1；内角为 29/32/35/38/41/44°，机械行程为 0/3/6/9/12/15°。三相电流与合同值逐行核对通过；Tavg 为六点均值，DeltaT 为六点最大值减最小值。

旧批行级 fingerprint 为 97cd342fe5d68919dffee97fe5ff2d37bcfdbfd461eb357e9ca92f75ee05c712，新批为 a5146b3c9c71ef337943ce10700903bb927c5d327564fa0f8ce35417a42700a4。完整合同摘要因主机/运行器身份不同而不同；去除这些身份后 canonical digest 为 b3676fdbfd636c5eeb8f5b692675d3bbafefbc063b0387b7203309d61d447d0a。

## 可复现命令

python -X utf8 audit_latest_data.py
python -X utf8 ..\..\..\..\femm_zone\results\FEMM_10000_results_20260917\verify_package.py
python -X utf8 ..\..\..\..\femm_zone\results\FEMM_results_20260913\verify_package.py
