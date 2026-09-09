# 数据区

`raw/` 保存只读的 `workspace_200.mat`；解析数据进入 `processed/`，可交换文件进入 `exports/`，通过审计的物理记录进入 03 独立的 `database/`。不得读写 01 或 02 的数据库。
