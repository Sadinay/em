"""Build the requested handoff folder from the completed, verified current run."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import shutil
import zipfile

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
DEST = PROJECT / "FEMM_results_20260913"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def copy(source, relative):
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest(), relative


def write_rows(path, data):
    fields = list(dict.fromkeys(k for row in data for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def main():
    entry = read(BASE / "femm_entry.json")
    run = BASE / entry["run_directory"]
    assert run.resolve().is_relative_to((BASE / "femm_runs").resolve())
    assert not (run / "active.lock").exists(), "Current run is locked"
    progress, report = read(run / "progress.json"), read(run / "report.json")
    assert progress["status"] == "completed" and report["complete_genes"] == 3299 and report["incomplete_genes"] == 0
    assert not DEST.exists(), "Export destination exists; do not overwrite it"
    DEST.mkdir()
    (DEST / "tables").mkdir()
    for name in ("labels.csv", "waveforms.csv", "report.json"):
        copy(run / name, Path("tables") / name)
    labels = {r["gene_id"]: r for r in rows(run / "labels.csv")}
    assert len(labels) == 3299 and all(r["status"] == "succeeded" for r in labels.values())
    for source_name, output_name in (("femm_queue", "dataset_all"), ("train_G", "train_G_labeled"),
                                     ("train_F", "train_F_labeled"), ("dev_common", "dev_common_labeled"),
                                     ("test_common", "test_common_labeled")):
        # Preserve each split's own selection metadata; labels supply current solve status and targets.
        merged = [{**row, **labels[row["gene_id"]]} for row in rows(BASE / f"{source_name}.csv")]
        write_rows(DEST / "tables" / f"{output_name}.csv", merged)
    for path in BASE.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            copy(path, Path("manifests") / path.name)
        elif path.suffix in (".json", ".md"):
            copy(path, Path("audit") / path.name)
        elif path.suffix == ".py" or path.name == "requirements-femm.txt":
            copy(path, Path("source_snapshot") / path.relative_to(PROJECT))
    for relative in ("femm_zone/femm_config.py", "femm_zone/scripts/spmsm_mapping.py", "femm_zone/models/SPMSM_discrete.fem",
                     "data_zone/raw/workspace_200.mat", "data_zone/SPMSM_Input_Distribution_Pilot_4Groups.md",
                     "cnn_zone/src/fem_mesh.py", "cnn_zone/src/training.py", "cnn_zone/src/training_inputs.py", "cnn_zone/src/models_v2.py"):
        copy(PROJECT / relative, Path("source_snapshot") / relative)
    for path in (BASE / "region_previews").rglob("*"):
        if path.is_file():
            copy(path, Path("region_previews") / path.relative_to(BASE / "region_previews"))
    copy(Path(__file__).with_name("verify_package.py"), Path("verify_package.py"))
    copy(Path(__file__), Path("source_snapshot/export_completed_20260913.py"))
    files = sorted(path for path in run.rglob("*") if path.is_file())
    assert not any(path.name == "active.lock" or path.suffix in (".fem", ".ans") for path in files)
    print(f"Archiving {len(files)} final run files...", flush=True)
    record_hashes = {}
    with zipfile.ZipFile(DEST / "full_run_records.zip", "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, path in enumerate(files, 1):
            data, name = path.read_bytes(), path.relative_to(run).as_posix()
            archive.writestr(name, data)
            record_hashes[name] = hashlib.sha256(data).hexdigest()
            if index % 10000 == 0:
                print(f"Archived {index}/{len(files)}", flush=True)
        archive.writestr("RUN_CHECKSUMS.json", json.dumps(record_hashes, indent=2))
    assert not (run / "active.lock").exists() and read(run / "progress.json") == progress, "Source execution changed during export"
    summary = {"exported_local": datetime.datetime.now().astimezone().isoformat(), "status": "completed_results_export",
               "condition_fingerprint": entry["condition_fingerprint"], "source_run_directory": str(run),
               "complete_genes": 3299, "successful_angles": 19794, "incomplete_genes": 0,
               "failed_gene_attempts": report["failed_gene_attempts"], "final_workers": progress["workers"],
               "archived_source_files": len(files), "final_report": report,
               "result_policy": "Results and hash-linked receipts retained; successful FEM/ANS work files were removed after validation.",
               "historical_audit_note": "audit/machine_validation and io_recovery describe earlier pilot/stoppage snapshots; tables/report.json is the final report.",
               "original_files_copied_without_move_or_delete": True}
    write_json(DEST / "package_summary.json", summary)
    text = """# FEMM 全量结果交付（2026-09-13）

3299 个基因、19794 个角度全部完成，最终失败基因数为 0。当前状态以 `package_summary.json` 和 `tables/report.json` 为准。

## 主机验证与使用

- `tables/dataset_all.csv`：全部 3299 个基因的 120 位编码、来源/分组、磁体信息、Tavg、DeltaT 和工况指纹，适合直接核验或导入。
- `tables/train_G_labeled.csv`、`train_F_labeled.csv`：各 1400 行；`dev_common_labeled.csv` 200 行、`test_common_labeled.csv` 400 行。G/F 训练集有 101 个共同基因；四组映射见 `manifests/memberships.csv`。
- `tables/labels.csv`：原始标签汇总；`tables/waveforms.csv`：19794 行原始转矩、角度、电流和 FEM/ANS 哈希。
- `full_run_records.zip`：当前最终工况目录的全部原始文件，含每基因 `label.json`、每角度 `result.json` / `state.json` / `artifact_retention.json`、`contract.json`、运行日志和中断后续跑记录。解压后层级与原工况目录相同。
- `manifests/`：原始冻结清单与候选/选样资料。这里 `femm_queue.csv` 的 pending 是选样时的历史登记值；实际完成状态请读 `tables/`。
- `audit/`：选样、试跑、回归和停机恢复资料；其中 20 个测试结果、1646 个完成结果等数字是历史快照。
- `source_snapshot/`：导出时的运行代码、物理配置、映射、原始 MAT 和 FEM 模板，以及方案文档。CNN 权重和大型特征缓存不在此交付包中。
- `region_previews/`：原始选样区域预览。

标签单位均为 N·m。实际内角为 29、32、35、38、41、44°；Tavg 是六点均值，DeltaT 是六点最大值减最小值。

## 拷贝完整性检查

把整个文件夹复制到主机后，在本文件夹打开终端运行：

```powershell
python verify_package.py
```

只需 Python 3.9+ 标准库，不启动 FEMM，也不需要 NumPy、torch 或解压原始记录。程序检查所有交付文件和压缩包内每份原始记录的 SHA256、基因编码、分组、六角度完整性、Tavg/DeltaT，以及结果与清理凭据的相互关联。最后应输出 `status: passed`、`genes: 3299`、`angles: 19794`。

`checksums.json` 覆盖除其自身外的全部交付文件；归档内部 `RUN_CHECKSUMS.json` 覆盖每份原始文件。成功的 FEM/ANS 工作文件已按要求清理，保留的输入模板、配置与哈希记录用于追溯；本包不会把数值一致性检查宣称为重新运行 FEMM。
"""
    (DEST / "README.md").write_text(text, encoding="utf-8")
    checksums = {path.relative_to(DEST).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(DEST.rglob("*")) if path.is_file()}
    write_json(DEST / "checksums.json", checksums)
    print(json.dumps({"folder": str(DEST), "bytes": sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()),
                      "copied_files": len(checksums), "archived_files": len(files)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
