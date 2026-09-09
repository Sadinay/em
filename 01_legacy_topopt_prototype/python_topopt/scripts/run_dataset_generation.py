from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_generation.backend import DeterministicAngleMockBackend
from dataset_generation.clonalg_runner import ClonalgDatasetRunner
from dataset_generation.femm_backend import IsolatedFemmAngleBackend
from dataset_generation.orchestrator import DatasetRunManager
from dataset_generation.state import RunStatus


class InterruptController:
    def __init__(self) -> None:
        self.count = 0

    def handler(self, signum: int, frame: object) -> None:
        del signum, frame
        self.count += 1
        if self.count == 1:
            print("\n收到第一次 Ctrl+C：当前角度完成后安全暂停；再按一次将强制终止当前 worker。")
            return
        print("\n收到第二次 Ctrl+C：强制终止当前任务。")
        raise KeyboardInterrupt

    def stop_requested(self) -> bool:
        return self.count > 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Durable single-process CLONALG + FEMM dataset runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new-run", metavar="NAME")
    mode.add_argument("--resume", type=Path, metavar="RUN_DIRECTORY")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dataset_merged_v5.json")
    parser.add_argument("--geometry-mode", choices=("historical_inset", "merged_copper_v5"))
    parser.add_argument("--target-valid-samples", type=int)
    parser.add_argument("--backend", choices=("mock", "femm"))
    parser.add_argument("--run-reference", action="store_true", help="evaluate/finish the six-angle reference first")
    parser.add_argument("--reference-only", action="store_true", help="stop after the reference is committed")
    parser.add_argument("--start", action="store_true", help="start or continue CLONALG after reference")
    parser.add_argument("--generations", type=int, help="small validation generation limit")
    return parser


def _backend(name: str, manager: DatasetRunManager):
    if name == "mock":
        return DeterministicAngleMockBackend()
    if name == "femm":
        return IsolatedFemmAngleBackend(manager.config)
    raise ValueError("--backend is required when evaluating a reference or starting CLONALG")


def _set_running(manager: DatasetRunManager) -> None:
    status = str(manager.database.get_run(manager.run_id)["status"])
    if status != RunStatus.RUNNING:
        manager.database.set_run_status(manager.run_id, RunStatus.RUNNING)


def _set_paused(manager: DatasetRunManager) -> None:
    status = str(manager.database.get_run(manager.run_id)["status"])
    if status == RunStatus.RUNNING:
        manager.database.set_run_status(manager.run_id, RunStatus.PAUSED)
    manager.write_state()


def _print_progress(state: dict) -> None:
    sample = state.get("current_sample_id") or "-"
    angle = state.get("current_angle_deg")
    angles = state.get("angle_counts", {})
    bands = state.get("fitness_bands", {})
    completed = int(angles.get("completed", 0))
    print(
        f"[progress] gen={state.get('generation')} sample={sample} angle={angle} "
        f"angles_done={completed} valid_unique={state.get('valid_unique_samples', 0)} "
        f"bands={bands}",
        flush=True,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.new_run:
        manager = DatasetRunManager.create(
            project_root=PROJECT_ROOT,
            run_name=args.new_run,
            config_path=args.config,
        )
        print(f"已创建 run：{manager.layout.root}")
    else:
        manager = DatasetRunManager.resume(args.resume)
        print(f"已恢复 run：{manager.layout.root}")

    if args.geometry_mode and args.geometry_mode != manager.config.geometry_mode:
        raise SystemExit("命令行 geometry mode 与不可变 run 配置不同；请创建新 run")
    if (
        args.target_valid_samples is not None
        and args.target_valid_samples != manager.config.target_valid_samples
    ):
        raise SystemExit("命令行目标数与不可变 run 配置不同；请修改配置后创建新 run")
    if not (args.run_reference or args.reference_only or args.start):
        run = manager.database.get_run(manager.run_id)
        print(
            "任务已初始化；参考六角度尚未完成。"
            if run["t_avg_ref"] is None
            else "任务已初始化，参考转矩已经提交。"
        )
        return 0
    backend = _backend(args.backend, manager)
    manager.progress_callback = _print_progress
    controller = InterruptController()
    previous = signal.signal(signal.SIGINT, controller.handler)
    started = time.monotonic()
    try:
        _set_running(manager)
        run = manager.database.get_run(manager.run_id)
        if run["t_avg_ref"] is None:
            if not (args.run_reference or args.reference_only):
                raise RuntimeError("必须先使用 --run-reference 完成 strukturFemm 六角度参考计算")
            reference_id = manager.create_reference_sample()
            result = manager.evaluate_sample(
                reference_id,
                backend=backend,
                chromosome=None,
                stop_requested=controller.stop_requested,
                commit_callback=lambda: manager.write_checkpoint(
                    {
                        "kind": "reference_evaluation",
                        "generation": 0,
                        "sample_id": reference_id,
                    }
                ),
            )
            if result != "completed":
                _set_paused(manager)
                print(f"参考计算状态：{result}；可使用 --resume 继续。")
                return 2
            run = manager.database.get_run(manager.run_id)
            print(f"参考计算完成：T_avg_ref={float(run['t_avg_ref']):.12g} N·m")
        if args.reference_only or not args.start:
            _set_paused(manager)
            return 0

        runner = (
            ClonalgDatasetRunner.resume(manager=manager, backend=backend)
            if args.resume and manager.layout.latest_checkpoint.is_file()
            and json.loads(manager.layout.latest_checkpoint.read_text(encoding="utf-8"))
            .get("payload", {}).get("algorithm_state", {}).get("schema_version")
            else ClonalgDatasetRunner(
                manager=manager,
                backend=backend,
                generation_limit=args.generations,
            )
        )
        status = runner.run(stop_requested=controller.stop_requested)
        if status in {"completed", "target_reached"}:
            # An explicit short generation limit is a validation stop, not
            # completion of the immutable dataset target.
            if args.generations is not None and status != "target_reached":
                _set_paused(manager)
            else:
                manager.database.set_run_status(manager.run_id, RunStatus.COMPLETED)
        else:
            _set_paused(manager)
        summary = manager.write_state()
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"本次运行耗时：{time.monotonic() - started:.1f} s；状态：{status}")
        return 0 if status == "completed" else 2
    except KeyboardInterrupt:
        _set_paused(manager)
        print("任务已强制终止；已完成角度仍保存在 SQLite 中。", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGINT, previous)
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
