from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_generation.backend import DeterministicAngleMockBackend
from dataset_generation.femm_backend import IsolatedFemmAngleBackend
from dataset_generation.orchestrator import DatasetRunManager
from dataset_generation.state import RunStatus
from improved_sampling.campaign_runner import ImprovedCampaignRunner


class InterruptController:
    def __init__(self) -> None:
        self.count = 0

    def handler(self, signum: int, frame: object) -> None:
        del signum, frame
        self.count += 1
        if self.count == 1:
            print("\n收到 Ctrl+C：当前 FEMM 角度结束后保存并暂停。", flush=True)
        else:
            print("\n再次收到 Ctrl+C：强制终止当前 worker。", flush=True)
            raise KeyboardInterrupt

    def requested(self) -> bool:
        return self.count > 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Interruptible multi-parent improved sampling campaigns with six-angle FEMM"
    )
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new-run", metavar="NAME")
    mode.add_argument("--resume", type=Path, metavar="RUN_DIRECTORY")
    result.add_argument("--parent-run", type=Path, help="completed 60-parent improved run")
    result.add_argument(
        "--physics-config", type=Path, default=ROOT / "configs" / "dataset_merged_v5.json"
    )
    result.add_argument(
        "--campaign-config", type=Path,
        default=ROOT / "configs" / "improved_campaign_evolution.json",
    )
    result.add_argument("--backend", choices=("mock", "femm"))
    result.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=range(1, 7),
        metavar="1..6",
        help="parallel candidate evaluations; each worker owns an isolated FEMM instance",
    )
    result.add_argument(
        "--show-femm-windows",
        action="store_true",
        help="debug only: show FEMM windows instead of hiding them",
    )
    result.add_argument("--start", action="store_true", help="finish reference if needed and run campaigns")
    result.add_argument("--reference-only", action="store_true")
    result.add_argument(
        "--stop-after-samples", type=int,
        help="pilot pause after this many unique candidate samples; immutable target remains 1000",
    )
    return result


def set_running(manager: DatasetRunManager) -> None:
    if manager.database.get_run(manager.run_id)["status"] != RunStatus.RUNNING:
        manager.database.set_run_status(manager.run_id, RunStatus.RUNNING)


def set_paused(manager: DatasetRunManager) -> None:
    if manager.database.get_run(manager.run_id)["status"] == RunStatus.RUNNING:
        manager.database.set_run_status(manager.run_id, RunStatus.PAUSED)
    manager.write_state()


def main() -> int:
    args = parser().parse_args()
    if args.new_run:
        if args.parent_run is None:
            raise SystemExit("--new-run requires --parent-run")
        manager = DatasetRunManager.create(
            project_root=ROOT, run_name=args.new_run, config_path=args.physics_config
        )
        creation_backend = DeterministicAngleMockBackend()
        runner = ImprovedCampaignRunner.create(
            manager=manager,
            backend=creation_backend,
            parent_run=args.parent_run,
            config_path=args.campaign_config,
        )
        print(f"已创建 improved campaign run：{manager.layout.root}")
    else:
        manager = DatasetRunManager.resume(args.resume)
        runner = None

    if not (args.start or args.reference_only):
        print("只完成初始化；未启动 FEMM。使用 --resume ... --backend femm --start 开始。")
        return 0
    if args.backend is None:
        raise SystemExit("--start 或 --reference-only 必须指定 --backend mock|femm")
    backend = (
        DeterministicAngleMockBackend(max_workers=args.workers)
        if args.backend == "mock"
        else IsolatedFemmAngleBackend(
            manager.config,
            max_workers=args.workers,
            hide_windows=not args.show_femm_windows,
        )
    )
    if runner is None:
        runner = ImprovedCampaignRunner.resume(manager=manager, backend=backend)
    else:
        runner.backend = backend
    controller = InterruptController()
    previous = signal.signal(signal.SIGINT, controller.handler)
    started = time.monotonic()
    try:
        set_running(manager)
        run = manager.database.get_run(manager.run_id)
        if run["t_avg_ref"] is None:
            reference_id = manager.create_reference_sample()
            status = manager.evaluate_sample(
                reference_id,
                backend=backend,
                chromosome=None,
                stop_requested=controller.requested,
                commit_callback=runner.save_checkpoint,
            )
            if status != "completed":
                set_paused(manager)
                runner.write_campaign_status()
                print(f"参考模型状态：{status}；再次执行恢复命令即可续算。")
                return 2
            run = manager.database.get_run(manager.run_id)
            print(f"参考六角度完成：T_avg_ref={float(run['t_avg_ref']):.12g} N·m")
        if args.reference_only:
            set_paused(manager)
            runner.write_campaign_status()
            return 0

        status = runner.run(
            stop_requested=controller.requested,
            stop_after_samples=args.stop_after_samples,
        )
        if status == "target_reached":
            manager.database.set_run_status(manager.run_id, RunStatus.COMPLETED)
        else:
            set_paused(manager)
        summary = manager.write_state()
        runner.write_campaign_status()
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"本次运行耗时：{time.monotonic() - started:.1f} s；状态：{status}")
        return 0 if status == "target_reached" else 2
    except KeyboardInterrupt:
        set_paused(manager)
        runner.save_checkpoint()
        runner.write_campaign_status()
        print("已强制暂停；已提交角度和候选均保留在 SQLite。", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGINT, previous)
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
