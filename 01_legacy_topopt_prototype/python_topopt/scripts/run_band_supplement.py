from __future__ import annotations

import argparse
import json
import msvcrt
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
from improved_sampling.band_supplement_runner import BandSupplementRunner


class RunProcessLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> "RunProcessLock":
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if self.handle.read(1) == b"":
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError(
                f"该 run 已由另一个启动器运行：{self.path.parent}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self.handle is not None:
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()


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
        description="B2-B7 directed supplement sampling with durable six-angle FEMM"
    )
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new-run", metavar="NAME")
    mode.add_argument("--resume", type=Path, metavar="RUN_DIRECTORY")
    result.add_argument(
        "--source-run",
        type=Path,
        default=ROOT / "runs" / "improved_dataset_1000_20260807_001",
        help="completed 1000-sample source dataset",
    )
    result.add_argument(
        "--physics-config",
        type=Path,
        default=ROOT / "configs" / "dataset_merged_v5_supplement_300.json",
    )
    result.add_argument(
        "--supplement-config",
        type=Path,
        default=ROOT / "configs" / "band_supplement_300.json",
    )
    result.add_argument("--backend", choices=("mock", "femm"))
    result.add_argument("--workers", type=int, default=1, choices=range(1, 7), metavar="1..6")
    result.add_argument("--show-femm-windows", action="store_true")
    result.add_argument("--start", action="store_true")
    result.add_argument("--reference-only", action="store_true")
    result.add_argument("--stop-after-samples", type=int)
    return result


def set_paused(manager: DatasetRunManager) -> None:
    if manager.database.get_run(manager.run_id)["status"] == RunStatus.RUNNING:
        manager.database.set_run_status(manager.run_id, RunStatus.PAUSED)
    manager.write_state()


def execute(args: argparse.Namespace) -> int:
    if args.new_run:
        manager = DatasetRunManager.create(
            project_root=ROOT,
            run_name=args.new_run,
            config_path=args.physics_config,
        )
        runner = BandSupplementRunner.create(
            manager=manager,
            backend=DeterministicAngleMockBackend(),
            source_run=args.source_run,
            config_path=args.supplement_config,
        )
        print(f"已创建补充采样 run：{manager.layout.root}")
    else:
        manager = DatasetRunManager.resume(args.resume)
        runner = None

    if not (args.start or args.reference_only):
        print("只完成初始化，尚未启动 FEMM。使用 --resume ... --backend femm --start 开始。")
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
        runner = BandSupplementRunner.resume(manager=manager, backend=backend)
    else:
        runner.backend = backend
    controller = InterruptController()
    previous = signal.signal(signal.SIGINT, controller.handler)
    started = time.monotonic()
    try:
        if manager.database.get_run(manager.run_id)["status"] != RunStatus.RUNNING:
            manager.database.set_run_status(manager.run_id, RunStatus.RUNNING)
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
                return 2
            print(f"参考模型完成：T_avg_ref={float(manager.database.get_run(manager.run_id)['t_avg_ref']):.12g} N·m")
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
        print("已强制暂停；已提交的角度和候选仍保留在 SQLite。", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGINT, previous)
        backend.close()


def main() -> int:
    args = parser().parse_args()
    if args.new_run and (args.start or args.reference_only):
        raise SystemExit(
            "为确保单启动器锁生效，请先用 --new-run 初始化，再用 --resume ... --start 启动。"
        )
    run_root = ROOT / "runs" / args.new_run if args.new_run else args.resume.resolve()
    if args.start or args.reference_only:
        with RunProcessLock(run_root / "active.lock"):
            return execute(args)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
