from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from improved_sampling.generator import ParentArchiveGenerator


class Interrupts:
    def __init__(self) -> None:
        self.count = 0

    def handler(self, signum: int, frame: object) -> None:
        del signum, frame
        self.count += 1
        if self.count == 1:
            print("\n收到 Ctrl+C：完成当前纯拓扑proposal后保存并暂停。", flush=True)
        else:
            # A topology proposal takes milliseconds.  Deferring even the
            # second signal to the next proposal boundary avoids a DB/checkpoint
            # split-brain window while remaining effectively immediate.
            print("\n已再次请求停止：将在当前proposal事务完成后退出。", flush=True)

    def requested(self) -> bool:
        return self.count > 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Interruptible improved parent-archive sampler")
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new-run", metavar="NAME")
    mode.add_argument("--resume", type=Path, metavar="RUN_DIRECTORY")
    result.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "improved_dataset_sampling.json",
    )
    result.add_argument(
        "--stop-after-parents",
        type=int,
        help="pilot stop; the immutable archive target remains 60",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    generator = (
        ParentArchiveGenerator.create(
            project_root=ROOT,
            run_name=args.new_run,
            config_path=args.config,
        )
        if args.new_run
        else ParentArchiveGenerator.resume(args.resume)
    )
    if args.stop_after_parents is not None:
        if not 1 <= args.stop_after_parents <= generator.config.parent_target:
            raise SystemExit("--stop-after-parents must be between 1 and parent_archive_target")
        if args.stop_after_parents < len(generator.state["parent_ids"]):
            raise SystemExit("requested stop is below the already committed parent count")
    interrupts = Interrupts()
    previous = signal.signal(signal.SIGINT, interrupts.handler)
    try:
        status = generator.run(
            stop_after_parents=args.stop_after_parents,
            stop_requested=interrupts.requested,
            progress=lambda summary: print(
                f"proposal={summary['proposal_count']} legal_unique={summary['legal_unique_proposals']} "
                f"parents={summary['parent_count']}/{generator.config.parent_target}",
                flush=True,
            ),
        )
        print(json.dumps(generator.database.summary(), ensure_ascii=False, indent=2, sort_keys=True))
        print(f"状态：{status}；目录：{generator.layout.root}")
        return 0 if status in {"completed", "pilot_stop"} else 2
    except KeyboardInterrupt:
        generator.database.set_status("forced_interrupted")
        generator.save_checkpoint()
        generator.write_exports()
        print("已强制停止；上一个已提交proposal及RNG状态可恢复。", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    raise SystemExit(main())
