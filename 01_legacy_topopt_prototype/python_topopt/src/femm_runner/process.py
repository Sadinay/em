from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Iterable


TRACKED_EXECUTABLES = ("femm.exe", "fkn.exe")


@contextmanager
def femm_startup_lock():
    """Serialize COM server creation so each worker can identify its own FEMM PID."""

    lock_path = Path(tempfile.gettempdir()) / "python_topopt_femm_startup.lock"
    handle = lock_path.open("a+b")
    if os.name != "nt":
        try:
            yield
        finally:
            handle.close()
        return

    import msvcrt

    handle.seek(0)
    if lock_path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    acquired = False
    try:
        while not acquired:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError:
                time.sleep(0.05)
        yield
    finally:
        if acquired:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def _process_ids(image_name: str) -> set[int]:
    completed = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    result: set[int] = set()
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) >= 2 and row[0].lower() == image_name.lower():
            try:
                result.add(int(row[1]))
            except ValueError:
                pass
    return result


def femm_process_snapshot() -> dict[str, set[int]]:
    return {name: _process_ids(name) for name in TRACKED_EXECUTABLES}


def hide_process_windows(process_ids: Iterable[int], *, timeout_seconds: float = 1.5) -> int:
    """Hide all top-level windows owned by the given Windows process IDs."""

    targets = {int(pid) for pid in process_ids}
    if os.name != "nt" or not targets:
        return 0
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    hidden: set[int] = set()

    def hide_once() -> None:
        @callback_type
        def callback(hwnd, _lparam):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if int(pid.value) in targets:
                user32.ShowWindow(hwnd, 0)  # SW_HIDE
                hidden.add(int(pid.value))
            return True

        user32.EnumWindows(callback, 0)

    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        hide_once()
        if targets <= hidden or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    return len(hidden)


def _windows_process_table() -> dict[int, tuple[int, str]]:
    """Return pid -> (parent pid, executable name) without spawning PowerShell."""

    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    TH32CS_SNAPPROCESS = 0x00000002
    MAX_PATH = 260

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * MAX_PATH),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    invalid = wintypes.HANDLE(-1).value
    if snapshot == invalid:
        return {}
    table: dict[int, tuple[int, str]] = {}
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            table[int(entry.th32ProcessID)] = (
                int(entry.th32ParentProcessID),
                str(entry.szExeFile).lower(),
            )
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return table


class OwnedFemmWindowHider:
    """Continuously hide FEMM and solver windows belonging to one worker."""

    def __init__(self, femm_pids: Iterable[int], *, poll_seconds: float = 0.025) -> None:
        self._roots = {int(pid) for pid in femm_pids}
        self._poll_seconds = float(poll_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if os.name != "nt" or not self._roots or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="femm-window-hider",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            table = _windows_process_table()
            owned = set(self._roots)
            changed = True
            while changed:
                changed = False
                for pid, (parent_pid, _name) in table.items():
                    if parent_pid in owned and pid not in owned:
                        owned.add(pid)
                        changed = True
            targets = {
                pid
                for pid in owned
                if table.get(pid, (0, ""))[1] in {"femm.exe", "fkn.exe"}
            }
            hide_process_windows(targets, timeout_seconds=0.0)
            self._stop.wait(self._poll_seconds)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


def _terminate_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False,
    )


def terminate_new_femm_processes(before: dict[str, set[int]]) -> tuple[int, ...]:
    after = femm_process_snapshot()
    new_pids = sorted(
        pid
        for name in TRACKED_EXECUTABLES
        for pid in after[name] - before.get(name, set())
    )
    for pid in reversed(new_pids):
        _terminate_process_tree(pid)
    return tuple(new_pids)


def terminate_recorded_femm_processes(manifest_path: Path) -> tuple[int, ...]:
    """Terminate only FEMM PIDs explicitly recorded by this worker."""

    path = Path(manifest_path)
    if not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded = {int(pid) for pid in payload.get("femm_pids", ())}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ()
    active = set().union(*femm_process_snapshot().values())
    owned_active = sorted(recorded & active)
    for pid in reversed(owned_active):
        _terminate_process_tree(pid)
    return tuple(owned_active)


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    worker_pid: int
    return_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    terminated_pids: tuple[int, ...]


def run_isolated_worker(
    job_path: Path,
    *,
    timeout_seconds: float,
    require_clean_process_state: bool = True,
    process_manifest_path: Path | None = None,
) -> WorkerOutcome:
    before = femm_process_snapshot()
    existing = sorted(pid for values in before.values() for pid in values)
    if require_clean_process_state and existing:
        raise RuntimeError(f"FEMM/fkern processes already exist: {existing}")
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_path = str(project_root / "src")
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    process = subprocess.Popen(
        [sys.executable, "-m", "femm_runner.worker", "--job", str(job_path)],
        cwd=project_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        ),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        terminated = (
            terminate_recorded_femm_processes(process_manifest_path)
            if process_manifest_path is not None
            else terminate_new_femm_processes(before)
        )
        return WorkerOutcome(
            worker_pid=process.pid,
            return_code=process.returncode,
            timed_out=False,
            stdout=stdout,
            stderr=stderr,
            terminated_pids=terminated,
        )
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process.pid)
        stdout, stderr = process.communicate()
        terminated = (
            terminate_recorded_femm_processes(process_manifest_path)
            if process_manifest_path is not None
            else terminate_new_femm_processes(before)
        )
        return WorkerOutcome(
            worker_pid=process.pid,
            return_code=None,
            timed_out=True,
            stdout=stdout,
            stderr=stderr,
            terminated_pids=terminated,
        )
    except BaseException:
        # A second Ctrl+C or another forced termination must never orphan the
        # worker/FEMM process tree owned by this call.
        _terminate_process_tree(process.pid)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process.pid)
        if process_manifest_path is not None:
            terminate_recorded_femm_processes(process_manifest_path)
        else:
            terminate_new_femm_processes(before)
        raise
