"""Storage failure/recovery and Windows transport regression checks; no FEMM launch."""
from pathlib import Path
import tempfile
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import pilot as p


class CommandDiagnosticsTests(unittest.TestCase):
    def test_full_traceback_and_filename_are_saved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            error = PermissionError(13, "access denied", str(root / "progress.json"))
            with patch.object(p, "HERE", root), patch.object(p, "solve_queue", side_effect=error), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(PermissionError):
                    p.solve_from_cli("all", 6)
            files = list((root / "failure_diagnostics").glob("*.json"))
            self.assertEqual(len(files), 1)
            saved = p.read(files[0])
            self.assertEqual(saved["filename"], str(root / "progress.json"))
            self.assertEqual(saved["errno"], 13)
            self.assertIn("Traceback", saved["traceback"])
            self.assertIn("solve_from_cli", saved["traceback"])

    def test_preflight_interrupt_is_reported_as_interruption(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            out, err = io.StringIO(), io.StringIO()
            with patch.object(p, "HERE", root), patch.object(p, "solve_queue", side_effect=KeyboardInterrupt), redirect_stdout(out), redirect_stderr(err):
                with self.assertRaises(SystemExit) as stopped:
                    p.solve_from_cli("all", 6)
            self.assertEqual(stopped.exception.code, 130)
            self.assertIn("数分钟", out.getvalue())
            self.assertIn("中断信号", err.getvalue())
            self.assertFalse((root / "failure_diagnostics").exists())


class FileAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_json_replace_retries_without_touching_previous_result(self):
        target = self.folder / "progress.json"
        p.save(target, {"complete": 10})
        original_replace = Path.replace
        calls = []
        def locked(source, destination):
            calls.append(destination)
            if len(calls) <= 2:
                self.assertEqual(p.read(target), {"complete": 10})
                raise PermissionError(13, "simulated Windows reader lock")
            return original_replace(source, destination)
        with patch.object(Path, "replace", locked), patch.object(p.time, "sleep"):
            p.save(target, {"complete": 11})
        self.assertEqual(len(calls), 3)
        self.assertEqual(p.read(target), {"complete": 11})

    def test_temporary_file_write_is_also_retried(self):
        target = self.folder / "progress.json"
        original_write = Path.write_text
        calls = []
        def locked(path, content, *args, **kwargs):
            calls.append(path)
            if len(calls) == 1:
                raise PermissionError(13, "temporary file locked")
            return original_write(path, content, *args, **kwargs)
        with patch.object(Path, "write_text", locked), patch.object(p.time, "sleep"):
            p.save(target, {"complete": 11})
        self.assertEqual(len(calls), 2)
        self.assertEqual(p.read(target), {"complete": 11})

    def test_csv_replace_retry_preserves_encoding_and_rows(self):
        target = self.folder / "labels.csv"
        original_replace = Path.replace
        calls = []
        def locked(source, destination):
            calls.append(destination)
            if len(calls) == 1:
                raise PermissionError(13, "CSV temporarily locked")
            return original_replace(source, destination)
        rows = [{"gene": "test", "note": "中文, quoted", "torque": "1.25"}]
        with patch.object(Path, "replace", locked), patch.object(p.time, "sleep"):
            p.csv_write(target, rows)
        self.assertEqual(p.csv_read(target), rows)
        self.assertTrue(target.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_persistent_denial_stops_and_preserves_previous_result(self):
        target = self.folder / "progress.json"
        p.save(target, {"complete": 10})
        with patch.object(Path, "replace", side_effect=PermissionError(13, "locked")) as replace, patch.object(p.time, "sleep"):
            with self.assertRaises(PermissionError) as raised:
                p.save(target, {"complete": 11})
        self.assertEqual(replace.call_count, 12)
        self.assertIn(str(target), raised.exception.args[1])
        self.assertEqual(p.read(target), {"complete": 10})
        self.assertEqual(p.read(target.with_suffix(".json.tmp")), {"complete": 11})

    def test_other_io_errors_are_not_retried(self):
        with patch.object(Path, "replace", side_effect=OSError(28, "disk full")) as replace, patch.object(p.time, "sleep") as sleep:
            with self.assertRaises(OSError):
                p.save(self.folder / "progress.json", {"complete": 11})
        self.assertEqual(replace.call_count, 1)
        sleep.assert_not_called()

    @unittest.skipUnless(p.os.name == "nt", "Windows file sharing semantics")
    def test_real_windows_reader_lock_recovers(self):
        import threading
        import win32file
        import win32con
        target = self.folder / "progress.json"
        p.save(target, {"complete": 10})
        handle = win32file.CreateFile(str(target), win32con.GENERIC_READ,
                                      win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
                                      None, win32con.OPEN_EXISTING, 0, None)
        release = threading.Timer(0.25, handle.Close)
        release.start()
        try:
            p.save(target, {"complete": 11})
        finally:
            release.join()
            handle.Close()
        self.assertEqual(p.read(target), {"complete": 11})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.here = Path(self.temp.name)
        self.patch = patch.object(p, "HERE", self.here)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        cfg = p.physical_config()
        self.contract = {"physical": cfg}
        bits = "0" * 120
        gene = p.mapping.genotype_sha256([0] * 120)
        self.folder = self.here / "femm_runs" / p.digest(self.contract) / gene / "angle_29"
        self.folder.mkdir(parents=True)
        p.save(self.folder.parents[1] / "contract.json", self.contract)
        template = p.physical.TEMPLATE_FILE.read_text(encoding="utf8")
        model = p.physical.configure_fem(p.mapping.replace_cell_materials(template, [0] * 120), cfg, 29, 0)
        (self.folder / "model.fem").write_bytes(model.encode("utf8"))
        (self.folder / "model.ans").write_text("synthetic answer, never a physical solve", encoding="utf8")
        self.identity = {"gene_id": gene, "bits": bits, "inner_angle_deg": 29,
                         "rotor_travel_deg": 0, "currents_a": p.physical.phase_currents(cfg, 29, 0),
                         "condition_fingerprint": p.digest(self.contract),
                         "prepared_sha256": p.sha(self.folder / "model.fem")}
        self.result = {**self.identity, "raw_torque_nm": 2.5,
                       "fem_sha256": p.sha(self.folder / "model.fem"),
                       "ans_sha256": p.sha(self.folder / "model.ans")}
        p.save(self.folder / "result.json", self.result)
        p.save(self.folder / "state.json", {**self.identity, "status": "succeeded", "attempts": [],
                                           "result_sha256": p.sha(self.folder / "result.json")})

    def test_pruned_results_remain_valid_and_unrelated_files_survive(self):
        (self.folder / "notes.txt").write_text("keep")
        p.prune_success(self.folder, self.identity)
        self.assertFalse((self.folder / "model.fem").exists())
        self.assertFalse((self.folder / "model.ans").exists())
        self.assertTrue((self.folder / "notes.txt").exists())
        self.assertEqual(p.validate_success(self.folder, self.identity), self.result)
        p.prune_success(self.folder, self.identity)

    def test_missing_model_without_receipt_is_rejected(self):
        (self.folder / "model.ans").unlink()
        with self.assertRaisesRegex(ValueError, "missing without"):
            p.validate_success(self.folder, self.identity)

    def test_corrupt_model_prevents_any_cleanup(self):
        (self.folder / "model.ans").write_text("corruption")
        with self.assertRaisesRegex(ValueError, "ans hash mismatch"):
            p.prune_success(self.folder, self.identity)
        self.assertTrue((self.folder / "model.fem").exists())
        self.assertFalse((self.folder / "artifact_retention.json").exists())

    def test_result_corruption_after_cleanup_is_rejected(self):
        p.prune_success(self.folder, self.identity)
        p.save(self.folder / "result.json", {**self.result, "raw_torque_nm": 0})
        with self.assertRaisesRegex(ValueError, "Result JSON hash"):
            p.validate_success(self.folder, self.identity)

    def test_receipt_is_bound_to_original_result(self):
        p.prune_success(self.folder, self.identity)
        receipt = self.folder / "artifact_retention.json"
        p.save(receipt, {**p.read(receipt), "ans_sha256": "0" * 64})
        state = p.read(self.folder / "state.json")
        p.save(self.folder / "state.json", {**state, "artifact_retention_sha256": p.sha(receipt)})
        with self.assertRaisesRegex(ValueError, "receipt differs"):
            p.validate_success(self.folder, self.identity)

    def test_interrupted_cleanup_resumes_from_receipt(self):
        original_unlink = Path.unlink
        def interrupted(path, *args, **kwargs):
            if path.name == "model.ans":
                raise PermissionError("simulated interruption after first deletion")
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", interrupted):
            with self.assertRaises(PermissionError):
                p.prune_success(self.folder, self.identity)
        self.assertFalse((self.folder / "model.fem").exists())
        self.assertTrue((self.folder / "model.ans").exists())
        self.assertEqual(p.validate_success(self.folder, self.identity), self.result)
        p.prune_success(self.folder, self.identity)
        self.assertFalse((self.folder / "model.ans").exists())

    def test_failed_angle_is_never_cleaned(self):
        state = p.read(self.folder / "state.json")
        p.save(self.folder / "state.json", {**state, "status": "failed"})
        with self.assertRaisesRegex(ValueError, "not successful"):
            p.prune_success(self.folder, self.identity)
        self.assertTrue((self.folder / "model.fem").exists())

    def test_cleanup_outside_run_root_is_rejected(self):
        with patch.object(p, "HERE", self.here / "other_project"):
            with self.assertRaisesRegex(ValueError, "outside pilot"):
                p.prune_success(self.folder, self.identity)
        self.assertTrue((self.folder / "model.fem").exists())

    def test_windows_com_does_not_use_file_transport(self):
        import win32com.client
        import femm
        commands = []
        class FakeCOM:
            def mlab2femm(self, command):
                commands.append(command)
                return "[2.5]" if command.startswith("mo_gapintegral(") else ""
        p.save(self.folder / "state.json", {**self.identity, "status": "pending", "attempts": []})
        with patch.object(femm, "windowsOS", False), patch.object(win32com.client, "DispatchEx", return_value=FakeCOM()):
            result = p.solve_angle(self.folder, self.contract)
        self.assertEqual(result["raw_torque_nm"], 2.5)
        self.assertEqual(result["attempt_number"], 1)
        self.assertIn("mi_analyze(1)", commands)
        self.assertEqual(p.read(self.folder / "state.json")["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
