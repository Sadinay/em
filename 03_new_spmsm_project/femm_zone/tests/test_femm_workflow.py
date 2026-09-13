"""离线验证重构；这些测试不会启动FEMM或新的参数筛选。"""
import builtins
import copy
import math
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ZONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ZONE))
import femm_config as config
import run_femm as runner

TEACHER_RUN = ZONE / "workspaces/teacher_angle29_5genes_20260908"


def field(text, name, header=False):
    key = rf"\[{name}\]" if header else rf"<{name}>"
    return float(re.search(key + r"\s*=\s*([^\r\n]+)", text)[1])


@pytest.mark.parametrize("phase,inner,previous", [(0, 0, "is35_plus"), (0, 15, "is35_plus"), (120, 3, "phase_p120")])
def test_written_model_matches_saved_current_and_preserves_geometry(phase, inner, previous):
    cfg = config.load_config()
    base = ZONE / "results/current_modes_5genes_20260908/bases/median_tavg/model.fem"
    before = base.read_text(encoding="utf-8")
    absolute_inner = inner + cfg["airgap"]["initial_inner_angle_deg"]
    after = config.configure_fem(before, cfg, absolute_inner, phase)
    gap = next(b for b in re.findall(r"<BeginBdry>.*?<EndBdry>", after, re.S) if '"sliding_airgap"' in b)
    assert field(gap, "innerangle") == absolute_inner
    assert field(gap, "outerangle") == 0
    old_root = "current_modes_5genes_20260908" if phase == 0 else "initial_current_phase_20260908"
    old = (ZONE / "results" / old_root / previous / "median_tavg" / f"angle_{inner:03d}/model.fem").read_text()
    for name in ("Frequency", "Precision", "MinAngle", "DoSmartMesh", "Depth"):
        assert field(after, name, True) == field(old, name, True)
    circuits = lambda text: re.findall(r"<BeginCircuit>.*?<EndCircuit>", text, re.S)
    for new_block, old_block in zip(circuits(after), circuits(old)):
        assert field(new_block, "TotalAmps_re") == pytest.approx(field(old_block, "TotalAmps_re"), abs=1e-12)
        assert field(new_block, "TotalAmps_im") == 0
    assert after.split("[NumPoints]", 1)[1] == before.split("[NumPoints]", 1)[1]
    assert re.findall(r"<BeginBlock>.*?<EndBlock>", after, re.S) == re.findall(r"<BeginBlock>.*?<EndBlock>", before, re.S)


def test_prepare_uses_config_and_never_imports_femm(tmp_path, monkeypatch):
    cfg = copy.deepcopy(config.load_config())
    cfg["genes"] = cfg["genes"][1:2]
    cfg["inner_angles_deg"], cfg["initial_phases_deg"] = [0, 3], [120]
    cfg["problem"]["MinAngle"] = 18  # 检验参数确实来自配置，而非求解脚本硬编码
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(config, "RUN_ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "femm", None)  # 意外导入FEMM会直接失败
    manifest = runner.prepare("offline")
    assert manifest["status"] == "prepared" and len(manifest["cases"]) == 2
    for case in manifest["cases"]:
        model = tmp_path / "offline" / case["model"]
        assert field(model.read_text(), "MinAngle", True) == 18
        assert config.sha256(model) == case["prepared_sha256"]
        assert not model.with_suffix(".ans").exists()
    with pytest.raises(FileExistsError):
        runner.prepare("offline")


def test_metrics_match_previous_five_gene_results():
    cfg = config.load_config()
    cfg["torque_multiplier"] = -2  # 旧试验用旧倍率，不能冒充本次直接积分结果
    previous = runner.read(ZONE / "results/current_modes_5genes_20260908/summary.json")
    for row in previous["results"]:
        if row["scenario_id"] == "is35_plus":
            actual = config.torque_metrics(row["raw_gap_torques_nm"], cfg)
            for key in ("tavg_nm", "peak_to_peak_nm", "ripple_relative"):
                assert actual[key] == pytest.approx(row[key], abs=1e-12)


def test_interrupt_closes_only_the_private_femm_instance(tmp_path, monkeypatch):
    cfg = config.load_config()
    monkeypatch.setattr(config, "RUN_ROOT", tmp_path)
    output = tmp_path / "interrupt"
    output.mkdir()
    model = output / "model.fem"
    model.write_text(config.configure_fem(config.TEMPLATE_FILE.read_text(), cfg, 29, 0))
    runner.save(output / "run.json", {"status": "prepared", "config": cfg,
                                      "cases": [{"model": "model.fem", "prepared_sha256": config.sha256(model)}]})
    calls = []
    def interrupted(_):
        raise KeyboardInterrupt()
    fake = SimpleNamespace(main_minimize=lambda: None, opendocument=lambda path: calls.append("open"),
                           mi_analyze=interrupted, closefemm=lambda: calls.append("close"))
    client = SimpleNamespace(DispatchEx=lambda progid: calls.append(progid))
    monkeypatch.setitem(sys.modules, "femm", fake)
    monkeypatch.setitem(sys.modules, "win32com", SimpleNamespace(client=client))
    monkeypatch.setitem(sys.modules, "win32com.client", client)
    with pytest.raises(KeyboardInterrupt):
        runner.solve("interrupt")
    assert calls == ["femm.ActiveFEMM", "open", "close"]
    assert runner.read(output / "run.json")["status"] == "interrupted"


def test_report_reuses_real_saved_answers_without_femm(tmp_path, monkeypatch):
    """用已有真实解验证报告链路，既不模拟转矩，也不启动求解器。"""
    cfg = config.load_config()
    monkeypatch.setattr(config, "RUN_ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "femm", None)
    saved = runner.read(TEACHER_RUN / "run.json")
    gene = next(g for g in saved["genes"] if g["id"] == "G2")
    output = tmp_path / "saved_answers"
    cases = []
    for inner in cfg["inner_angles_deg"]:
        source = TEACHER_RUN / "G2/phase_0" / f"angle_{inner:g}"
        target = output / f"angle_{inner:g}"
        target.mkdir(parents=True)
        for name in ("model.fem", "model.ans", "result.json"):
            shutil.copy2(source / name, target / name)
        cases.append({"gene_id": "G2", "initial_phase_deg": 0, "inner_angle_deg": inner,
                      "model": str((target / "model.fem").relative_to(output))})
    runner.save(output / "run.json", {"status": "solved", "config": cfg, "genes": [gene], "cases": cases})
    report = runner.report("saved_answers")
    actual = report["rows"][0]
    assert actual["tavg_nm"] == pytest.approx(gene["reference_tavg_nm"], rel=0, abs=1e-12)
    assert actual["peak_to_peak_nm"] == pytest.approx(gene["reference_delta_t"], rel=0, abs=1e-12)
    assert report["report_definition"]["reference_delta_t_metric"] == "peak_to_peak_nm"
    assert actual["comparison_torques_nm"] == actual["raw_torques_nm"]
    assert (output / "comparison_phase_0.png").is_file()


@pytest.mark.parametrize("inner,expected", [(29, (3.5, -1.75, -1.75)), (44, (1.75, 1.75, -3.5))])
def test_teacher_initial_offset_does_not_enter_current_phase(inner, expected):
    cfg = config.load_config()
    assert cfg["inner_angles_deg"] == [29, 32, 35, 38, 41, 44]
    currents = config.phase_currents(cfg, inner, 0)
    assert tuple(currents.values()) == pytest.approx(expected, abs=1e-12)
    assert cfg["torque_multiplier"] == 1


def test_six_angle_config_and_written_currents_match_verified_teacher_run():
    """核对冻结的真实六角度工况，并独立按电角度 4×机械行程计算三相电流。"""
    cfg = config.load_config()
    saved = runner.read(TEACHER_RUN / "run.json")
    for key in ("current", "problem", "airgap", "rotor_travel_angles_deg",
                "inner_angles_deg", "initial_phases_deg", "torque_multiplier"):
        assert cfg[key] == saved["config"][key], key
    assert cfg["rotor_travel_angles_deg"] == [0, 3, 6, 9, 12, 15]
    assert cfg["inner_angles_deg"] == [29, 32, 35, 38, 41, 44]
    assert cfg["initial_phases_deg"] == [0]
    assert cfg["torque_multiplier"] == 1
    template = config.TEMPLATE_FILE.read_text(encoding="utf-8")
    cases = [case for case in saved["cases"] if case["gene_id"] == "G2"]
    for travel, inner, case in zip(cfg["rotor_travel_angles_deg"], cfg["inner_angles_deg"], cases, strict=True):
        expected = {name: 3.5 * math.cos(math.radians(4 * travel - phase))
                    for name, phase in (("A", 0), ("B", 120), ("C", 240))}
        assert config.phase_currents(cfg, inner, 0) == pytest.approx(expected, rel=0, abs=1e-12)
        assert case["currents_a"] == pytest.approx(expected, rel=0, abs=1e-12)
        model = config.configure_fem(template, cfg, inner, 0)
        for key in ("Frequency", "Precision", "MinAngle", "DoSmartMesh", "Depth", "ACSolver", "PrevType"):
            assert field(model, key, True) == cfg["problem"][key]
        gap = next(b for b in re.findall(r"<BeginBdry>.*?<EndBdry>", model, re.S) if '"sliding_airgap"' in b)
        assert field(gap, "BdryType") == 6
        assert field(gap, "innerangle") == inner
        assert field(gap, "outerangle") == 0
        circuits = re.findall(r"<BeginCircuit>.*?<EndCircuit>", model, re.S)
        assert len(circuits) == 3
        for block in circuits:
            name = re.search(r'<CircuitName>\s*=\s*"([ABC])"', block)[1]
            assert field(block, "TotalAmps_re") == pytest.approx(expected[name], rel=0, abs=1e-12)
            assert field(block, "TotalAmps_im") == 0


@pytest.mark.parametrize("drift", ["legacy_0_minus2", "current", "problem", "airgap",
                                  "rotor_travel_angles_deg", "inner_angles_deg",
                                  "initial_phases_deg", "torque_multiplier", "missing_field"])
def test_solve_rejects_stale_prepared_physics_before_femm_import(tmp_path, monkeypatch, drift):
    cfg = copy.deepcopy(config.load_config())
    if drift == "legacy_0_minus2":
        cfg["airgap"]["initial_inner_angle_deg"] = 0
        cfg["inner_angles_deg"] = [0, 3, 6, 9, 12, 15]
        cfg["torque_multiplier"] = -2
    elif drift == "current":
        cfg["current"]["amplitude_a"] = 5
    elif drift == "problem":
        cfg["problem"]["MinAngle"] = 30
    elif drift == "airgap":
        cfg["airgap"]["outer_angle_deg"] = 3
    elif drift == "rotor_travel_angles_deg":
        cfg[drift] = [0, 5, 10, 15]
    elif drift == "inner_angles_deg":
        cfg[drift] = [29, 34, 39, 44]
    elif drift == "initial_phases_deg":
        cfg[drift] = [120]
    elif drift == "torque_multiplier":
        cfg[drift] = -2
    else:
        del cfg["rotor_travel_angles_deg"]
    monkeypatch.setattr(config, "RUN_ROOT", tmp_path)
    output = tmp_path / "stale"
    output.mkdir()
    manifest = {"status": "prepared", "config": cfg, "cases": []}
    runner.save(output / "run.json", manifest)
    original_import = builtins.__import__
    def reject_solver_import(name, *args, **kwargs):
        if name.split(".")[0] in {"femm", "win32com"}:
            pytest.fail(f"Stale physics reached solver import: {name}")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_solver_import)
    with pytest.raises(ValueError):
        runner.solve("stale")
    assert runner.read(output / "run.json") == manifest


def test_solve_config_accepts_verified_physics_without_requiring_historical_notes():
    saved = runner.read(TEACHER_RUN / "run.json")["config"]
    runner.check_solve_config(saved)
