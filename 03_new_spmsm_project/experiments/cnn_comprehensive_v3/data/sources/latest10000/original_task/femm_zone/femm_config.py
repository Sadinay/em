"""03 的唯一 FEMM 配置入口：参数来源、电流公式、写入模型、转矩口径。"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from scipy.io import loadmat

ZONE = Path(__file__).resolve().parent
MAT_FILE = ZONE.parent / "data_zone/raw/workspace_200.mat"
TEMPLATE_FILE = ZONE / "models/SPMSM_discrete.fem"
RUN_ROOT = ZONE / "workspaces"

# 修改试验时只改这里；角度单位均为度。本次只复现5个基因，不恢复整圈筛选。
INITIAL_INNER_ANGLE = 29.0               # 导师补充代码 ang_0；不是从MAT读取
ROTOR_TRAVEL_ANGLES = (0, 3, 6, 9, 12, 15)  # ANG_R：从初始位置转过的机械角
INITIAL_PHASES = (0.0,)                  # t=0时的电流电角度
GENES = ((0, 338), (31, 190), (193, 258), (200, 491), (200, 33))  # MAT 的代/行，均为0-based
REFERENCE_DELTA_T_METRIC = "peak_to_peak_nm"  # 29°重算：4个基因与MAT DeltaT逐一符合至1e-12 N·m


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict:
    """直接读取原始 MAT，不继承旧试验的 run_spec.json。"""
    inp = loadmat(MAT_FILE, variable_names=["inp"], squeeze_me=True, struct_as_record=False)["inp"]
    cfg = {
        "mat_file": str(MAT_FILE), "template_file": str(TEMPLATE_FILE),
        "input_sha256": {str(p): sha256(p) for p in (MAT_FILE, TEMPLATE_FILE)},
        "current": {"amplitude_a": float(inp.Is_amp), "pole_pairs": int(inp.P),
                    "omega_e_rad_s": 2 * math.pi * float(inp.F), "omega_m_rad_s": float(inp.wmech)},
        "current_source": "导师cos(omega*t)公式；MAT inp.Is_amp、F、wmech、P。t=deg2rad(ANG_R)/wmech。",
        "problem": {
            "Frequency": 0,              # 静磁求解；不是 MAT 中运行频率 F=400 Hz
            "Precision": 1e-8,
            "MinAngle": 15,
            "DoSmartMesh": 1,
            "Depth": float(inp.Lfe),     # MAT inp.Lfe=36 mm
            "LengthUnits": "millimeters", "ProblemType": "planar",
            "Coordinates": "cartesian", "ACSolver": 0, "PrevType": 0, "PrevSoln": "",
        },
        "airgap": {"name": "sliding_airgap", "boundary_type": 6, "outer_angle_deg": 0,
                   "initial_inner_angle_deg": INITIAL_INNER_ANGLE},
        "angle_source": "导师补充ang_0=29°；用户指定ANG_R=0:3:15°。原脚本T/ANG_R生成部分未提供。",
        "rotor_travel_angles_deg": list(ROTOR_TRAVEL_ANGLES),
        "inner_angles_deg": [INITIAL_INNER_ANGLE + a for a in ROTOR_TRAVEL_ANGLES],
        "initial_phases_deg": list(INITIAL_PHASES),
        "gene_source": "population_all",
        "genes": [{"id": f"G{i}", "state": state, "row": row} for i, (state, row) in enumerate(GENES, 1)],
        "torque_multiplier": 1.0,        # 直接报告FEMM原始气隙积分；不拟合倍率或符号
        "comparison_note": "六点算术平均；MAT DeltaT按峰峰转矩差(N·m)比较。旧-2经验换算另存对照。",
        "material_source": "几何、绕组和材料BH曲线来自FEM模板；基因只切换设计区Air/N38。",
    }
    for key in ("inner_angles_deg", "initial_phases_deg"):
        values = cfg[key]
        if not values or not all(math.isfinite(v) for v in values):
            raise ValueError(f"{key} 必须包含有限角度值")
    if len({round(p % 360, 9) for p in cfg["initial_phases_deg"]}) != len(cfg["initial_phases_deg"]):
        raise ValueError("初相位含等价重复值，例如0°和360°")
    if not cfg["genes"]:
        raise ValueError("至少配置一个基因")
    if float(inp.P) != cfg["current"]["pole_pairs"] or inp.P <= 0 or not math.isfinite(inp.Is_amp) or inp.Is_amp <= 0:
        raise ValueError("MAT 中的极对数/电流幅值无效")
    current = cfg["current"]
    if not (current["omega_m_rad_s"] > 0 and math.isfinite(current["omega_m_rad_s"]) and
            math.isclose(current["omega_e_rad_s"], inp.P * current["omega_m_rad_s"], rel_tol=1e-12)):
        raise ValueError("MAT电频率、机械角速度与极对数不一致")
    return cfg


def phase_currents(cfg: dict, inner_deg: float, initial_deg: float) -> dict[str, float]:
    """导师公式：t=ANG_R/wmech，θe=omega*t；29°只加到滑动气隙内角。"""
    travel = inner_deg - cfg["airgap"]["initial_inner_angle_deg"]
    time_s = math.radians(travel) / cfg["current"]["omega_m_rad_s"]
    theta = cfg["current"]["omega_e_rad_s"] * time_s + math.radians(initial_deg)
    amp = cfg["current"]["amplitude_a"]
    return {name: amp * math.cos(theta + shift)
            for name, shift in (("A", 0), ("B", -2*math.pi/3), ("C", -4*math.pi/3))}


def _set(text: str, pattern: str, value: str) -> str:
    changed, count = re.subn(pattern, lambda m: m[1] + value, text)
    if count != 1:
        raise ValueError(f"FEM字段应出现一次，实际{count}次：{pattern}")
    return changed


def configure_fem(text: str, cfg: dict, inner_deg: float, initial_deg: float) -> str:
    """直接写入 .fem 文本；准备阶段即可审查全部输入，不需要启动 FEMM。"""
    for name, value in cfg["problem"].items():
        rendered = f'"{value}"' if name == "PrevSoln" else format(value, ".17g") if isinstance(value, (int, float)) else value
        text = _set(text, rf"(\[{re.escape(name)}\][ \t]*=[ \t]*)[^\r\n]*", rendered)
    gaps = [b for b in re.findall(r"<BeginBdry>.*?<EndBdry>", text, re.S)
            if re.search(r'<BdryName>\s*=\s*"' + re.escape(cfg["airgap"]["name"]) + '"', b)]
    if len(gaps) != 1 or int(re.search(r"<BdryType>\s*=\s*(\d+)", gaps[0])[1]) != cfg["airgap"]["boundary_type"]:
        raise ValueError("滑动气隙名称或边界类型与配置不符")
    gap = _set(gaps[0], r"(<innerangle>[ \t]*=[ \t]*)[^\r\n]*", format(inner_deg, ".17g"))
    gap = _set(gap, r"(<outerangle>[ \t]*=[ \t]*)[^\r\n]*", format(cfg["airgap"]["outer_angle_deg"], ".17g"))
    text = text.replace(gaps[0], gap, 1)
    currents, seen = phase_currents(cfg, inner_deg, initial_deg), set()
    for block in re.findall(r"<BeginCircuit>.*?<EndCircuit>", text, re.S):
        name = re.search(r'<CircuitName>\s*=\s*"([^"]+)"', block)[1]
        if name not in currents or name in seen:
            raise ValueError(f"意外或重复的电路名称：{name}")
        updated = _set(block, r"(<TotalAmps_re>[ \t]*=[ \t]*)[^\r\n]*", format(currents[name], ".17g"))
        updated = _set(updated, r"(<TotalAmps_im>[ \t]*=[ \t]*)[^\r\n]*", "0")
        text = text.replace(block, updated, 1)
        seen.add(name)
    if seen != currents.keys():
        raise ValueError("FEM模板缺少A/B/C电路")
    return text


def torque_metrics(raw_torques: list[float], cfg: dict) -> dict:
    """保留原始气隙积分值；比较转矩的缩放、均值和波动只在这里计算。"""
    if not raw_torques or not all(math.isfinite(t) for t in raw_torques):
        raise ValueError("转矩必须是非空有限数值序列")
    torque = [cfg["torque_multiplier"] * t for t in raw_torques]
    mean = math.fsum(torque) / len(torque)
    peak = max(torque) - min(torque)
    return {"raw_torques_nm": raw_torques, "comparison_torques_nm": torque,
            "tavg_nm": mean, "peak_to_peak_nm": peak,
            "ripple_relative": peak / abs(mean) if mean else None}
