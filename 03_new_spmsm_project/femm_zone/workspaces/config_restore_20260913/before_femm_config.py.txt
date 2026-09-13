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

# 修改试验时只改这里；角度单位均为度。扫描已停止，默认仅初相位 0°。
INNER_ANGLES = (0, 3, 6, 9, 12, 15)       # 气隙内角：机械角
INITIAL_PHASES = (0.0,)                  # 电流初始电角度；整圈粗筛可用 tuple(range(0, 360, 30))
GENES = ((0, 338), (31, 190), (193, 258), (200, 491), (200, 33))  # MAT 的代/行，均为0-based


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict:
    
    inp = loadmat(MAT_FILE, variable_names=["inp"], squeeze_me=True, struct_as_record=False)["inp"]
    cfg = {
        "mat_file": str(MAT_FILE), "template_file": str(TEMPLATE_FILE),
        "input_sha256": {str(p): sha256(p) for p in (MAT_FILE, TEMPLATE_FILE)},
        "current": {"amplitude_a": float(inp.Is_amp), "pole_pairs": int(inp.P)},
        "current_source": "MAT inp.Is_amp、inp.P；本方案不使用 Id/Iq 或保存的零值 Ia/Ib/Ic。",
        "problem": {
            "Frequency": 0,              # 静磁求解；不是 MAT 中运行频率 F=400 Hz
            "Precision": 1e-8,
            "MinAngle": 15,
            "DoSmartMesh": 1,
            "Depth": float(inp.Lfe),     # MAT inp.Lfe=36 mm
            "LengthUnits": "millimeters", "ProblemType": "planar",
            "Coordinates": "cartesian", "ACSolver": 0, "PrevType": 0, "PrevSoln": "",
        },
        "airgap": {"name": "sliding_airgap", "boundary_type": 6, "outer_angle_deg": 0},
        "inner_angles_deg": list(INNER_ANGLES), "initial_phases_deg": list(INITIAL_PHASES),
        "gene_source": "population_all",
        "genes": [{"id": f"G{i}", "state": state, "row": row} for i, (state, row) in enumerate(GENES, 1)],
        "torque_multiplier": -2.0,       # 沿用以前的经验比较尺度，历史物理依据尚未确认
        "comparison_note": "六点算术平均；DeltaT暂按相对峰峰值比较，历史公式尚未确认。",
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
    return cfg


def phase_currents(cfg: dict, inner_deg: float, initial_deg: float) -> dict[str, float]:
    """正向同步：θe=P×θinner+φ0；A/B/C依次相差0°、−120°、+120°。"""
    theta = cfg["current"]["pole_pairs"] * inner_deg + initial_deg
    amp = cfg["current"]["amplitude_a"]
    return {name: amp * math.cos(math.radians(theta + shift))
            for name, shift in (("A", 0), ("B", -120), ("C", 120))}


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
