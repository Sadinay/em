"""Audit the returned pilot ZIP and independently rerun two preselected genes.

Run from any directory: python path/to/check.py [--solve]
Uses the local, previously validated mapping/configuration; never executes ZIP code.
"""
from pathlib import Path
import argparse
import csv
import difflib
import hashlib
import io
import json
import math
import sys
import time
import zipfile

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
from femm_zone import femm_config as physical
from femm_zone.scripts import spmsm_mapping as mapping
from experiments.input_distribution_pilot_v1 import pilot

ARCHIVE = PROJECT / "outputs/pilot20_results_20260909.zip"
TOLERANCE_NM = 1e-6


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_write(name, rows):
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit():
    cfg = pilot.physical_config()
    template = physical.TEMPLATE_FILE.read_text(encoding="utf-8")
    positions = loadmat(physical.MAT_FILE, variable_names=["MaterialPosition"], squeeze_me=True)["MaterialPosition"]
    mapped = mapping.match_material_positions(template, positions)
    models, results, angle_audit = {}, {}, []
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        require(archive.testzip() is None, "ZIP CRC failure")
        checksums = {}
        for line in archive.read("checksums.sha256").decode("utf-8-sig").splitlines():
            digest, name = line.split("  ", 1)
            require(name not in checksums, "Duplicate checksum entry")
            require(sha(archive.read(name)) == digest, f"ZIP checksum mismatch: {name}")
            checksums[name] = digest
        uncovered = sorted(set(names) - set(checksums) - {"checksums.sha256"})
        require(uncovered == ["TRANSFER_README.txt"], f"Unexpected unlisted files: {uncovered}")
        read = lambda name: json.loads(archive.read(name))
        read_csv = lambda name: list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))
        root = next(n for n in names if n.endswith("/contract.json")).rsplit("/", 1)[0]
        contract = read(root + "/contract.json")
        fingerprint = root.split("/")[-1]
        require(pilot.digest(contract) == fingerprint, "Contract fingerprint differs")
        identities = {
            "template_sha256": physical.sha256(physical.TEMPLATE_FILE),
            "mat_sha256": physical.sha256(physical.MAT_FILE),
            "mapping_source_sha256": physical.sha256(Path(mapping.__file__)),
            "physical_source_sha256": physical.sha256(Path(physical.__file__)),
        }
        for key, value in identities.items():
            require(contract[key] == value, f"Local/remote mismatch: {key}")
        for key in ("current", "problem", "airgap", "rotor_travel_angles_deg", "inner_angles_deg", "initial_phases_deg", "torque_multiplier"):
            require(cfg[key] == contract["physical"][key], f"Physical configuration differs: {key}")
        solver = pilot.solver_identity()
        require(solver["binary_sha256"] == contract["solver"]["binary_sha256"], "FEMM executable differs")
        manifests = ("pilot20.csv", "femm_queue.csv", "train_G.csv", "train_F.csv", "dev_common.csv", "test_common.csv", "memberships.csv")
        for name in manifests:
            require(sha((pilot.HERE / name).read_bytes()) == sha(archive.read(name)), f"Frozen manifest differs: {name}")
        rows = read_csv("pilot20.csv")
        labels_all = read_csv(root + "/labels.csv")
        labels = {r["gene_id"]: r for r in labels_all if r["status"] == "succeeded"}
        waveforms = read_csv(root + "/waveforms.csv")
        wave = {(r["gene_id"], float(r["inner_angle_deg"])): r for r in waveforms}
        require(len(rows) == len(labels) == 20 and len(wave) == len(waveforms) == 120, "Unexpected batch counts")
        require(set(labels) == {r["gene_id"] for r in rows}, "Pilot and successful gene sets differ")
        max_aggregate_error = 0.0
        for row in rows:
            gid = row["gene_id"]
            bits = np.array([int(c) for c in row["bits"]], dtype=np.uint8)
            require(len(bits) == 120 and mapping.genotype_sha256(bits) == gid, "Invalid gene hash")
            require(int(bits.sum()) == int(row["magnet_cells"]), "Incorrect PM count")
            base = mapping.replace_cell_materials(template, bits)
            mapping.validate_generated_model(base, mapped, bits)
            values = []
            for inner in cfg["inner_angles_deg"]:
                prefix = f"{root}/{gid}/angle_{inner:g}/"
                state, result, receipt = [read(prefix + f"{name}.json") for name in ("state", "result", "artifact_retention")]
                content = physical.configure_fem(base, cfg, inner, 0).encode("utf-8")
                expected = {"gene_id": gid, "bits": row["bits"], "inner_angle_deg": inner,
                            "rotor_travel_deg": inner - 29, "currents_a": physical.phase_currents(cfg, inner, 0),
                            "condition_fingerprint": fingerprint, "prepared_sha256": sha(content)}
                require(state["status"] == "succeeded", "Unsuccessful angle")
                for key, value in expected.items():
                    require(state[key] == result[key] == value, f"Angle identity/input mismatch: {gid}/{inner}/{key}")
                require(state["result_sha256"] == sha(archive.read(prefix + "result.json")), "Result hash mismatch")
                require(state["artifact_retention_sha256"] == sha(archive.read(prefix + "artifact_retention.json")), "Receipt hash mismatch")
                for key in ("condition_fingerprint", "gene_id", "inner_angle_deg", "fem_sha256", "ans_sha256"):
                    require(receipt[key] == result[key], f"Receipt field mismatch: {key}")
                require(receipt["model_settings_verified"] is True and receipt["result_sha256"] == state["result_sha256"], "Invalid receipt")
                require(math.isfinite(result["raw_torque_nm"]), "Nonfinite torque")
                w = wave[gid, inner]
                for key in ("raw_torque_nm", "rotor_travel_deg"):
                    require(float(w[key]) == result[key], f"Waveform differs: {key}")
                for phase, current in expected["currents_a"].items():
                    require(float(w[phase]) == current, "Waveform current differs")
                for key in ("fem_sha256", "ans_sha256"):
                    require(w[key] == result[key], "Waveform artifact hash differs")
                models[gid, inner], results[gid, inner] = content, result
                values.append(result["raw_torque_nm"])
                angle_audit.append({**expected, "prepared_input_matches_local": True, "remote_raw_torque_nm": result["raw_torque_nm"]})
            mean_error = abs(float(np.mean(values)) - float(labels[gid]["tavg_nm"]))
            ripple_error = abs(max(values) - min(values) - float(labels[gid]["delta_t_nm"]))
            require(max(mean_error, ripple_error) < 1e-12, "Stored aggregate is incorrect")
            max_aggregate_error = max(max_aggregate_error, mean_error, ripple_error)
        selected = [max(rows, key=lambda r: float(labels[r["gene_id"]]["tavg_nm"])),
                    max(rows, key=lambda r: float(labels[r["gene_id"]]["delta_t_nm"]))]
        require(selected[0]["gene_id"] != selected[1]["gene_id"], "Selection criteria selected same gene")
        remote_code = archive.read("pilot.py").decode("utf-8-sig")
        local_code = Path(pilot.__file__).read_text(encoding="utf-8")
        (HERE / "remote_runner.diff").write_text("".join(difflib.unified_diff(local_code.splitlines(True), remote_code.splitlines(True), fromfile="local/pilot.py", tofile="archive/pilot.py")), encoding="utf-8")
        info = {"archive": str(ARCHIVE), "archive_sha256": physical.sha256(ARCHIVE), "archive_files": len(names),
                "checksums_verified": len(checksums), "unlisted_files": uncovered,
                "frozen_manifests_identical": list(manifests), "local_source_hashes": identities,
                "solver": solver, "physical": cfg, "remote_condition_fingerprint": fingerprint,
                "verified_genes": len(rows), "verified_angles": len(angle_audit),
                "max_remote_aggregation_error_nm": max_aggregate_error,
                "selection_rule": "Maximum remote mean torque and maximum remote peak-to-peak ripple, chosen before rerun",
                "selected_genes": selected, "absolute_tolerance_nm": TOLERANCE_NM,
                "limitation": "Remote FEM/ANS files are absent; checked hash-linked receipts and reconstructed all 120 prepared FEM hashes. Fresh solves cover only the two selected genes."}
        save("archive_audit.json", info)
        save("angle_audit.json", angle_audit)
        save("remote_selected_labels.json", [labels[r["gene_id"]] for r in selected])
    print(f"AUDIT PASS: {len(checksums)} checksums, 20 genes, 120 reconstructed input hashes", flush=True)
    print("SELECTED: " + ", ".join(r["gene_id"] for r in selected), flush=True)
    return cfg, selected, models, results, labels


def solve(cfg, selected, models, remote, labels):
    # Explicit COM initialization/transport; no ZIP executable code is imported.
    import pythoncom
    import win32com.client
    import femm
    pythoncom.CoInitialize()
    comparisons, summaries = [], []
    started = time.perf_counter()
    try:
        for row in selected:
            gid = row["gene_id"]
            values, differences = [], []
            for inner in cfg["inner_angles_deg"]:
                folder = HERE / "rerun" / gid / f"angle_{inner:g}"
                folder.mkdir(parents=True, exist_ok=True)
                model = folder / "model.fem"
                require(not model.exists(), f"Refusing to overwrite existing rerun: {model}")
                model.write_bytes(models[gid, inner])
                pilot.validate_model_settings(model, cfg, inner)
                tick = time.perf_counter()
                connected = False
                try:
                    femm.HandleToFEMM = win32com.client.DispatchEx("femm.ActiveFEMM")
                    connected = True
                    femm.windowsOS = True
                    femm.main_minimize()
                    femm.opendocument(str(model))
                    femm.mi_analyze(1)
                    femm.mi_loadsolution()
                    raw = complex(femm.mo_gapintegral("sliding_airgap", 0))
                    require(math.isfinite(raw.real) and abs(raw.imag) <= 1e-10, "Invalid FEMM torque")
                    pilot.validate_model_settings(model, cfg, inner)
                    value = raw.real
                    record = {"gene_id": gid, "inner_angle_deg": inner, "rotor_travel_deg": inner - 29,
                              "currents_a": physical.phase_currents(cfg, inner, 0),
                              "prepared_sha256": sha(models[gid, inner]), "raw_torque_nm": value,
                              "fem_sha256": physical.sha256(model), "ans_sha256": physical.sha256(model.with_suffix(".ans")),
                              "elapsed_seconds": time.perf_counter() - tick}
                    (folder / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                finally:
                    if connected:
                        femm.closefemm()
                expected = remote[gid, inner]["raw_torque_nm"]
                error = value - expected
                values.append(value)
                differences.append(abs(error))
                comparisons.append({"gene_id": gid, "source": row["source"], "magnet_cells": row["magnet_cells"],
                                    "inner_angle_deg": inner, "rotor_travel_deg": inner - 29,
                                    "remote_torque_nm": expected, "local_torque_nm": value, "difference_nm": error,
                                    "prepared_sha256": record["prepared_sha256"],
                                    "solved_fem_hash_matches": record["fem_sha256"] == remote[gid, inner]["fem_sha256"],
                                    "solved_ans_hash_matches": record["ans_sha256"] == remote[gid, inner]["ans_sha256"]})
                csv_write("angle_comparison.csv", comparisons)
                print(f"SOLVED {gid[:12]} inner={inner:g}: local={value:.15g}, remote={expected:.15g}, diff={error:.3g} Nm", flush=True)
            local_mean, local_ripple = float(np.mean(values)), max(values) - min(values)
            remote_mean, remote_ripple = float(labels[gid]["tavg_nm"]), float(labels[gid]["delta_t_nm"])
            summaries.append({"gene_id": gid, "source": row["source"], "magnet_cells": int(row["magnet_cells"]),
                              "remote_tavg_nm": remote_mean, "local_tavg_nm": local_mean,
                              "remote_delta_t_nm": remote_ripple, "local_delta_t_nm": local_ripple,
                              "tavg_difference_nm": local_mean - remote_mean, "delta_t_difference_nm": local_ripple - remote_ripple,
                              "max_point_difference_nm": max(differences),
                              "passed": max(differences + [abs(local_mean - remote_mean), abs(local_ripple - remote_ripple)]) <= TOLERANCE_NM})
            csv_write("summary.csv", summaries)
    finally:
        pythoncom.CoUninitialize()
    save("rerun_summary.json", {"elapsed_seconds": time.perf_counter() - started, "genes": summaries,
                                 "passed": all(s["passed"] for s in summaries), "absolute_tolerance_nm": TOLERANCE_NM})
    figure(comparisons, summaries)
    print(json.dumps(summaries, indent=2), flush=True)


def figure(points, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, summary in zip(axes, summaries):
        rows = [r for r in points if r["gene_id"] == summary["gene_id"]]
        x = [r["rotor_travel_deg"] for r in rows]
        ax.plot(x, [r["remote_torque_nm"] for r in rows], "o-", label="Other device", linewidth=2.5, markersize=7)
        ax.plot(x, [r["local_torque_nm"] for r in rows], "x--", label="Local fresh FEMM", linewidth=1.4, markersize=7)
        ax.set(title=f"{summary['gene_id'][:12]} | {summary['source']} | PM cells: {summary['magnet_cells']}",
               xlabel="Rotor travel (mechanical deg); inner = travel + 29 deg", ylabel="Torque (N m)", xticks=x)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        ax.text(0.02, -0.25, f"Other: mean {summary['remote_tavg_nm']:.9f}; ripple {summary['remote_delta_t_nm']:.9f} N m\n"
                f"Local:  mean {summary['local_tavg_nm']:.9f}; ripple {summary['local_delta_t_nm']:.9f} N m\n"
                f"Max point difference: {summary['max_point_difference_nm']:.3g} N m", transform=ax.transAxes, fontsize=9)
    fig.suptitle("03 pilot cross-device check | 3.5 A, current phase = 4 x travel | Min Angle = 15 deg")
    fig.savefig(HERE / "torque_comparison.png", dpi=180)
    fig.savefig(HERE / "torque_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solve", action="store_true", help="Actually run 12 FEMM angle solves")
    args = parser.parse_args()
    data = audit()
    if args.solve:
        solve(*data)
