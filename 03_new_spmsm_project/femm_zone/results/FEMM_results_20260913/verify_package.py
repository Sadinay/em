"""Verify the transferred results with Python's standard library; no FEMM or NumPy needed."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import zipfile


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(base):
    base = Path(base).resolve()
    summary = json.loads((base / "package_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((base / "checksums.json").read_text(encoding="utf-8"))
    actual = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
    check(actual == set(manifest) | {"checksums.json"}, "Package file list changed")
    for name, expected in manifest.items():
        path = (base / name).resolve()
        check(path.is_relative_to(base), "Checksum path leaves package")
        check(sha(path.read_bytes()) == expected, f"File checksum mismatch: {name}")
    labels = csv_rows(base / "tables/labels.csv")
    dataset = csv_rows(base / "tables/dataset_all.csv")
    waves = csv_rows(base / "tables/waveforms.csv")
    check(len(labels) == len(dataset) == 3299 and len(waves) == 19794, "Wrong table counts")
    indexed = {row["gene_id"]: row for row in labels}
    genes = {row["gene_id"]: row for row in dataset}
    check(len(indexed) == len(genes) == 3299 and indexed.keys() == genes.keys(), "Gene IDs differ")
    grouped = {gid: {} for gid in genes}
    for row in waves:
        gid, angle = row["gene_id"], float(row["inner_angle_deg"])
        check(gid in grouped and angle not in grouped[gid], "Unknown gene or duplicate angle")
        grouped[gid][angle] = row
    split_sets = {}
    for name, count in (("train_G", 1400), ("train_F", 1400), ("dev_common", 200), ("test_common", 400)):
        split = csv_rows(base / f"tables/{name}_labeled.csv")
        ids = {row["gene_id"] for row in split}
        check(len(split) == len(ids) == count and ids <= genes.keys(), f"Invalid split: {name}")
        for row in split:
            check(row["status"] == "succeeded" and row["bits"] == genes[row["gene_id"]]["bits"], "Split data differs")
            check(all(row[k] == indexed[row["gene_id"]][k] for k in ("tavg_nm", "delta_t_nm", "condition_fingerprint")), "Split labels differ")
        split_sets[name] = ids
    train = split_sets["train_G"] | split_sets["train_F"]
    dev, test = split_sets["dev_common"], split_sets["test_common"]
    check(len(split_sets["train_G"] & split_sets["train_F"]) == 101, "Train overlap differs")
    check(not (train & dev or train & test or dev & test) and train | dev | test == genes.keys(), "Split leakage/omission")
    verified_angles = 0
    retried_angles = 0
    with zipfile.ZipFile(base / "full_run_records.zip") as archive:
        archive_manifest = json.loads(archive.read("RUN_CHECKSUMS.json"))
        check(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate archive members")
        check(set(archive.namelist()) == set(archive_manifest) | {"RUN_CHECKSUMS.json"}, "Archive file list differs")
        print(f"Verifying {len(archive_manifest)} archived files...", flush=True)
        for name, expected in archive_manifest.items():
            check(sha(archive.read(name)) == expected, f"Archived file checksum mismatch: {name}")
        read = lambda name: json.loads(archive.read(name))
        contract = read("contract.json")
        check(canonical(contract) == summary["condition_fingerprint"], "Contract fingerprint mismatch")
        angles = contract["physical"]["inner_angles_deg"]
        check(angles == [29, 32, 35, 38, 41, 44], "Unexpected solve angles")
        for gid, gene in genes.items():
            bits = gene["bits"]
            check(len(bits) == 120 and set(bits) <= {"0", "1"}, f"Invalid bits: {gid}")
            check(sha(bytes(int(bits[i:i+8], 2) for i in range(0, 120, 8))) == gid, f"Gene hash mismatch: {gid}")
            label = indexed[gid]
            check(label["status"] == gene["status"] == "succeeded", f"Incomplete label: {gid}")
            check(set(grouped[gid]) == set(angles), f"Missing angles: {gid}")
            raw = [float(grouped[gid][a]["raw_torque_nm"]) for a in angles]
            check(all(math.isfinite(t) for t in raw), "Nonfinite torque")
            check(math.isclose(float(label["tavg_nm"]), sum(raw)/6, abs_tol=1e-12, rel_tol=1e-12), "Tavg differs")
            check(math.isclose(float(label["delta_t_nm"]), max(raw)-min(raw), abs_tol=1e-12, rel_tol=1e-12), "DeltaT differs")
            check(all(gene[k] == label[k] for k in ("tavg_nm", "delta_t_nm", "condition_fingerprint")), "Merged dataset differs")
            nested = read(f"{gid}/label.json")
            for angle in angles:
                folder = f"{gid}/angle_{angle:g}/"
                result_bytes = archive.read(folder + "result.json")
                result, state = json.loads(result_bytes), read(folder + "state.json")
                receipt_bytes = archive.read(folder + "artifact_retention.json")
                receipt = json.loads(receipt_bytes)
                check(state["status"] == "succeeded" and sha(result_bytes) == state["result_sha256"], "Result/state checksum differs")
                check(sha(receipt_bytes) == state["artifact_retention_sha256"], "Retention receipt checksum differs")
                for key in ("gene_id", "bits", "inner_angle_deg", "rotor_travel_deg", "currents_a", "condition_fingerprint", "prepared_sha256"):
                    check(result[key] == state[key], f"Result identity differs: {gid}/{angle}/{key}")
                check(result["gene_id"] == gid and result["bits"] == bits and result["inner_angle_deg"] == angle, "Angle identity differs")
                check(result["condition_fingerprint"] == summary["condition_fingerprint"], "Angle contract differs")
                check(receipt["result_sha256"] == state["result_sha256"] and receipt["policy"] == "results_only_v1" and receipt["model_settings_verified"] is True, "Invalid cleanup receipt")
                for key in ("fem_sha256", "ans_sha256", "gene_id", "inner_angle_deg", "condition_fingerprint"):
                    check(receipt[key] == result[key], "Cleanup identity differs")
                wave = grouped[gid][angle]
                check(result["raw_torque_nm"] == float(wave["raw_torque_nm"]), "Raw torque table differs")
                for phase in ("A", "B", "C"):
                    check(result["currents_a"][phase] == float(wave[phase]), "Current table differs")
                check(any(r == result for r in nested["angles"]), "Per-gene result differs from angle record")
                verified_angles += 1
                retried_angles += len(state["attempts"]) > 1
        check(not any(Path(name).suffix in (".fem", ".ans") for name in archive_manifest), "Completed work files remain")
    return {"status": "passed", "genes": len(genes), "angles": verified_angles,
            "retried_angles": retried_angles, "copied_files_verified": len(manifest),
            "archived_files_verified": len(archive_manifest), "split_counts": {k: len(v) for k, v in split_sets.items()}}


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    print(json.dumps(verify(folder), ensure_ascii=False, indent=2))
