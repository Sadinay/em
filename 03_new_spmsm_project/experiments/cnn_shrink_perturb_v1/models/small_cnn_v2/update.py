"""SmallCNN one-time Shrink-and-Perturb + v2 replay. Six runs; no test inference.
Training/audit/evaluation reused from frozen v2 logical runner; source_diff.json records unchanged functions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, replace
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys as _layout_sys
_LAYOUT_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'cnn_zone').is_dir())
_layout_sys.path.insert(0, str(_LAYOUT_PROJECT.parent / 'maintenance'))
from experiment_paths import audit_dir, report_dir, artifact_sha, relocated

import random
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT = _LAYOUT_PROJECT
VERSION = 'v2'
ALPHA = 0.95
BETA = 0.01
SP_SEED = 20261914
CONTROL = PROJECT / 'experiments/cnn_replay_update_v2/models/small_cnn_v2'
OLD_PER = 48 if VERSION == 'v1' else 56
NEW_PER = 64 - OLD_PER
MAX_UPDATES = 10000 if VERSION == 'v1' else 20000
ARCHITECTURES = ('small_cnn_v2',)
BASELINES = {
 'small_cnn_v2': ('5e50973958fcf8baad986e9b9f69879e1b766223674c8ef5e90aaad791f48126', 40, 'Logical6x20SmallCNNV2'),
 'mini_inception_v2': ('ecb2d1671d4725b1cff2fadac768b97210dc33b8d401f7b8e6654030d77ace75', 34, 'Logical6x20MiniInceptionV2'),
 'resnet20_v2': ('7ae9772b953a02a2e4b20bb9afde965b8e3ceab38d3d7838a6be4758f9a8054b', 77, 'Logical6x20ResNet20V2'),
}
ARCH = 'small_cnn_v2'
HERE = SOURCE_ROOT / 'alpha_095'
sys.path.insert(0, str(PROJECT))
from cnn_zone.src import training as tr
from cnn_zone.src.dataset import SPMSMGeneDataset
from femm_zone import femm_config
from femm_zone.scripts.spmsm_mapping import genotype_sha256

PILOT = PROJECT / "experiments/input_distribution_pilot_v1"
ACCEPTED = PILOT / "post_femm_baseline_20260913"
OLD = PROJECT / "data_zone/processed/spmsm_topology_dataset/training_corrected_binary"
MODEL = PROJECT / f"cnn_zone/models/v3_40000_6runs/logical6x20/{ARCH}/seed_20260903"
SPLIT = PROJECT / "cnn_zone/outputs/splits/scheme_a_tavg_bands_train40000_val6483_test6483.npz"
CHECKPOINT_SHA = BASELINES[ARCH][0]
PHYSICS_SHA = "52459cf11c89ee1589f4d3a11a7e86e92450ac14ea082037a94a6c86e8a7835f"
CONDITION = "97cd342fe5d68919dffee97fe5ff2d37bcfdbfd461eb357e9ca92f75ee05c712"
SEED = 20260914
TARGETS = tr.TARGET_NAMES
ROLES = ("old_train", "old_validation", "train_G", "train_F", "dev_common")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(relocated(path).read_text(encoding="utf-8"))


def save(path, value):
    tr.atomic_json(Path(path), value)


def sha(path):
    return artifact_sha(path)


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def csv_read(path):
    with relocated(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def old_identities():
    # Only the first three identity columns are decoded. Label columns are skipped.
    out = []
    with (OLD / "split_manifest.csv").open("rb") as stream:
        header = stream.readline().decode("utf-8-sig").split(",")[:3]
        require(header == ["sample_index", "topology_hash", "repair_component_id"], "Old identity schema changed")
        for line in stream:
            index, gid, component = line.split(b",", 3)[:3]
            out.append((int(index), gid.decode("ascii"), component.decode("ascii")))
    require([r[0] for r in out] == list(range(len(out))), "Old identity order changed")
    return out


def array_dataset(bits, targets):
    # Use exactly the existing __getitem__, with only allowlisted rows reachable.
    dataset = SPMSMGeneDataset.__new__(SPMSMGeneDataset)
    dataset.bits = np.asarray(bits, dtype=np.uint8)
    dataset.targets = np.asarray(targets, dtype=np.float32)
    return dataset


def load_data():
    with np.load(SPLIT) as z:
        split = {k: np.asarray(z[k], dtype=np.int64) for k in ("train", "validation", "test", "unused_train_pool")}
    identities = old_identities()
    grids = np.load(OLD / "topology_bits.npy", mmap_mode="r")
    labels = np.load(OLD / "targets_tavg_delta.npy", mmap_mode="r")
    require(grids.shape == (len(identities), 6, 20) and labels.shape == (len(identities), 2), "Old array shape changed")
    data, meta = {}, {}
    for role, key in (("old_train", "train"), ("old_validation", "validation")):
        indices = split[key]
        bits = np.array(grids[indices]).swapaxes(1, 2).reshape(-1, 120)
        target = np.array(labels[indices], dtype=np.float32)
        data[role] = array_dataset(bits, target)
        meta[role] = [{"gene_id": identities[i][1], "source": "old_GA", "old_index": int(i)} for i in indices]
    # The mapped old test target rows are never indexed, copied, or summarized.
    del grids, labels
    for role in ("train_G", "train_F", "dev_common"):
        directory = ACCEPTED / "data" / role
        source = SPMSMGeneDataset(directory)
        data[role] = array_dataset(np.array(source.bits), np.array(source.targets))
        meta[role] = csv_read(directory / "manifest.csv")
    return data, meta, split, identities


def allowed_dataset(data, role):
    require(role in ROLES, f"Sealed or unsupported dataset role: {role}")
    return data[role]


def settings(old_config):
    return {
        "experiment": f"cnn_shrink_perturb_v1/alpha_{round(ALPHA*100):03d}", "seed": SEED, "groups": ["G-S", "F-S"],
        "initialization": "one-time alpha*f0 + beta*natively initialized same-architecture parameters; independent fresh optimizer/scheduler/AMP",
        "sp": {"alpha": ALPHA, "beta": BETA, "random_seed": SP_SEED, "scope": "all trainable named parameters including biases and BN affine", "buffers": "inherit all f0 buffers exactly; no scaling/no noise/no reset", "timing": "once before first training update; never on resumed state", "randomness": "isolated CPU torch.random.fork_rng; native builder; shared direction across all coefficients/G/F", "beta_meaning": "multiplier of native random parameters, not Gaussian standard deviation"},
        "initial_checkpoint": str((MODEL / "best_checkpoint.pt").relative_to(PROJECT)),
        "initial_checkpoint_sha256": CHECKPOINT_SHA,
        "model_class": "cnn_zone.src.models_v2."+BASELINES[ARCH][2], "input_shape": [2, 6, 20],
        "baseline_config": old_config, "target_order": list(TARGETS),
        "target_mean": old_config["target_mean"], "target_std": old_config["target_std"],
        "target_scaler_policy": "frozen f0 old-training scaler; never recomputed",
        "loss": "mean squared error over two independently standardized targets",
        "optimizer": old_config["optimizer"], "initial_learning_rate": old_config["learning_rate"] * 0.1,
        "weight_decay": old_config["weight_decay"], "effective_batch_size": 64,
        "physical_batch_size": old_config["physical_batch_size"], "old_per_update": OLD_PER, "new_per_update": NEW_PER,
        "max_updates": MAX_UPDATES, "validate_every": 500, "checkpoint_every": 100,
        "fixed_budget": True, "gradient_accumulation_steps": 1, "improvement_epsilon": 1e-12,
        "old_mae_limit_multiplier": 1.05,
        "selection": "both old MAEs <= 1.05*f0; then minimize new-dev dual-target mean standardized MSE",
        "early_stop": "disabled; fixed user-approved 10000 (v1) / 20000 (v2) successful updates",
        "scheduler": {"class": "ReduceLROnPlateau", "monitor": "new_dev_standardized_mse", "factor": 0.5,
                      "patience": old_config["scheduler_patience"], "min_lr": 1e-6},
        "amp": "CUDA float16", "grad_scaler_init_scale": 128.0, "clip_grad_norm": 100.0,
        "train_all_parameters": True, "num_workers": 0, "cpu_threads": 4,
        "randomness": "shared shuffled old/new index plans; same initialization/seed/dropout RNG; deterministic algorithms warn_only like baseline; CUDA AMP/device nondeterminism remains possible",
        "settings_source": "user-approved logical extension: v1 10000/v2 20000; baseline physical batch64 retained for BatchNorm; own baseline learning rate x0.1; replay/selection rules retained",
        "test_policy": "test identities/membership only; no test target access or prediction",
    }


def audit():
    HERE.mkdir(parents=True, exist_ok=True)
    require(sha(MODEL / "best_checkpoint.pt") == CHECKPOINT_SHA, "Wrong f0 checkpoint")
    require(sha(femm_config.__file__) == PHYSICS_SHA, "FEMM configuration regressed")
    require(read(ACCEPTED / "data_audit.json")["status"] == "passed", "FEMM import not accepted")
    seed_validation = read(PILOT / "seed_validation.json")
    provenance = read(PILOT / "provenance.json")["input_sha256"]
    versions = {}
    # Hash the source/renderer/model/lookup/split inputs; do not scan old target
    # bytes containing sealed test labels. Record accepted full hash and fresh allowed-slice hashes instead.
    for name, expected in provenance.items():
        normalized = name.replace("\\", "/")
        if normalized.startswith("cnn_zone/src/") or normalized.endswith(("config.json", "best_checkpoint.pt")) or "/lookups/" in normalized or "/splits/" in normalized or normalized == "femm_zone/femm_config.py":
            path = PROJECT / name
            require(sha(path) == expected, f"Frozen dependency changed: {name}")
            versions[normalized] = expected
    for name in ("train_G.csv", "train_F.csv", "dev_common.csv", "memberships.csv"):
        actual = sha(PILOT / name)
        require(actual == seed_validation["file_sha256"][name], f"Frozen membership changed: {name}")
        versions[str((PILOT / name).relative_to(PROJECT))] = actual
    output_checksums = read(ACCEPTED / "OUTPUT_CHECKSUMS.json")
    for name, expected in output_checksums.items():
        if name.startswith(("data/train_G/", "data/train_F/", "data/dev_common/")) or name in ("data/labels_train_dev.csv", "data/waveforms_train_dev.csv"):
            require(sha(ACCEPTED / name) == expected, f"Accepted training/dev output changed: {name}")
            versions[str((ACCEPTED / name).relative_to(PROJECT))] = expected
    old_config = read(MODEL / "config.json")
    checkpoint = torch.load(MODEL / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    require(checkpoint["config"] == old_config and checkpoint["epoch"] == BASELINES[ARCH][1], "Checkpoint/config identity mismatch")
    for key in ("target_mean", "target_std"):
        require(np.array_equal(checkpoint[key].numpy(), np.asarray(old_config[key], dtype=np.float32)), f"Frozen {key} differs")
    require(old_config["architecture"] == ARCH and old_config["input_mode"] == "logical6x20" and
            old_config["optimizer"] == "AdamW" and old_config["physical_batch_size"] == 64, "Unexpected baseline implementation")
    data, meta, split, identities = load_data()
    counts = {k: len(v) for k, v in split.items()}
    require(all(counts[k] == n for k, n in (("train", 40000), ("validation", 6483), ("test", 6483), ("unused_train_pool", 11838))), "Old split counts changed")
    all_indices = np.concatenate(list(split.values()))
    require(len(np.unique(all_indices)) == len(all_indices) == len(identities), "Old split overlap/missing identity")
    old_ids = [x[1] for x in identities]
    require(len(set(old_ids)) == len(old_ids), "Duplicate old identities")
    component_sets = {k: {identities[i][2] for i in split[k]} for k in ("train", "validation", "test")}
    require(not component_sets["train"] & (component_sets["validation"] | component_sets["test"]) and
            not component_sets["validation"] & component_sets["test"], "Old repair family leakage")
    membership = csv_read(PILOT / "memberships.csv")  # ID, role, group only: no labels.
    test_ids = {r["gene_id"] for r in membership if r["split_role"] == "test"}
    ids = {role: {r["gene_id"] for r in rows} for role, rows in meta.items()}
    require(len(test_ids) == 400, "Sealed membership count changed")
    require(not (ids["train_G"] | ids["train_F"]) & (ids["dev_common"] | test_ids) and not ids["dev_common"] & test_ids, "New training/holdout leakage")
    new_ids = ids["train_G"] | ids["train_F"] | ids["dev_common"] | test_ids
    require(not new_ids & set(old_ids), "New and old identities intersect")
    require(len(ids["train_G"] & ids["train_F"]) == 101 and len(new_ids) == 3299, "New intersection/count mismatch")
    label_rows = csv_read(ACCEPTED / "data/labels_train_dev.csv")
    canonical = {r["gene_id"]: r for r in label_rows}
    require(len(canonical) == len(label_rows) == 2899 and not set(canonical) & test_ids, "Train/dev canonical identity conflict")
    slice_hashes = {}
    for role in ROLES:
        ds, rows = data[role], meta[role]
        require(len(ds) == len(rows) == len(ids[role]), f"Duplicate {role} records")
        require(np.isin(ds.bits, [0, 1]).all() and np.isfinite(ds.targets).all() and (ds.targets[:, 1] >= 0).all(), f"Invalid bits/labels in {role}")
        require(all(genotype_sha256(bits) == row["gene_id"] for bits, row in zip(ds.bits, rows)), f"Gene encoding mismatch {role}")
        if role.startswith("old_"):
            slice_hashes[role] = {"bits_sha256": array_sha(ds.bits), "target_float32_sha256": array_sha(ds.targets)}
        else:
            expected_role = "dev" if role == "dev_common" else "train"
            seed_rows = csv_read(PILOT / (role + ".csv"))
            require([r["gene_id"] for r in seed_rows] == [r["gene_id"] for r in rows], f"Selection order changed {role}")
            for row, seed_row, target in zip(rows, seed_rows, ds.targets):
                require(all(row[k] == v for k, v in seed_row.items()), f"Seed/repair metadata changed {role}")
                require(row["status"] == "verified" and row["condition_fingerprint"] == CONDITION and row["split_role"] == expected_role, "Unaccepted label/version")
                c = canonical[row["gene_id"]]
                require(np.array_equal(target, np.array([c["tavg_nm"], c["delta_t_nm"]], dtype=np.float32)) and
                        all(row[k] == c[k] for k in ("bits", "tavg_nm", "delta_t_nm", "condition_fingerprint")), "Duplicate label conflict")
            wanted = {r["gene_id"] for r in membership if r["split_role"] == expected_role and
                      (role == "dev_common" or r["experiment_group"] == role[-1] + "-S")}
            require(wanted == ids[role], f"Membership mismatch {role}")
    require([len(data[r]) for r in ("train_G", "train_F", "dev_common")] == [1400, 1400, 200], "Accepted view count mismatch")
    summary = read(OLD / "dataset_summary.json")
    cfg = settings(old_config)
    report = {"status": "passed", "counts": {r: len(data[r]) for r in ROLES}, "sealed_new_test_ids": len(test_ids),
              "old_split_counts": counts, "gf_intersection": 101, "new_training_union": 2699, "unique_new": 3299,
              "all_memberships_disjoint_except_gf_train": True, "old_repair_families_disjoint": True,
              "labels_finite_and_conflict_free": True, "condition_fingerprint": CONDITION,
              "physical_definition": "Tavg = mean of 6 torques; DeltaT = max-min of 6 torques; both N*m; 29:3:44 inner angle; multiplier 1",
              "old_exception_policy": summary["integrity"]["pm_count_mismatch_policy"],
              "old_dataset_exceptions_retained": summary["integrity"]["pm_count_volume_pm_mismatches"],
              "old_deduplication": summary["deduplication"], "samples_removed_or_relabeled_this_run": 0,
              "fresh_old_allowed_slice_hashes": slice_hashes,
              "old_full_target_hash_from_accepted_provenance_only": next(v for k, v in provenance.items() if k.endswith("targets_tavg_delta.npy")),
              "test_target_rows_accessed": 0, "new_test_files_opened": 0,
              "test_identity_read_method": "new: memberships.csv only; old: split indices and first 3 ID columns only; old targets mmap indexed solely by train/validation",
              "versions": versions, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT).decode().strip(),
              "checkpoint_epoch": BASELINES[ARCH][1], "checkpoint_sha256": CHECKPOINT_SHA, "source_sha256": sha(__file__)}
    for path, value in ((HERE / "config.json", cfg), (audit_dir(HERE) / 'data_audit.json', report)):
        if path.exists():
            require(read(path) == value, f"Frozen experiment changed; use a new directory: {path}")
        else:
            save(path, value)
    print("AUDIT passed: 40000 old + 1400 per group, old val 6483, dev 200; test IDs only", flush=True)
    return cfg, data, meta, report


def cycle_plan(count, per_step, max_steps, seed):
    rng = np.random.default_rng(seed)
    chunks, needed = [], per_step * max_steps
    while needed:
        part = rng.permutation(count)[:min(count, needed)].astype(np.int32)
        chunks.append(part)
        needed -= len(part)
    return np.concatenate(chunks).reshape(max_steps, per_step)


class ReplaySampler(Sampler):
    def __init__(self, old_plan, new_plan, start, stop, old_count=40000):
        self.old_plan, self.new_plan = old_plan, new_plan
        self.start, self.stop, self.old_count = start, stop, old_count

    def __iter__(self):
        for step in range(self.start, self.stop):
            yield np.concatenate((self.old_plan[step], self.new_plan[step] + self.old_count)).tolist()

    def __len__(self):
        return self.stop - self.start


def plans(cfg):
    directory = audit_dir(HERE)
    values = [cycle_plan(40000, cfg["old_per_update"], cfg["max_updates"], SEED+1), cycle_plan(1400, cfg["new_per_update"], cfg["max_updates"], SEED+2)]
    for name, value in zip(("old_replay_indices.npy", "new_replay_indices.npy"), values):
        path = directory / name
        if path.exists():
            require(np.array_equal(np.load(path), value), "Replay plan changed")
        else:
            np.save(path, value)
    return values


def initialize(cfg, perturb=False):
    require(torch.cuda.is_available(), "CUDA training device unavailable")
    torch.set_num_threads(cfg["cpu_threads"])
    tr.configure_reproducibility(SEED)
    device = torch.device("cuda")
    old = cfg["baseline_config"]
    spec = tr.TrainingSpec(**{f.name: old[f.name] for f in fields(tr.TrainingSpec)})
    spec = replace(spec, learning_rate=cfg["initial_learning_rate"])
    ckpt = torch.load(MODEL / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    require(sha(MODEL / "best_checkpoint.pt") == CHECKPOINT_SHA, "f0 changed")
    model = tr.build_model(spec).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state"], strict=True)
    require(all(torch.equal(v.detach().cpu(), ckpt["model_state"][k]) for k, v in model.state_dict().items()), "Weights not exactly f0")
    model.requires_grad_(True)
    renderer = tr.build_renderer(spec, PROJECT, device)
    require(renderer is None and type(model).__name__ == BASELINES[ARCH][2], "Logical input/model class mismatch")
    if perturb:
        model._sp_info = apply_sp(model, spec, cfg["sp"]["alpha"], cfg["sp"]["beta"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=spec.scheduler_patience, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    return model, renderer, spec, ckpt["target_mean"], ckpt["target_std"], device, optimizer, scheduler, scaler


def evaluation(model, renderer, spec, mean, std, device, data, meta, directory=None):
    result = {}
    for role in ("old_validation", "dev_common"):
        ds = allowed_dataset(data, role)
        loss, _, indices, actual, predicted = tr.evaluate(model, ds, np.arange(len(ds)), spec, renderer, mean, std, device)
        require(np.isfinite(predicted).all() and math.isfinite(loss), "Non-finite validation prediction")
        require(np.array_equal(indices, np.arange(len(ds))), "Validation order changed")
        truth = actual.astype(np.float64) if role == "old_validation" else np.array([[r["tavg_nm"], r["delta_t_nm"]] for r in meta[role]], dtype=np.float64)
        metrics = tr.regression_metrics(truth, predicted.astype(np.float64))
        result[role] = {"n": len(ds), "standardized_mse": loss, "metrics": metrics}
        if role == "dev_common":
            result[role]["sources"] = {}
            for source in "ULBP":
                mask = np.array([r["source"] == source for r in meta[role]])
                result[role]["sources"][source] = {"n": int(mask.sum()), "metrics": tr.regression_metrics(truth[mask], predicted[mask].astype(np.float64))}
        if directory is not None:
            rows = []
            for item, y, p in zip(meta[role], truth, predicted):
                row = {"gene_id": item["gene_id"], "source": item["source"]}
                for j, name in enumerate(TARGETS):
                    row.update({name+"_true_nm": float(y[j]), name+"_pred_nm": float(p[j]), name+"_error_nm": float(p[j]-y[j])})
                rows.append(row)
            csv_write(Path(directory) / (role + "_predictions.csv"), rows)
    if directory is not None:
        save(Path(directory) / "metrics.json", result)
    return result


def admissible(value, baseline, multiplier=1.05):
    return all(value["old_validation"]["metrics"][t]["mae"] <= multiplier * baseline["old_validation"]["metrics"][t]["mae"] for t in TARGETS)


def improvement_eligible(value, baseline, multiplier=1.05, epsilon=1e-12):
    return admissible(value, baseline, multiplier) and value["dev_common"]["standardized_mse"] < baseline["dev_common"]["standardized_mse"] - epsilon


def perform_update(model, renderer, spec, mean, std, device, optimizer, scaler, batches):
    model.train()
    mean, std = mean.to(device), std.to(device)
    attempts = 0
    while True:
        buffers = {k:v.detach().clone() for k,v in model.named_buffers()}
        optimizer.zero_grad(set_to_none=True)
        sums = np.zeros(3, dtype=np.float64)
        for _, bits, target in batches:
            bits, target = bits.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = model(tr.make_inputs(bits, spec, renderer))
                errors = F.mse_loss(prediction, (target-mean)/std, reduction="none")
                loss_sum = errors.sum()
                loss = loss_sum / (64*2)
            require(bool(torch.isfinite(loss_sum)), "Non-finite training loss")
            scaler.scale(loss).backward()
            sums += [float(loss_sum.detach()), float(errors[:OLD_PER].sum().detach()), float(errors[OLD_PER:].sum().detach())]
        previous_scale = scaler.get_scale()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        scaler.step(optimizer)
        scaler.update()
        attempts += 1
        if scaler.get_scale() >= previous_scale:
            require(bool(torch.isfinite(norm)), "Invalid gradient update")
            return {"mse": sums[0]/128, "old_mse": sums[1]/(OLD_PER*2), "new_mse": sums[2]/(NEW_PER*2),
                    "attempts": attempts, "grad_norm": float(norm), "amp_scale": scaler.get_scale()}
        with torch.no_grad():
            for k, value in model.named_buffers():
                value.copy_(buffers[k])
        require(attempts < 20, "Repeated AMP overflow; abort without changing training settings")


def pack_state(model, optimizer, scheduler, scaler, mean, std, cfg, group, step, history, trace, elapsed, best, no_improvement, sampled):
    return {"group": group, "step": step, "seed": SEED, "config": cfg, "config_sha256": digest(cfg),
            "audit_sha256": sha(audit_dir(HERE) / 'data_audit.json'), "script_sha256": sha(__file__),
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "target_mean": mean, "target_std": std, "history": history, "trace": trace,
            "elapsed_seconds": elapsed, "best": best, "no_improvement": no_improvement, "sampled": sampled,
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state()}


def restore(state, cfg, group, model, optimizer, scheduler, scaler):
    require(state["group"] == group and state["config_sha256"] == digest(cfg) and state["seed"] == SEED,
            "Resume experiment identity mismatch")
    require(state["script_sha256"] == sha(__file__) and state["audit_sha256"] == sha(audit_dir(HERE) / 'data_audit.json'), "Resume source/data changed")
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng_states"]])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])


def preflight():
    cfg, data, meta, audit_report = audit()
    directory = audit_dir(HERE) / 'preflight'
    directory.mkdir(parents=True, exist_ok=True)
    old_plan, new_plan = plans(cfg)
    model, renderer, spec, mean, std, device, opt, scheduler, scaler = initialize(cfg)
    mixed = array_dataset(np.concatenate((data["old_train"].bits, data["train_G"].bits)), np.concatenate((data["old_train"].targets, data["train_G"].targets)))
    loader = DataLoader(mixed, batch_sampler=ReplaySampler(old_plan, new_plan, 0, 2), num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(SEED+3))
    batches = list(loader)
    require(len(batches) == 2 and all(int((b[0] < 40000).sum()) == OLD_PER for b in batches), "Replay microbatch ratio failed")
    inputs = tr.make_inputs(batches[0][1].to(device), spec, renderer)
    require(tuple(inputs.shape) == (64, 2, 6, 20), "Input encoding shape failed")
    sample_target = batches[0][2]
    require(torch.allclose(((sample_target-mean)/std)*std+mean, sample_target, atol=3e-7, rtol=0), "Target roundtrip failed")
    for role in ("test", "test_common", "old_test", "sealed_test"):
        try:
            allowed_dataset(data, role)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Test access guard failed")
    tiny_data = {r: array_dataset(data[r].bits[:8], data[r].targets[:8]) for r in ("old_validation", "dev_common")}
    tiny_meta = {r: meta[r][:8] for r in tiny_data}
    # Use the normal evaluator on all four dev sources for the small check.
    tiny_data["dev_common"] = data["dev_common"]
    tiny_meta["dev_common"] = meta["dev_common"]
    before = evaluation(model, renderer, spec, mean, std, device, tiny_data, tiny_meta)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    first = perform_update(model, renderer, spec, mean, std, device, opt, scaler, batches[:1])
    scheduler.step(before["dev_common"]["standardized_mse"])
    state = pack_state(model, opt, scheduler, scaler, mean, std, cfg, "preflight", 1, [], [], 0, {}, 0, {})
    tr.atomic_torch_save(directory / "roundtrip_checkpoint.pt", state)
    continuous = perform_update(model, renderer, spec, mean, std, device, opt, scaler, batches[1:])
    reference = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    saved = torch.load(directory / "roundtrip_checkpoint.pt", map_location=device, weights_only=False)
    restore(saved, cfg, "preflight", model, opt, scheduler, scaler)
    resumed = perform_update(model, renderer, spec, mean, std, device, opt, scaler, batches[1:])
    require(all(torch.equal(v.detach().cpu(), reference[k]) for k, v in model.state_dict().items()), "Checkpoint/RNG resume changed the next update")
    require(continuous == resumed, "Resume loss/AMP state differs")
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3
    del model, renderer, opt, scheduler, scaler, saved, state, reference, inputs
    gc.collect(); torch.cuda.empty_cache()
    # Fresh initialization and full step-zero evaluation after discarding probe weights.
    model, renderer, spec, mean, std, device, opt, scheduler, scaler = initialize(cfg)
    model.eval().requires_grad_(False)
    baseline = evaluation(model, renderer, spec, mean, std, device, data, meta, HERE / "baseline")
    repeated = evaluation(model, renderer, spec, mean, std, device, tiny_data, tiny_meta)
    require(repeated == before, "Fresh f0 after probe differs from initial f0")
    max_diff = 0.0
    other = _LAYOUT_PROJECT / "experiments/cnn_replay_update_v1/models" / ARCH / "baseline/metrics.json"
    if VERSION == "v2" and other.exists():
        require(read(other) == baseline, "v1/v2 own frozen baseline predictions differ")
    save(audit_dir(HERE) / 'preflight.json', {"status": "passed", "input_shape": [64,2,6,20], "replay": [OLD_PER,NEW_PER],
         "checkpoint_next_update_bit_exact": True, "scaler_roundtrip": True, "test_guards_passed": True,
         "probe_seconds_for_three_updates_and_checkpoint_roundtrip": seconds, "peak_allocated_gib": peak,
         "first_update": first, "max_step0_prediction_difference_nm": max_diff,
         "formal_weights_reset_to_f0": True, "baseline": baseline, "gpu": torch.cuda.get_device_name(0),
         "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "script_sha256": sha(__file__),
         "config_sha256": digest(cfg), "audit_sha256": sha(audit_dir(HERE) / 'data_audit.json')})
    sp_preflight(cfg, data, meta)
    # The successful resume probe checkpoint is disposable, not a training start.
    (directory / "roundtrip_checkpoint.pt").unlink()
    print(f"PREFLIGHT passed; f0 max prediction difference={max_diff:g}; peak={peak:.2f} GiB", flush=True)


def save_weights(path, model, mean, std, cfg, group, step, metrics):
    tr.atomic_torch_save(path, {"model_state": model.state_dict(), "target_mean": mean, "target_std": std,
                         "config": cfg, "group": group, "step": step, "seed": SEED, "metrics": metrics,
                         "f0_sha256": CHECKPOINT_SHA, "audit_sha256": sha(audit_dir(HERE) / 'data_audit.json')})


def train_group(group, cfg, data, meta, resume):
    directory = HERE / "runs" / group
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        require(resume, "Completed run exists; --resume skips it without overwriting")
        return read(directory / "result.json")
    last = directory / "last_checkpoint.pt"
    require(resume or not last.exists(), "Existing run requires --resume")
    model, renderer, spec, mean, std, device, optimizer, scheduler, scaler = initialize(cfg, perturb=not last.exists())
    old_plan, new_plan = plans(cfg)
    new_role = "train_" + group[0]
    mixed = array_dataset(np.concatenate((data["old_train"].bits, data[new_role].bits)),
                          np.concatenate((data["old_train"].targets, data[new_role].targets)))
    baseline = read(HERE / "baseline/metrics.json")
    history, trace, step, no_improvement, elapsed_before = [], [], 0, 0, 0.0
    best = {"feasible_score": None, "feasible_step": None, "unconstrained_score": None, "unconstrained_step": None,
            "constrained_incumbent": baseline["dev_common"]["standardized_mse"]}
    sampled = {"old_draws": 0, "new_draws": 0, "amp_skipped_attempts": 0}
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        restore(state, cfg, group, model, optimizer, scheduler, scaler)
        history, trace, step = state["history"], state["trace"], state["step"]
        best, no_improvement, elapsed_before, sampled = state["best"], state["no_improvement"], state["elapsed_seconds"], state["sampled"]
        del state
        print(f"RESUME {group} step={step}", flush=True)
    else:
        initial = evaluation(model, renderer, spec, mean, std, device, data, meta, directory / "step00000")
        require(model._sp_info["applications"] == 1, "SP must run exactly once before training")
        history.append({"step": 0, "elapsed_seconds": 0, "learning_rate": cfg["initial_learning_rate"],
                        "feasible": admissible(initial, baseline), "metrics": initial})
        save(directory / "initialization.json", {"f0_sha256": CHECKPOINT_SHA, "weights_equal_f0": False, "sp": model._sp_info,
             "new_optimizer_state_empty": len(optimizer.state) == 0, "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
             "seed": SEED, "old_plan_sha256": sha(audit_dir(HERE) / 'old_replay_indices.npy'),
             "new_plan_sha256": sha(audit_dir(HERE) / 'new_replay_indices.npy')})
    started = time.perf_counter()
    stop_reason = f"fixed_{cfg['max_updates']}_successful_updates"
    try:
        while step < cfg["max_updates"]:
            stop = min(((step // cfg["validate_every"]) + 1) * cfg["validate_every"], cfg["max_updates"])
            loader = iter(DataLoader(mixed, batch_sampler=ReplaySampler(old_plan, new_plan, step, stop), num_workers=0, pin_memory=True,
                                     generator=torch.Generator().manual_seed(SEED+3)))
            while step < stop:
                batches = [next(loader)]
                tick = time.perf_counter()
                lr = optimizer.param_groups[0]["lr"]
                value = perform_update(model, renderer, spec, mean, std, device, optimizer, scaler, batches)
                step += 1
                sampled["old_draws"] += cfg["old_per_update"] * value["attempts"]
                sampled["new_draws"] += cfg["new_per_update"] * value["attempts"]
                sampled["amp_skipped_attempts"] += value["attempts"] - 1
                trace.append({"step": step, "learning_rate": lr, "seconds": time.perf_counter()-tick, **value})
                if step % 100 == 0:
                    print(f"{group} step={step} loss={value['mse']:.5f} old={value['old_mse']:.5f} new={value['new_mse']:.5f} lr={lr:.2g}", flush=True)
                if step % cfg["checkpoint_every"] == 0 and step != stop:
                    elapsed = elapsed_before + time.perf_counter() - started
                    tr.atomic_torch_save(last, pack_state(model, optimizer, scheduler, scaler, mean, std, cfg, group, step, history, trace, elapsed, best, no_improvement, sampled))
                    save(directory / "progress.json", {"status": "running", "step": step, "elapsed_seconds": elapsed, "sampled": sampled, "last_validation": history[-1]})
            elapsed = elapsed_before + time.perf_counter() - started
            metrics = evaluation(model, renderer, spec, mean, std, device, data, meta, directory / f"step{step:05d}")
            feasible = admissible(metrics, baseline, cfg["old_mae_limit_multiplier"])
            score = metrics["dev_common"]["standardized_mse"]
            if best["unconstrained_score"] is None or score < best["unconstrained_score"] - cfg["improvement_epsilon"]:
                best.update(unconstrained_score=score, unconstrained_step=step)
                save_weights(directory / "best_unconstrained.pt", model, mean, std, cfg, group, step, metrics)
            if improvement_eligible(metrics, baseline, cfg["old_mae_limit_multiplier"], cfg["improvement_epsilon"]) and (best["feasible_score"] is None or score < best["feasible_score"] - cfg["improvement_epsilon"]):
                best.update(feasible_score=score, feasible_step=step)
                save_weights(directory / "best_feasible.pt", model, mean, std, cfg, group, step, metrics)
            improved = feasible and score < best["constrained_incumbent"] - cfg["improvement_epsilon"]
            if improved:
                best["constrained_incumbent"] = score
                no_improvement = 0
            else:
                no_improvement += 1
            scheduler.step(score)
            history.append({"step": step, "elapsed_seconds": elapsed_before+time.perf_counter()-started,
                            "learning_rate": optimizer.param_groups[0]["lr"], "feasible": feasible,
                            "improved_constrained_incumbent": improved, "no_improvement": no_improvement,
                            "metrics": metrics, "sampled": dict(sampled),
                            "old_unique_covered": int(len(np.unique(old_plan[:step]))),
                            "new_unique_covered": int(len(np.unique(new_plan[:step])))})
            save(directory / "history.json", history)
            csv_write(directory / "training_trace.csv", trace)
            tr.atomic_torch_save(last, pack_state(model, optimizer, scheduler, scaler, mean, std, cfg, group, step, history, trace,
                                elapsed_before+time.perf_counter()-started, best, no_improvement, sampled))
            save(directory / "progress.json", {"status": "running", "step": step, "elapsed_seconds": history[-1]["elapsed_seconds"], "sampled": sampled, "last_validation": history[-1]})
            print(f"VALID {group} step={step} feasible={feasible} oldMAE={[metrics['old_validation']['metrics'][t]['mae'] for t in TARGETS]} newMSE={score:.6g} no_improvement={no_improvement}", flush=True)
        require(step == cfg["max_updates"], "Incomplete fixed-budget run cannot be marked complete")
        elapsed = elapsed_before + time.perf_counter() - started
        success = best["feasible_score"] is not None and best["feasible_score"] < baseline["dev_common"]["standardized_mse"] - cfg["improvement_epsilon"]
        result = {"status": "complete", "group": group, "stop_reason": stop_reason, "updates": step,
                  "elapsed_seconds": elapsed, "best": best, "success": success,
                  "conclusion": "found constrained improvement" if success else "本轮未找到满足条件的改善模型",
                  "selected_step": best["feasible_step"] if success else 0,
                  "selected_checkpoint": str((directory / "best_feasible.pt").relative_to(HERE)) if success else str((MODEL / "best_checkpoint.pt").relative_to(PROJECT)),
                  "fallback_to_f0": not success, "final_checkpoint": "last_checkpoint.pt", "final_checkpoint_step": step, "sampled": sampled,
                  "old_unique_covered": int(len(np.unique(old_plan[:step]))),
                  "new_unique_covered": int(len(np.unique(new_plan[:step]))),
                  "successful_update_old_draws": step*cfg["old_per_update"], "successful_update_new_draws": step*cfg["new_per_update"],
                  "final_learning_rate": optimizer.param_groups[0]["lr"], "test_predictions": 0,
                  "checkpoint_sha256": {p.name: sha(p) for p in directory.glob("*.pt")}}
        save(directory / "result.json", result)
        save(directory / "progress.json", result)
        print(f"DONE {group}: {result['conclusion']}; step={step}; elapsed={elapsed/60:.1f} min", flush=True)
        return result
    except BaseException as exc:
        save(directory / "interruption.json", {"error": repr(exc), "last_completed_in_memory_step": step,
             "resume": "train --resume loads the last atomic checkpoint (at most 100 completed steps earlier); partial progress is replayed"})
        raise
    finally:
        del model, renderer, optimizer, scheduler, scaler
        gc.collect(); torch.cuda.empty_cache()


def train(resume=False):
    cfg, data, meta, _ = audit()
    probe = read(audit_dir(HERE) / 'preflight.json')
    require(probe["status"] == "passed" and probe["script_sha256"] == sha(__file__) and probe["config_sha256"] == digest(cfg) and
            probe["audit_sha256"] == sha(audit_dir(HERE) / 'data_audit.json'), "Run preflight for the current source/config first")
    lock = PROJECT / "experiments/logical6x20_training.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as stream:
        json.dump({"pid": os.getpid(), "started": time.time()}, stream)
    try:
        (HERE / "source_snapshot").mkdir(exist_ok=True)
        snapshot = HERE / "source_snapshot/update.py"
        if snapshot.exists():
            require(sha(snapshot) == sha(__file__), "Frozen training source differs")
        else:
            shutil.copyfile(__file__, snapshot)
        save(audit_dir(HERE) / 'runtime.json', {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
             "gpu": torch.cuda.get_device_name(0), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "source_sha256": sha(__file__), "process_pid": os.getpid()})
        for group in cfg["groups"]:
            train_group(group, cfg, data, meta, resume)
        require(sha(MODEL / "best_checkpoint.pt") == CHECKPOINT_SHA, "Frozen f0 was modified")
    finally:
        lock.unlink(missing_ok=True)




def tensor_digest(items):
    h = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def apply_sp(model, spec, alpha, beta):
    require(not getattr(model, '_sp_applied', False), 'SP cannot be applied twice to one model')
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [s.clone() for s in torch.cuda.get_rng_state_all()]
    before_buffers = {k:v.detach().clone() for k,v in model.named_buffers()}
    before = {k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
    # Native construction is CPU-only; fork restores its entire RNG consumption.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SP_SEED)
        random_model = tr.build_model(spec)
    random_params = dict(random_model.named_parameters())
    require(before.keys() == random_params.keys(), 'Named parameter mismatch')
    direction_hash = tensor_digest(random_model.named_parameters())
    stats = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            q = random_params[name].to(device=p.device, dtype=p.dtype)
            expected = before[name]*alpha + q*beta
            p.copy_(expected)
            require(torch.equal(p, expected), 'SP formula mismatch: '+name)
            old_norm = float(torch.linalg.vector_norm(before[name].double()))
            change = float(torch.linalg.vector_norm((p-before[name]).double()))
            stats.append({'name':name, 'numel':p.numel(), 'f0_l2':old_norm,
                          'random_l2':float(torch.linalg.vector_norm(q.double())),
                          'delta_l2':change, 'relative_delta_l2':change/old_norm if old_norm else None})
    require(all(torch.equal(v, before_buffers[k]) for k,v in model.named_buffers()), 'SP changed a buffer')
    require(torch.equal(cpu_rng, torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng, torch.cuda.get_rng_state_all())), 'SP consumed training RNG')
    model._sp_applied = True
    del random_model
    return {'alpha':alpha, 'beta':beta, 'random_seed':SP_SEED, 'applications':1,
            'f0_parameter_sha256':tensor_digest(before.items()), 'random_parameter_sha256':direction_hash,
            'start_parameter_sha256':tensor_digest(model.named_parameters()),
            'buffers_sha256':tensor_digest(model.named_buffers()), 'buffers_exactly_f0':True,
            'training_rng_unchanged':True, 'cpu_rng_sha256':array_sha(cpu_rng.numpy()),
            'cuda_rng_sha256':[array_sha(s.cpu().numpy()) for s in cuda_rng],
            'delta_l2':math.sqrt(sum(s['delta_l2']**2 for s in stats)),
            'relative_delta_l2':math.sqrt(sum(s['delta_l2']**2 for s in stats)/sum(s['f0_l2']**2 for s in stats)),
            'parameter_count':sum(s['numel'] for s in stats), 'by_parameter':stats}


def sp_preflight(cfg, data, meta):
    control_cfg = read(CONTROL/'config.json')
    excluded = {'experiment','initialization','sp'}
    require({k:v for k,v in cfg.items() if k not in excluded} == {k:v for k,v in control_cfg.items() if k not in excluded}, 'v2 training settings differ')
    require(read(HERE/'baseline/metrics.json') == read(CONTROL/'baseline/metrics.json'), 'Original f0 full evaluation differs')
    for name in ('old_replay_indices.npy','new_replay_indices.npy'):
        require(sha(audit_dir(HERE)/name) == sha(audit_dir(CONTROL)/name), 'Control sampling plan differs')
    runtime=read(audit_dir(CONTROL) / 'runtime.json')
    require(runtime['torch']==torch.__version__ and runtime['cuda']==torch.version.cuda and runtime['gpu']==torch.cuda.get_device_name(0), 'Control runtime differs')
    values=initialize(cfg)
    model,renderer,spec,mean,std,device,opt,scheduler,scaler=values
    f0 = {k:v.clone() for k,v in model.state_dict().items()}
    identity=apply_sp(model,spec,1.0,0.0)
    require(all(torch.equal(v,f0[k]) for k,v in model.state_dict().items()), 'alpha=1,beta=0 is not identity')
    try:
        apply_sp(model,spec,ALPHA,BETA)
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Repeated SP guard failed')
    old_plan,new_plan=plans(cfg)
    mixed=array_dataset(np.concatenate((data['old_train'].bits,data['train_G'].bits)),np.concatenate((data['old_train'].targets,data['train_G'].targets)))
    batches=list(DataLoader(mixed,batch_sampler=ReplaySampler(old_plan,new_plan,0,2),num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(SEED+3)))
    # v2 calls full step-zero evaluation before creating its training loader.
    # Mirror that order in both arms to verify Dropout and evaluator RNG parity.
    evaluation(model,renderer,spec,mean,std,device,data,meta)
    first=perform_update(model,renderer,spec,mean,std,device,opt,scaler,batches[:1])
    expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model,renderer,spec,mean,std,device,opt,scheduler,scaler=initialize(cfg)
    evaluation(model,renderer,spec,mean,std,device,data,meta)
    other=perform_update(model,renderer,spec,mean,std,device,opt,scaler,batches[:1])
    require(first==other and all(torch.equal(v.detach().cpu(),expected[k]) for k,v in model.state_dict().items()), 'Identity SP alters next update')
    model,renderer,spec,mean,std,device,opt,scheduler,scaler=initialize(cfg,perturb=True)
    info=model._sp_info
    require(info['random_parameter_sha256']==identity['random_parameter_sha256'], 'Random direction differs')
    perform_update(model,renderer,spec,mean,std,device,opt,scaler,batches[:1])
    state=pack_state(model,opt,scheduler,scaler,mean,std,cfg,'sp_probe',1,[],[],0,{},0,{})
    path=audit_dir(HERE) / 'preflight/sp_roundtrip.pt'
    tr.atomic_torch_save(path,state)
    continuous=perform_update(model,renderer,spec,mean,std,device,opt,scaler,batches[1:])
    expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # initialize without SP for resume, then restore all weights/states/RNG.
    model,renderer,spec,mean,std,device,opt,scheduler,scaler=initialize(cfg)
    restore(torch.load(path,map_location=device,weights_only=False),cfg,'sp_probe',model,opt,scheduler,scaler)
    resumed=perform_update(model,renderer,spec,mean,std,device,opt,scaler,batches[1:])
    require(continuous==resumed and all(torch.equal(v.detach().cpu(),expected[k]) for k,v in model.state_dict().items()), 'SP resume mismatch')
    path.unlink()
    save(audit_dir(HERE) / 'sp_preflight.json',{'passed':True,'identity_parameters_and_next_update_exact':True,
        'sp_applied_once_guard':True,'sp_resume_next_update_exact':True,'control_config_equal_except_sp_identity':True,
        'control_runtime_equal':True,'control_sampling_plans_exact':True,'control_f0_full_validation_exact':True,
        'reuse_existing_no_sp_control':True,'parameter_buffer_checks':info,'source_sha256':sha(__file__),'test_used':False})
    print('SP PREFLIGHT passed; native direction shared; identity next update exact; v2 control reused',flush=True)


def suite(resume=False):
    logs=SOURCE_ROOT/'logs';logs.mkdir(exist_ok=True)
    for alpha in (.95,.80,.50):
        label=f'alpha_{round(alpha*100):03d}'
        for command in ('preflight','train'):
            if command=='preflight' and (audit_dir(SOURCE_ROOT/label) / 'sp_preflight.json').exists():
                require(read(audit_dir(SOURCE_ROOT/label) / 'sp_preflight.json')['source_sha256']==sha(__file__),'SP preflight source changed')
                continue
            args=[sys.executable,'-u',str(Path(__file__)),command,'--alpha',str(alpha)]
            if command=='train' and resume:args.append('--resume')
            save(SOURCE_ROOT/'STATUS.json',{'status':'running','alpha':alpha,'command':command,'test_used':False})
            print('SUITE',label,command,flush=True)
            with (logs/f'{label}_{command}.log').open('ab') as stream:
                result=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                save(SOURCE_ROOT/'STATUS.json',{'status':'failed','alpha':alpha,'command':command,'exit_code':result.returncode})
                raise RuntimeError(f'{label} {command} failed; inspect logs')
    save(SOURCE_ROOT/'STATUS.json',{'status':'training_complete','completed_runs':6,'report':'pending','test_used':False})
    print('SUITE six SP runs completed',flush=True)


def main():
    global ALPHA,HERE
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('audit','preflight','train','suite'))
    parser.add_argument('--alpha',type=float,choices=(.95,.80,.50),default=.95)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();ALPHA=args.alpha;HERE=SOURCE_ROOT/f'alpha_{round(ALPHA*100):03d}'
    if args.command=='suite':suite(args.resume)
    elif args.command=='audit':audit()
    elif args.command=='preflight':preflight()
    else:
        require(read(audit_dir(HERE) / 'sp_preflight.json')['passed'],'Run SP preflight first')
        train(args.resume)


if __name__=='__main__':
    main()
