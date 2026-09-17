"""G-S/F-S first replay update. One entry: audit, preflight, train, report, status.

Reuse cnn_zone model/renderer/evaluation; never call the old train_one (it evaluates test).
All experiment code is here. No FEMM, test prediction, routing, or automatic search.
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

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
from cnn_zone.src import training as tr
from cnn_zone.src.dataset import SPMSMGeneDataset
from femm_zone import femm_config
from femm_zone.scripts.spmsm_mapping import genotype_sha256

PILOT = PROJECT / "experiments/input_distribution_pilot_v1"
ACCEPTED = PILOT / "post_femm_baseline_20260913"
OLD = PROJECT / "data_zone/processed/spmsm_topology_dataset/training_corrected_binary"
MODEL = PROJECT / "cnn_zone/models/v3_40000_6runs/polar90_224/vgg16_v2/seed_20260903"
SPLIT = PROJECT / "cnn_zone/outputs/splits/scheme_a_tavg_bands_train40000_val6483_test6483.npz"
CHECKPOINT_SHA = "34cdee2e373e36ecd40e82197de5ea34941397bafea5f586cb00eb87dbfe58ba"
PHYSICS_SHA = "52459cf11c89ee1589f4d3a11a7e86e92450ac14ea082037a94a6c86e8a7835f"
CONDITION = "97cd342fe5d68919dffee97fe5ff2d37bcfdbfd461eb357e9ca92f75ee05c712"
SEED = 20260914
TARGETS = tr.TARGET_NAMES
ROLES = ("old_train", "old_validation", "train_G", "train_F", "dev_common")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    tr.atomic_json(Path(path), value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
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
        "experiment": "cnn_replay_update_v1", "seed": SEED, "groups": ["G-S", "F-S"],
        "initialization": "same frozen f0 weights independently; new optimizer/scheduler/AMP states",
        "initial_checkpoint": str((MODEL / "best_checkpoint.pt").relative_to(PROJECT)),
        "initial_checkpoint_sha256": CHECKPOINT_SHA,
        "model_class": "cnn_zone.src.models_v2.Semantic224VGG16V2", "input_shape": [8, 224, 224],
        "baseline_config": old_config, "target_order": list(TARGETS),
        "target_mean": old_config["target_mean"], "target_std": old_config["target_std"],
        "target_scaler_policy": "frozen f0 old-training scaler; never recomputed",
        "loss": "mean squared error over two independently standardized targets",
        "optimizer": old_config["optimizer"], "initial_learning_rate": old_config["learning_rate"] * 0.1,
        "weight_decay": old_config["weight_decay"], "effective_batch_size": 64,
        "physical_batch_size": old_config["physical_batch_size"], "old_per_update": 48, "new_per_update": 16,
        "max_updates": 10000, "validate_every": 500, "checkpoint_every": 100,
        "minimum_updates": 2000, "early_stopping_patience": 5, "improvement_epsilon": 1e-12,
        "old_mae_limit_multiplier": 1.05,
        "selection": "both old MAEs <= 1.05*f0; then minimize new-dev dual-target mean standardized MSE",
        "early_stop": "at >=2000 successful optimizer updates, 5 consecutive validations without improving constrained incumbent (includes f0)",
        "scheduler": {"class": "ReduceLROnPlateau", "monitor": "new_dev_standardized_mse", "factor": 0.5,
                      "patience": old_config["scheduler_patience"], "min_lr": 1e-6},
        "amp": "CUDA float16", "grad_scaler_init_scale": 128.0, "clip_grad_norm": 100.0,
        "train_all_parameters": True, "num_workers": 0, "cpu_threads": 4,
        "randomness": "shared shuffled old/new index plans; same initialization/seed/dropout RNG; deterministic algorithms warn_only like baseline; CUDA AMP/device nondeterminism remains possible",
        "settings_source": "user first-round defaults; scheduler type/factor/patience/min_lr, optimizer, loss, AMP and microbatch inherited from baseline",
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
    require(checkpoint["config"] == old_config and checkpoint["epoch"] == 39, "Checkpoint/config identity mismatch")
    for key in ("target_mean", "target_std"):
        require(np.array_equal(checkpoint[key].numpy(), np.asarray(old_config[key], dtype=np.float32)), f"Frozen {key} differs")
    require(old_config["architecture"] == "vgg16_v2" and old_config["input_mode"] == "polar90_224" and
            old_config["optimizer"] == "AdamW" and old_config["physical_batch_size"] == 8, "Unexpected baseline implementation")
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
              "checkpoint_epoch": 39, "checkpoint_sha256": CHECKPOINT_SHA, "source_sha256": sha(__file__)}
    for path, value in ((HERE / "config.json", cfg), (HERE / "audit/data_audit.json", report)):
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
            for micro in range(8):
                yield np.concatenate((self.old_plan[step, micro*6:(micro+1)*6],
                                      self.new_plan[step, micro*2:(micro+1)*2] + self.old_count)).tolist()

    def __len__(self):
        return (self.stop - self.start) * 8


def plans(cfg):
    directory = HERE / "audit"
    values = [cycle_plan(40000, 48, cfg["max_updates"], SEED+1), cycle_plan(1400, 16, cfg["max_updates"], SEED+2)]
    for name, value in zip(("old_replay_indices.npy", "new_replay_indices.npy"), values):
        path = directory / name
        if path.exists():
            require(np.array_equal(np.load(path), value), "Replay plan changed")
        else:
            np.save(path, value)
    return values


def initialize(cfg):
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
    renderer = tr.build_renderer(spec, PROJECT, device).eval()
    require(renderer.coordinate_system == "polar90", "Renderer coordinates changed")
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


def perform_update(model, renderer, spec, mean, std, device, optimizer, scaler, batches):
    model.train()
    mean, std = mean.to(device), std.to(device)
    attempts = 0
    while True:
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
            sums += [float(loss_sum.detach()), float(errors[:6].sum().detach()), float(errors[6:].sum().detach())]
        previous_scale = scaler.get_scale()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        scaler.step(optimizer)
        scaler.update()
        attempts += 1
        if scaler.get_scale() >= previous_scale:
            require(bool(torch.isfinite(norm)), "Invalid gradient update")
            return {"mse": sums[0]/128, "old_mse": sums[1]/96, "new_mse": sums[2]/32,
                    "attempts": attempts, "grad_norm": float(norm), "amp_scale": scaler.get_scale()}
        require(attempts < 20, "Repeated AMP overflow; abort without changing training settings")


def pack_state(model, optimizer, scheduler, scaler, mean, std, cfg, group, step, history, trace, elapsed, best, no_improvement, sampled):
    return {"group": group, "step": step, "seed": SEED, "config": cfg, "config_sha256": digest(cfg),
            "audit_sha256": sha(HERE / "audit/data_audit.json"), "script_sha256": sha(__file__),
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "target_mean": mean, "target_std": std, "history": history, "trace": trace,
            "elapsed_seconds": elapsed, "best": best, "no_improvement": no_improvement, "sampled": sampled,
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state()}


def restore(state, cfg, group, model, optimizer, scheduler, scaler):
    require(state["group"] == group and state["config_sha256"] == digest(cfg) and state["seed"] == SEED,
            "Resume experiment identity mismatch")
    require(state["script_sha256"] == sha(__file__) and state["audit_sha256"] == sha(HERE / "audit/data_audit.json"), "Resume source/data changed")
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
    directory = HERE / "audit/preflight"
    directory.mkdir(parents=True, exist_ok=True)
    old_plan, new_plan = plans(cfg)
    model, renderer, spec, mean, std, device, opt, scheduler, scaler = initialize(cfg)
    mixed = array_dataset(np.concatenate((data["old_train"].bits, data["train_G"].bits)), np.concatenate((data["old_train"].targets, data["train_G"].targets)))
    loader = DataLoader(mixed, batch_sampler=ReplaySampler(old_plan, new_plan, 0, 2), num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(SEED+3))
    batches = list(loader)
    require(len(batches) == 16 and all(int((b[0] < 40000).sum()) == 6 for b in batches), "Replay microbatch ratio failed")
    inputs = tr.make_inputs(batches[0][1].to(device), spec, renderer)
    require(tuple(inputs.shape) == (8, 8, 224, 224), "Input encoding shape failed")
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
    first = perform_update(model, renderer, spec, mean, std, device, opt, scaler, batches[:8])
    scheduler.step(before["dev_common"]["standardized_mse"])
    state = pack_state(model, opt, scheduler, scaler, mean, std, cfg, "preflight", 1, [], [], 0, {}, 0, {})
    tr.atomic_torch_save(directory / "roundtrip_checkpoint.pt", state)
    continuous = perform_update(model, renderer, spec, mean, std, device, opt, scaler, batches[8:])
    reference = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    saved = torch.load(directory / "roundtrip_checkpoint.pt", map_location=device, weights_only=False)
    restore(saved, cfg, "preflight", model, opt, scheduler, scaler)
    resumed = perform_update(model, renderer, spec, mean, std, device, opt, scaler, batches[8:])
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
    reference_rows = {r["gene_id"]: r for r in csv_read(ACCEPTED / "baseline_predictions.csv") if r["split_role"] in ("old_validation", "dev")}
    max_diff = 0.0
    for role in ("old_validation", "dev_common"):
        for row in csv_read(HERE / "baseline" / (role + "_predictions.csv")):
            previous = reference_rows[row["gene_id"]]
            max_diff = max(max_diff, *(abs(float(row[t+"_pred_nm"])-float(previous[t+"_pred_nm"])) for t in TARGETS))
    require(max_diff <= 1e-5, "Step-zero predictions differ from accepted f0")
    save(HERE / "audit/preflight.json", {"status": "passed", "input_shape": [8,8,224,224], "replay": [48,16],
         "checkpoint_next_update_bit_exact": True, "scaler_roundtrip": True, "test_guards_passed": True,
         "probe_seconds_for_three_updates_and_checkpoint_roundtrip": seconds, "peak_allocated_gib": peak,
         "first_update": first, "max_step0_prediction_difference_nm": max_diff,
         "formal_weights_reset_to_f0": True, "baseline": baseline, "gpu": torch.cuda.get_device_name(0),
         "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "script_sha256": sha(__file__),
         "config_sha256": digest(cfg), "audit_sha256": sha(HERE / "audit/data_audit.json")})
    # The successful resume probe checkpoint is disposable, not a training start.
    (directory / "roundtrip_checkpoint.pt").unlink()
    print(f"PREFLIGHT passed; f0 max prediction difference={max_diff:g}; peak={peak:.2f} GiB", flush=True)


def save_weights(path, model, mean, std, cfg, group, step, metrics):
    tr.atomic_torch_save(path, {"model_state": model.state_dict(), "target_mean": mean, "target_std": std,
                         "config": cfg, "group": group, "step": step, "seed": SEED, "metrics": metrics,
                         "f0_sha256": CHECKPOINT_SHA, "audit_sha256": sha(HERE / "audit/data_audit.json")})


def train_group(group, cfg, data, meta, resume):
    directory = HERE / "runs" / group
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        require(resume, "Completed run exists; --resume skips it without overwriting")
        return read(directory / "result.json")
    last = directory / "last_checkpoint.pt"
    require(resume or not last.exists(), "Existing run requires --resume")
    model, renderer, spec, mean, std, device, optimizer, scheduler, scaler = initialize(cfg)
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
        require(initial == baseline, f"{group} step zero differs from frozen f0")
        history.append({"step": 0, "elapsed_seconds": 0, "learning_rate": cfg["initial_learning_rate"],
                        "feasible": True, "metrics": initial})
        save(directory / "initialization.json", {"f0_sha256": CHECKPOINT_SHA, "weights_equal_f0": True,
             "new_optimizer_state_empty": len(optimizer.state) == 0, "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
             "seed": SEED, "old_plan_sha256": sha(HERE / "audit/old_replay_indices.npy"),
             "new_plan_sha256": sha(HERE / "audit/new_replay_indices.npy")})
    started = time.perf_counter()
    stop_reason = "max_updates"
    try:
        while step < cfg["max_updates"]:
            if step >= cfg["minimum_updates"] and no_improvement >= cfg["early_stopping_patience"]:
                stop_reason = "early_stopping_no_constrained_improvement"
                break
            stop = min(((step // cfg["validate_every"]) + 1) * cfg["validate_every"], cfg["max_updates"])
            loader = iter(DataLoader(mixed, batch_sampler=ReplaySampler(old_plan, new_plan, step, stop), num_workers=0, pin_memory=True,
                                     generator=torch.Generator().manual_seed(SEED+3)))
            while step < stop:
                batches = [next(loader) for _ in range(8)]
                tick = time.perf_counter()
                lr = optimizer.param_groups[0]["lr"]
                value = perform_update(model, renderer, spec, mean, std, device, optimizer, scaler, batches)
                step += 1
                sampled["old_draws"] += 48 * value["attempts"]
                sampled["new_draws"] += 16 * value["attempts"]
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
            if feasible and (best["feasible_score"] is None or score < best["feasible_score"] - cfg["improvement_epsilon"]):
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
                            "old_unique_covered": int(len(np.unique(old_plan[:step])))})
            save(directory / "history.json", history)
            csv_write(directory / "training_trace.csv", trace)
            tr.atomic_torch_save(last, pack_state(model, optimizer, scheduler, scaler, mean, std, cfg, group, step, history, trace,
                                elapsed_before+time.perf_counter()-started, best, no_improvement, sampled))
            print(f"VALID {group} step={step} feasible={feasible} oldMAE={[metrics['old_validation']['metrics'][t]['mae'] for t in TARGETS]} newMSE={score:.6g} no_improvement={no_improvement}", flush=True)
        elapsed = elapsed_before + time.perf_counter() - started
        success = best["feasible_score"] is not None and best["feasible_score"] < baseline["dev_common"]["standardized_mse"] - cfg["improvement_epsilon"]
        result = {"status": "complete", "group": group, "stop_reason": stop_reason, "updates": step,
                  "elapsed_seconds": elapsed, "best": best, "success": success,
                  "conclusion": "found constrained improvement" if success else "本轮未找到满足条件的改善模型",
                  "selected_step": best["feasible_step"] if success else 0,
                  "selected_checkpoint": str((directory / "best_feasible.pt").relative_to(HERE)) if success else str((MODEL / "best_checkpoint.pt").relative_to(PROJECT)),
                  "fallback_to_f0": not success, "sampled": sampled,
                  "old_unique_covered": int(len(np.unique(old_plan[:step]))),
                  "new_unique_covered": int(len(np.unique(new_plan[:step]))),
                  "successful_update_old_draws": step*48, "successful_update_new_draws": step*16,
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
    probe = read(HERE / "audit/preflight.json")
    require(probe["status"] == "passed" and probe["script_sha256"] == sha(__file__) and probe["config_sha256"] == digest(cfg) and
            probe["audit_sha256"] == sha(HERE / "audit/data_audit.json"), "Run preflight for the current source/config first")
    lock = HERE / "training.lock"
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
        save(HERE / "audit/runtime.json", {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
             "gpu": torch.cuda.get_device_name(0), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "source_sha256": sha(__file__), "process_pid": os.getpid()})
        for group in cfg["groups"]:
            train_group(group, cfg, data, meta, resume)
        require(sha(MODEL / "best_checkpoint.pt") == CHECKPOINT_SHA, "Frozen f0 was modified")
        report()
    finally:
        lock.unlink(missing_ok=True)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    out = HERE / "report"
    out.mkdir(exist_ok=True)
    cfg, baseline = read(HERE / "config.json"), read(HERE / "baseline/metrics.json")
    results = {g: read(HERE / "runs" / g / "result.json") for g in cfg["groups"]}
    histories = {g: read(HERE / "runs" / g / "history.json") for g in cfg["groups"]}
    metrics = {"f0": baseline}
    prediction_dirs = {"f0": HERE / "baseline"}
    variants = []
    for group, result in results.items():
        selected = result["selected_step"]
        name = "f_G" if group == "G-S" else "f_F"
        folder = HERE / "runs" / group / f"step{selected:05d}"
        metrics[name] = read(folder / "metrics.json")
        prediction_dirs[name] = folder
        for variant, step in (("selected", selected), ("unconstrained", result["best"]["unconstrained_step"]), ("last", result["updates"])):
            value = read(HERE / "runs" / group / f"step{step:05d}" / "metrics.json")
            for role in ("old_validation", "dev_common"):
                for target in TARGETS:
                    m = value[role]["metrics"][target]
                    variants.append({"group": group, "variant": variant, "step": step, "role": role, "target": target,
                                     "mae_nm": m["mae"], "rmse_nm": m["rmse"], "p95_nm": m["absolute_error_p95"],
                                     "standardized_mse": value[role]["standardized_mse"], "feasible": admissible(value, baseline)})
    rows = []
    for name, value in metrics.items():
        for role in ("old_validation", "dev_common"):
            strata = {"all": value[role], **(value[role].get("sources", {}))}
            for source, values in strata.items():
                for target in TARGETS:
                    m = values["metrics"][target]
                    ref = baseline[role] if source == "all" else baseline[role]["sources"][source]
                    rows.append({"model": name, "role": role, "source": source, "n": values["n"], "target": target,
                                 "mae_nm": m["mae"], "rmse_nm": m["rmse"], "p95_nm": m["absolute_error_p95"],
                                 "mae_change_percent_vs_f0": 100*(m["mae"]/ref["metrics"][target]["mae"]-1),
                                 "new_mae_improvement_percent": 100*(1-m["mae"]/ref["metrics"][target]["mae"]) if role == "dev_common" else None})
    csv_write(out / "validation_metrics.csv", rows)
    csv_write(out / "checkpoint_tradeoffs.csv", variants)
    save(out / "comparison.json", {"selected_models": metrics, "runs": results, "test_evaluated": False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for group, history in histories.items():
        trace = csv_read(HERE / "runs" / group / "training_trace.csv")
        for j, key in enumerate(("old_mse", "new_mse")):
            xs, ys = [], []
            for start in range(0, len(trace), 100):
                block = trace[start:start+100]
                xs.append(int(block[-1]["step"]))
                ys.append(float(np.mean([float(r[key]) for r in block])))
            axes[j,0].plot(xs, ys, label=group)
        xs = [r["step"] for r in history]
        for j, target in enumerate(TARGETS):
            for k, role in enumerate(("old_validation", "dev_common"), 1):
                axes[j,k].plot(xs, [h["metrics"][role]["metrics"][target]["mae"] for h in history], "o-", ms=3, label=group)
    for j, target in enumerate(TARGETS):
        axes[j,1].axhline(baseline["old_validation"]["metrics"][target]["mae"]*1.05, color="black", ls="--", label="f0 × 1.05")
        axes[j,2].axhline(baseline["dev_common"]["metrics"][target]["mae"], color="black", ls="--", label="f0")
        axes[j,1].set_title("旧验证 " + target + " MAE (N·m)")
        axes[j,2].set_title("新验证 " + target + " MAE (N·m)")
    axes[0,0].set_title("回放旧样本训练标准化 MSE（100步均值）")
    axes[1,0].set_title("补样训练标准化 MSE（100步均值）")
    for ax in axes.flat:
        ax.set_xlabel("优化器更新次数"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("75:25 回放首轮训练；验证集分开计算；最终测试未使用")
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("training_curves."+suffix), dpi=160)
    plt.close(fig)
    for role in ("old_validation", "dev_common"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        for column, (name, folder) in enumerate(prediction_dirs.items()):
            predictions = csv_read(folder / (role + "_predictions.csv"))
            for j, target in enumerate(TARGETS):
                x = np.array([float(r[target+"_true_nm"]) for r in predictions])
                y = np.array([float(r[target+"_pred_nm"]) for r in predictions])
                ax = axes[j,column]
                if role == "dev_common":
                    for source in "ULBP":
                        mask = np.array([r["source"] == source for r in predictions])
                        ax.scatter(x[mask], y[mask], s=14, alpha=.7, label=source)
                    ax.legend(fontsize=7)
                else:
                    ax.scatter(x, y, s=3, alpha=.25)
                lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
                ax.plot([lo,hi], [lo,hi], "k--", lw=1)
                m = metrics[name][role]["metrics"][target]
                fallback = name != "f0" and results["G-S" if name == "f_G" else "F-S"]["fallback_to_f0"]
                ax.set_title(f"{name}{'（保留f0）' if fallback else ''} · {target}\nMAE={m['mae']:.5f}, P95={m['absolute_error_p95']:.5f}")
                ax.set_xlabel("真实值 (N·m)"); ax.set_ylabel("预测值 (N·m)"); ax.grid(alpha=.2)
        fig.suptitle(("旧验证集（6483）" if role == "old_validation" else "共同新验证集（200）") + "；展示按预定规则选用的模型")
        for suffix in ("png", "pdf"):
            fig.savefig(out / (role + "_truth_prediction." + suffix), dpi=160)
        plt.close(fig)
    text = "# G-S / F-S 首轮 CNN 回放更新\n\n"
    text += "本轮固定一个随机种子20260914，使用同一冻结f0独立热启动、全网络更新、旧/新样本48:16回放。最终测试未解封。\n\n"
    text += "| 组别 | 实际更新 | 用时(min) | 可接受改善 | 选用步数 | 无约束最优步数 | 旧/新累计抽取 | 旧覆盖 |\n|---|---:|---:|---|---:|---:|---|---:|\n"
    for group, r in results.items():
        text += f"| {group} | {r['updates']} | {r['elapsed_seconds']/60:.2f} | {'是' if r['success'] else '本轮未找到满足条件的改善模型'} | {r['selected_step']} | {r['best']['unconstrained_step']} | {r['sampled']['old_draws']}/{r['sampled']['new_draws']} | {r['old_unique_covered']}/40000 |\n"
    text += "\n选用步数0表示继续保留f0，不能解释为更新训练成功。最后权重和无约束最优权重仍保存，便于检查精度取舍。\n\n"
    text += "| 模型 | 验证集 | 目标 | MAE | RMSE | 绝对误差P95 | MAE相对f0变化 |\n|---|---|---|---:|---:|---:|---:|\n"
    for row in rows:
        if row['source'] == 'all':
            text += f"| {row['model']} | {row['role']} | {row['target']} | {row['mae_nm']:.6f} | {row['rmse_nm']:.6f} | {row['p95_nm']:.6f} | {row['mae_change_percent_vs_f0']:+.2f}% |\n"
    text += "\n误差单位均为N·m。负变化表示改善。U/L/B/P各50例的逐目标MAE、RMSE、P95及改善比例见 [validation_metrics.csv](validation_metrics.csv)。\n\n"
    text += "## 未加约束模型的取舍\n\n"
    for group, r in results.items():
        step = r['best']['unconstrained_step']
        value = read(HERE / 'runs' / group / f'step{step:05d}' / 'metrics.json')
        old_change = [100*(value['old_validation']['metrics'][t]['mae']/baseline['old_validation']['metrics'][t]['mae']-1) for t in TARGETS]
        new_change = [100*(1-value['dev_common']['metrics'][t]['mae']/baseline['dev_common']['metrics'][t]['mae']) for t in TARGETS]
        text += f"- {group}无约束最优在{step}步：新验证Tavg/DeltaT MAE改善{new_change[0]:.2f}%/{new_change[1]:.2f}%；旧验证MAE变化{old_change[0]:+.2f}%/{old_change[1]:+.2f}%，{'满足' if admissible(value,baseline) else '不满足'}双目标各自5%容限。\n"
    text += "\n详细取舍见 [checkpoint_tradeoffs.csv](checkpoint_tradeoffs.csv)。训练曲线及旧/新验证真实值—预测图分别为本目录PNG/PDF。\n\n"
    text += "## 设置、证据与边界\n\n"
    text += "AdamW初始学习率1e−5、weight_decay=1e−4；ReduceLROnPlateau监控新验证标准化MSE，factor=0.5、patience=3、最低1e−6。microbatch=8，每批6旧+2新，累积8批；sum-MSE除以64×2，沿用AMP、梯度裁剪100和冻结旧目标尺度。\n\n"
    text += "每500步分别验证旧6483和新200，旧两目标MAE均不超过f0×1.05时才参加受约束选择；不按数据量合并验证集。至少2000步且连续5次验证无受约束改善时早停，否则上限10000步。没有根据结果改变比例、门槛或训练预算。\n\n"
    text += "旧完整40000样本打乱轮换，新1400样本循环；两组共享相同索引抽取计划和随机种子。AMP若溢出会降低scale并重试同一有效批次，仅成功optimizer.step计入更新预算，累计抽取另记实际尝试。CPU线程4、DataLoader worker0；Dropout和CUDA AMP仍可能带来设备相关随机差异。\n\n"
    text += "旧数据原有898组冲突隔离、538条初始代磁体数量异常保留策略沿用，不删除样本、不修改标签。旧标签历史质量限制仍存在，本次结论针对已验收的六点均值与峰峰差口径。\n\n"
    text += "源码：[../update.py](../update.py)；冻结配置：[../config.json](../config.json)；数据来源和隔离：[../audit/data_audit.json](../audit/data_audit.json)；GPU/恢复检查：[../audit/preflight.json](../audit/preflight.json)。每组runs下保存日志、验证预测、检查点和哈希。\n\n"
    text += "这是单种子验证；不能证明G稳定优于F，也不能声称优于同预算随机补样。过拟合、来源不足与下一步建议见随后基于实测结果补充的 [interpretation.md](interpretation.md)。\n"
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    print("REPORT written: " + str(out / "REPORT.md"), flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "preflight", "train", "report", "status"), nargs="?", default="status")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        audit()
    elif args.command == "preflight":
        preflight()
    elif args.command == "train":
        train(args.resume)
    elif args.command == "report":
        report()
    else:
        for group in ("G-S", "F-S"):
            path = HERE / "runs" / group / "progress.json"
            print(group, json.dumps(read(path), ensure_ascii=False) if path.exists() else "not started")


if __name__ == "__main__":
    main()
