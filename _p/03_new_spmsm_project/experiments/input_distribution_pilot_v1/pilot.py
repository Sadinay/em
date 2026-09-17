"""03 四组输入分布试验。seeds 只选种子；prepare 不求解；solve 需手动调用。

新增流程集中于本文件；复用 03 的模型、renderer、基因映射及唯一 FEMM 配置。
默认 status。完整用法见同目录 README.md。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import csv
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import multiprocessing as mp
import signal

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from femm_zone import femm_config as physical
from femm_zone.scripts import spmsm_mapping as mapping
from cnn_zone.src.fem_mesh import parse_ans_mesh, _fixed_class_for_label

DATA = PROJECT / "data_zone/processed/spmsm_topology_dataset/training_corrected_binary"
MODEL = PROJECT / "cnn_zone/models/v3_40000_6runs/polar90_224/vgg16_v2/seed_20260903"
LOOKUP = PROJECT / "cnn_zone/outputs/lookups/spmsm_polar90_224_full_motor.npz"
SPLIT = PROJECT / "cnn_zone/outputs/splits/scheme_a_tavg_bands_train40000_val6483_test6483.npz"
HISTORY_RUN = PROJECT / "femm_zone/workspaces/teacher_angle29_5genes_20260908"
REFERENCE_ANS = HISTORY_RUN / "G2/phase_0/angle_29/model.ans"
SEED = 20260909
DEFAULT_WORKERS = 6  # 用户追加执行设置；不修改已冻结的种子/物理配置。
WORKER_STOP = None
STREAMS = {name: int.from_bytes(hashlib.sha256(f"{SEED}/{name}".encode()).digest()[:8], "big")
           for name in ("pool", "dev", "test", "anchors", "audit", "parents")}
SCALES = {"L": [(0.5, 0.5), (0.5, 2), (1.5, 0.5), (1.5, 2)],
          "B": [(2, 4), (2, 8), (4, 4), (4, 8)]}
HELD_SCALES = {"L": [(0.75, 1), (1, 1.5)], "B": [(1, 8), (4, 2)]}
CSV_FIELDS = ["gene_id", "bits", "source", "family_id", "split_role", "generator_parameters",
              "seed", "parent_old_index", "parameter_regime", "selection_method", "selection_rank",
              "selection_distance_squared", "display_region_id", "experiment_groups", "magnet_cells",
              "raw_bits", "raw_gene_id", "repair_changed_cells", "repair_passes"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return mapping.file_sha256(Path(path))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(rows) or columns, f"Empty CSV requires columns: {path}")
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def bits_of(rows):
    bits = np.array([[int(c) for c in r["bits"]] for r in rows], dtype=np.uint8)
    validate_bits(bits)
    require(all(mapping.genotype_sha256(b) == r["gene_id"] for b, r in zip(bits, rows)), "Gene hash mismatch")
    return bits


def validate_bits(bits):
    require(bits.ndim == 2 and bits.shape[1] == 120 and np.isin(bits, [0, 1]).all(), "Expected binary [N,120]")
    grids = bits.reshape(-1, 20, 6).swapaxes(1, 2)
    require(np.array_equal(grids.swapaxes(1, 2).reshape(-1, 120), bits), "Encoding roundtrip failed")


def isolated_mask(grid):
    """Eight neighbors within the 6x20 domain; only an entirely opposite ring flips."""
    same = np.zeros_like(grid, dtype=bool)
    for dr in (-1, 0, 1):
        for da in (-1, 0, 1):
            if dr == da == 0:
                continue
            target = (slice(max(0, dr), min(6, 6+dr)), slice(max(0, da), min(20, 20+da)))
            neighbor = (slice(max(0, -dr), min(6, 6-dr)), slice(max(0, -da), min(20, 20-da)))
            same[target] |= grid[target] == grid[neighbor]
    return ~same


def repair_isolated(grid):
    grid = np.array(grid, dtype=np.uint8, copy=True)
    seen = set(); passes = 0
    while True:
        flip = isolated_mask(grid)
        if not flip.any():
            return grid, passes
        key = grid.tobytes()
        require(key not in seen, "Isolated-cell repair oscillated; do not silently choose a topology")
        seen.add(key); grid[flip] ^= 1; passes += 1


def validate_repaired(bits):
    validate_bits(bits)
    require(all(not isolated_mask(b.reshape(20, 6).T).any() for b in bits), "An isolated Air/PM cell remains")


def old_data():
    grids = np.load(DATA / "topology_bits.npy", mmap_mode="r")
    require(grids.shape == (64804, 6, 20), "Old dataset shape changed")
    genes = np.asarray(grids.swapaxes(1, 2).reshape(-1, 120), dtype=np.uint8)
    validate_bits(genes)
    with np.load(SPLIT) as f:
        split = {k: np.asarray(f[k], dtype=np.int64) for k in ("train", "validation", "test", "unused_train_pool")}
    require([len(v) for v in split.values()] == [40000, 6483, 6483, 11838], "Old split sizes changed")
    require(np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(genes))), "Old split overlap or omission")
    require(hashlib.sha256(split["train"].tobytes()).hexdigest() ==
            "ade8379a3c0c72ed4a4127ac60efce2dc0799de9a95c8df5d0d0d61e38db596f", "Train index hash changed")
    return genes, split


def all_history_keys(genes):
    # Only genotype identity is read; historical test targets never enter selection.
    keys = {np.packbits(b).tobytes() for b in genes}
    mat = loadmat(physical.MAT_FILE, variable_names=["population_all", "population_noChange_all", "population"])
    counts = {}
    for name, a in mat.items():
        if name.startswith("__"):
            continue
        if a.ndim == 3:
            a = a.transpose(2, 0, 1).reshape(-1, 120)
        validate_bits(a)
        packed = np.packbits(a.astype(np.uint8), axis=1)
        keys.update(row.tobytes() for row in packed)
        counts[name] = len(a)
    return keys, counts


def physical_config():
    cfg = physical.load_config()
    require(cfg["current"]["amplitude_a"] == 3.5 and cfg["current"]["pole_pairs"] == 4 and
            cfg["problem"]["Depth"] == 36 and cfg["inner_angles_deg"] == [29, 32, 35, 38, 41, 44] and
            cfg["initial_phases_deg"] == [0] and cfg["airgap"]["outer_angle_deg"] == 0 and
            cfg["torque_multiplier"] == 1 and cfg["problem"]["MinAngle"] == 15 and
            cfg["problem"]["Precision"] == 1e-8 and cfg["problem"]["DoSmartMesh"] == 1 and
            cfg["problem"]["Frequency"] == 0, "03 physical settings no longer match this frozen pilot")
    # The five historical genes are not the batch queue or part of a solve condition.
    return {k: v for k, v in cfg.items() if k not in ("genes", "gene_source")}


def config_payload():
    return {"protocol": "input_distribution_pilot_v1", "seed": SEED, "named_stream_seeds": STREAMS,
            "candidate_per_source": 5000, "train_per_method": 1400, "dev_per_source": 50,
            "test_per_source": {"same_range": 50, "parameter_holdout": 50}, "anchors": 512,
            "parent_counts_pool_dev_test": [128, 64, 64], "magnet_count_range": [12, 108],
            "scales": SCALES, "held_scales": HELD_SCALES, "held_U_counts": [3, 6, 114, 117],
            "held_U_allocation": [13, 13, 12, 12], "rectangles": [[1, 2], [2, 3], [3, 5]],
            "held_rectangles": [[4, 10], [6, 8]], "covariance": "exp(-0.5*((dr/lr)^2+(da/la)^2)); no wrap",
            "covariance_factor": "symmetric eigh; roundoff-negative eigenvalues clipped to zero, no added jitter",
            "G": "position-labelled WL0+WL1 exact categorical matches, each layer divided by sqrt(2)*492",
            "F": "f0 eval FP32; final-Linear pre-hooks 256+256; centered scalar RMS per head; divide sqrt(2)",
            "selection": "LCMD, exact block distances; 512 shared old anchors; SHA256 lexical ties",
            "user_amendment_isolated_cell_filter": {"neighbors": 8, "boundary": "in-domain only; no periodic wrap",
                 "rule": "flip only if all existing surrounding neighbors differ; synchronous passes to fixed point",
                 "applies_to": "all newly generated pool/dev/test before hashing, history exclusion, features and selection",
                 "not_general_majority_smoothing": True, "magnet_count_and_P_rectangles_describe_raw_generator": True},
            "reserve_policy": "no automatic replacements; failures remain failures, no unregistered budget expansion",
            "test_feature_policy": "no dev/test F feature extraction during selection",
            "worker_count": 1, "automatic_retries": 1, "regression_atol": 1e-6, "regression_rtol": 1e-6,
            "physical": physical_config()}


def source_files():
    return [MODEL / "best_checkpoint.pt", MODEL / "config.json", MODEL / "result.json", LOOKUP,
            DATA / "topology_bits.npy", DATA / "targets_tavg_delta.npy", SPLIT,
            physical.MAT_FILE, physical.TEMPLATE_FILE, REFERENCE_ANS,
            PROJECT / "cnn_zone/src/training.py", PROJECT / "cnn_zone/src/training_inputs.py",
            PROJECT / "cnn_zone/src/models_v2.py", PROJECT / "cnn_zone/src/fem_mesh.py",
            Path(physical.__file__), Path(mapping.__file__)]


def bootstrap():
    cfg = config_payload()
    path = HERE / "pilot_config.json"
    if path.exists():
        require(digest(read(path)) == digest(cfg), "Frozen pilot configuration differs; use a separate version directory")
    else:
        save(path, cfg)
    hashes = {str(p.relative_to(PROJECT)): sha(p) for p in source_files()}
    prov_path = HERE / "provenance.json"
    if prov_path.exists():
        require(read(prov_path)["input_sha256"] == hashes, "Frozen input or reused source has changed")
    else:
        genes, split = old_data()
        save(prov_path, {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                         "input_sha256": hashes, "config_sha256": sha(path), "split_counts": {k: len(v) for k, v in split.items()},
                         "train_index_bytes_sha256": hashlib.sha256(split["train"].tobytes()).hexdigest(),
                         "initial_pilot_source_sha256": sha(__file__), "python": sys.version, "numpy": np.__version__})
    return cfg


def generate():
    bootstrap()
    if (HERE / "generation_validation.json").exists():
        for name, value in read(HERE / "generation_validation.json")["file_sha256"].items():
            require(sha(HERE / name) == value, f"Generated input changed: {name}")
        return
    genes, split = old_data()
    existing, history_counts = all_history_keys(genes)
    historical_unique = len(existing)
    parent_rng = np.random.default_rng(STREAMS["parents"])
    chosen = parent_rng.choice(split["train"], 256, replace=False)
    parents = {"pool": chosen[:128], "dev": chosen[128:192], "test": chosen[192:]}
    save(HERE / "parent_families.json", {k: [int(i) for i in v] for k, v in parents.items()})
    anchor_ids = np.random.default_rng(STREAMS["anchors"]).choice(split["train"], 512, replace=False)
    anchors = [{"old_index": int(i), "gene_id": mapping.genotype_sha256(genes[i]),
                "bits": "".join(map(str, genes[i])), "split_role": "old_train_anchor"} for i in anchor_ids]
    csv_write(HERE / "old_anchors.csv", anchors)
    rr, aa = np.indices((6, 20)); rr, aa = rr.ravel(), aa.ravel()
    factors = {}
    for scale in {s for values in [*SCALES.values(), *HELD_SCALES.values()] for s in values}:
        covariance = np.exp(-0.5 * (((rr[:, None]-rr)/scale[0])**2 + ((aa[:, None]-aa)/scale[1])**2))
        vals, vecs = np.linalg.eigh(covariance)
        require(vals.min() > -1e-10, "Gaussian covariance is not PSD")
        factors[scale] = vecs * np.sqrt(np.maximum(vals, 0))[None, :]
    collisions = Counter()
    files = ["parent_families.json", "old_anchors.csv"]
    for stream, total in (("pool", 5000), ("dev", 50), ("test", 100)):
        rng = np.random.default_rng(STREAMS[stream])
        rows = []
        for source in "ULBP":
            for n in range(total):
                held = stream == "test" and n >= 50
                slot = n - 50 if held else n
                for attempt in range(10000):
                    m = int([3, 6, 114, 117][slot % 4] if source == "U" and held else rng.integers(12, 109))
                    params = {}; parent = ""
                    if source == "U":
                        grid = np.zeros(120, dtype=np.uint8)
                        grid[rng.choice(120, m, replace=False)] = 1
                        grid = grid.reshape(6, 20)
                        params = {"m": m}
                    elif source in "LB":
                        scales = HELD_SCALES[source] if held else SCALES[source]
                        scale = scales[slot % len(scales)]
                        field = factors[scale] @ rng.standard_normal(120)
                        grid = np.zeros(120, dtype=np.uint8)
                        grid[np.argsort(field, kind="stable")[-m:]] = 1
                        grid = grid.reshape(6, 20)
                        params = {"m": m, "scale": list(scale)}
                    else:
                        parent = int(parents[stream][n % len(parents[stream])])
                        grid = genes[parent].reshape(20, 6).T.copy()
                        masks, rectangles = [], []
                        for _ in range(1 if held else 2):
                            for _trial in range(10000):
                                h, w = ([(4, 10), (6, 8)][slot % 2] if held else [(1, 2), (2, 3), (3, 5)][int(rng.integers(3))])
                                r, a = int(rng.integers(7-h)), int(rng.integers(21-w))
                                mask = np.zeros((6, 20), dtype=bool); mask[r:r+h, a:a+w] = True
                                if all(not (mask & prior).any() for prior in masks):
                                    break
                            else:
                                raise RuntimeError("Could not place nonoverlapping rectangles")
                            masks.append(mask); rectangles.append([r, a, h, w]); grid[mask] ^= 1
                        params = {"rectangles_r_a_h_w": rectangles}
                    raw_bits = grid.T.reshape(120).copy()
                    grid, passes = repair_isolated(grid)
                    b = grid.T.reshape(120)
                    key = np.packbits(b).tobytes()
                    if key in existing:
                        collisions[f"{stream}/{source}"] += 1
                        continue
                    existing.add(key)
                    family = (f"{source}/parent/{parent}/{stream}" if source == "P" else
                              f"{source}/{stream}/field_{n:05d}/attempt_{attempt}/params_{digest(params)[:16]}")
                    rows.append({"gene_id": mapping.genotype_sha256(b), "bits": "".join(map(str, b)),
                                 "source": source, "family_id": family, "split_role": "candidate_train" if stream == "pool" else stream,
                                 "generator_parameters": json.dumps({**params, "attempt": attempt}, sort_keys=True),
                                 "seed": STREAMS[stream], "parent_old_index": parent,
                                 "parameter_regime": "parameter_holdout" if held else "same_range",
                                 "selection_method": "", "selection_rank": "", "selection_distance_squared": "",
                                 "display_region_id": "", "experiment_groups": "" if stream == "pool" else "G-S;F-S;G-E;F-E",
                                 "magnet_cells": int(b.sum()), "raw_bits": "".join(map(str, raw_bits)),
                                 "raw_gene_id": mapping.genotype_sha256(raw_bits), "repair_changed_cells": int(np.count_nonzero(raw_bits != b)),
                                 "repair_passes": passes})
                    break
                else:
                    raise RuntimeError("Candidate collision retry budget exhausted")
        validate_repaired(bits_of(rows))
        name = {"pool": "candidates_train.csv", "dev": "dev_common.csv", "test": "test_common.csv"}[stream]
        csv_write(HERE / name, rows, CSV_FIELDS); files.append(name)
        print(f"Generated {stream}: {len(rows)}", flush=True)
    save(HERE / "generation_validation.json", {"historical_unique_excluded": historical_unique,
         "history_variables_rows": history_counts, "exact_new_unique": len(existing)-historical_unique,
         "collisions_regenerated": dict(collisions), "random_streams": STREAMS,
         "file_sha256": {name: sha(HERE / name) for name in files},
         "P_note": "New split families use disjoint old-train parents; those parents remain old training data."})


def graph_build():
    path = HERE / "region_graph.json"
    if path.exists():
        graph = read(path)
        require(graph["ans_sha256"] == sha(REFERENCE_ANS) and graph["template_sha256"] == sha(physical.TEMPLATE_FILE), "Graph sources changed")
        return graph
    template = physical.TEMPLATE_FILE.read_text(encoding="utf8")
    labels = mapping.parse_labels(template)
    solved_labels = mapping.parse_labels(REFERENCE_ANS.read_text(encoding="utf8"))
    require(len(labels) == len(solved_labels) == 492, "Expected 492 physical regions")
    require(all(np.allclose([a["x_mm"], a["y_mm"]], [b["x_mm"], b["y_mm"]], atol=1e-10, rtol=0)
                for a, b in zip(labels, solved_labels)), "ANS label coordinates/order changed")
    positions = loadmat(physical.MAT_FILE, variable_names=["MaterialPosition"], squeeze_me=True)["MaterialPosition"]
    mapped = mapping.match_material_positions(template, positions)
    to_gene = np.full(492, -1, dtype=int); to_replica = np.full(492, -1, dtype=int)
    for row in mapped:
        i = row["label_index_1based"]-1
        to_gene[i], to_replica[i] = row["gene_index_1based"]-1, row["copy_index_1based"]-1
    with np.load(LOOKUP) as lookup:
        require(np.array_equal(to_gene, lookup["label_to_gene"]) and np.array_equal(to_replica, lookup["label_to_replica"]), "Graph and frozen CNN lookup mapping differ")
    nodes, triangles, owners = parse_ans_mesh(REFERENCE_ANS)
    require(np.array_equal(np.unique(owners), np.arange(492)), "Missing or invalid mesh regions")
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    owner = np.tile(owners, 3)
    order = np.lexsort((edges[:, 1], edges[:, 0])); edges, owner = edges[order], owner[order]
    _, starts, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    require(counts.max() <= 2, "Non-manifold mesh edge")
    pairs = starts[counts == 2]
    connections = sorted({tuple(sorted((int(owner[i]), int(owner[i+1])))) for i in pairs if owner[i] != owner[i+1]})
    xy = nodes[triangles]
    areas = np.abs((xy[:, 1, 0]-xy[:, 0, 0])*(xy[:, 2, 1]-xy[:, 0, 1]) -
                   (xy[:, 1, 1]-xy[:, 0, 1])*(xy[:, 2, 0]-xy[:, 0, 0]))/2
    region_area = np.bincount(owners, weights=areas, minlength=492)
    blocks = mapping.material_blocks(template)
    regions = [{"region_id": i, "gene_index": int(to_gene[i]), "replica_index": int(to_replica[i]),
                "magnetization_deg": r["magnetization_deg"], "fixed_material_class": _fixed_class_for_label(r, blocks) if to_gene[i] < 0 else None,
                "fixed_material_name": mapping._block_name(blocks[r["block_type_1based"]-1]) if to_gene[i] < 0 else None,
                "mesh_area_mm2": float(region_area[i]), "x_mm": r["x_mm"], "y_mm": r["y_mm"]} for i, r in enumerate(labels)]
    graph = {"regions": regions, "shared_edge_adjacency": connections, "mesh_nodes": len(nodes), "mesh_triangles": len(triangles),
             "boundary_policy": "Finite 90-degree section, actual shared triangle edges only; no vertex-only or periodic seam edges.",
             "area_definition": "Sum of mesh triangle areas in the 90-degree section (4 full-motor sectors); curved-boundary discretization approximation.",
             "ans_sha256": sha(REFERENCE_ANS), "template_sha256": sha(physical.TEMPLATE_FILE), "mat_sha256": sha(physical.MAT_FILE)}
    save(path, graph)
    return graph


def wl_codes(bits, graph):
    """Exact position-labelled WL categories. IDs are compared for equality only."""
    nodes = graph["regions"]
    neighbors = [set([i]) for i in range(len(nodes))]
    for a, b in graph["shared_edge_adjacency"]:
        neighbors[a].add(b); neighbors[b].add(a)
    columns, memo = [], {}
    for layer in (0, 1):
        for i, node in enumerate(nodes):
            scope = [i] if layer == 0 else sorted(neighbors[i])
            deps = tuple(sorted({nodes[j]["gene_index"] for j in scope if nodes[j]["gene_index"] >= 0}))
            if deps not in memo:
                if not deps:
                    memo[deps] = np.zeros(len(bits), dtype=np.int32)
                else:
                    packed = np.packbits(bits[:, deps], axis=1)
                    _, inv = np.unique(packed, axis=0, return_inverse=True)
                    memo[deps] = inv.astype(np.int32)
            columns.append(memo[deps])
    return np.stack(columns, axis=1)


def extract_features(batch_size=16):
    import torch
    from cnn_zone.src.training import TrainingSpec, build_model, build_renderer, make_inputs
    generate()
    config = read(MODEL / "config.json")
    ckpt = torch.load(MODEL / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    require(ckpt["config"] == config and ckpt["epoch"] == read(MODEL / "result.json")["best_epoch"] == 39 and
            ckpt["seed"] == 20260903 and config["input_mode"] == "polar90_224" and
            config["architecture"] == "vgg16_v2" and not config["circular_angular_padding"], "f0 identity mismatch")
    for key in ("target_mean", "target_std"):
        require(np.array_equal(ckpt[key].numpy(), np.array(config[key], dtype=np.float32)), f"{key} differs from checkpoint")
    spec = TrainingSpec(**{f.name: config[f.name] for f in fields(TrainingSpec)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = build_model(spec).to(device=device, memory_format=torch.channels_last).eval()
    model.load_state_dict(ckpt["model_state"], strict=True); model.requires_grad_(False)
    renderer = build_renderer(spec, PROJECT, device).eval()
    capture = {}
    def hook(name):
        def callback(_module, inputs):
            capture[name] = inputs[0].detach()
        return callback
    handles = [getattr(model.regressor, name)[-1].register_forward_pre_hook(hook(name)) for name in ("head_tavg", "head_delta_t")]
    genes, split = old_data()
    rows = csv_read(HERE / "candidates_train.csv")
    sources = {"old_train": genes[split["train"]], "candidates": bits_of(rows)}
    contract = {"checkpoint": sha(MODEL / "best_checkpoint.pt"), "lookup": sha(LOOKUP),
                "inputs": read(HERE / "provenance.json")["input_sha256"], "candidate_csv": sha(HERE / "candidates_train.csv"),
                "extractor_code": hashlib.sha256(inspect.getsource(extract_features).encode()).hexdigest(),
                "torch": torch.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "dtype": "FP32; TF32 off; no autocast", "batch_size": batch_size,
                "target_mean": ckpt["target_mean"].tolist(), "target_std": ckpt["target_std"].tolist()}
    cache = HERE / "cache"; cache.mkdir(exist_ok=True)
    contract_path = cache / "feature_contract.json"
    if contract_path.exists():
        require(read(contract_path) == contract, "Feature contract changed; preserve old cache and use a fresh experiment")
    else:
        save(contract_path, contract)
    for name, b in sources.items():
        path = cache / f"features_{name}.npy"; status_path = cache / f"features_{name}.json"
        state = read(status_path) if status_path.exists() else {"completed": 0, "elapsed_seconds": 0.0}
        if state.get("sha256"):
            require(sha(path) == state["sha256"], "Feature cache hash changed")
            continue
        values = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=(len(b), 512))
        start_time = time.perf_counter(); previous = state["elapsed_seconds"]
        with torch.inference_mode():
            for start in range(state["completed"], len(b), batch_size):
                tensor = torch.from_numpy(np.array(b[start:start+batch_size], copy=True)).to(device)
                prediction = model(make_inputs(tensor, spec, renderer))
                physical_pred = prediction.float() * ckpt["target_std"].to(device) + ckpt["target_mean"].to(device)
                h = torch.cat([capture[k] for k in ("head_tavg", "head_delta_t")], dim=1)
                require(h.shape == (len(tensor), 512) and torch.isfinite(h).all().item() and torch.isfinite(physical_pred).all().item(), "Invalid f0 feature output")
                end = start + len(tensor); values[start:end] = h.cpu().numpy()
                if end % 512 == 0 or end == len(b):
                    values.flush()
                    state.update(completed=end, elapsed_seconds=previous + time.perf_counter()-start_time)
                    save(status_path, state)
                    print(f"F features {name}: {end}/{len(b)}, {state['elapsed_seconds']:.1f}s", flush=True)
        values.flush(); del values
        state["sha256"] = sha(path); save(status_path, state)
    for h in handles:
        h.remove()
    raw = np.load(cache / "features_old_train.npy", mmap_mode="r")
    center = raw.astype(np.float64).mean(axis=0)
    scale = [float(np.sqrt(np.mean(np.sum((raw[:, k:k+256].astype(np.float64)-center[k:k+256])**2, axis=1)))) for k in (0, 256)]
    require(min(scale) > 1e-10, "Degenerate f0 head scale")
    save(HERE / "feature_scaler.json", {"center": center.tolist(), "head_scale": scale, "fit_role": "old_train_only", "count": 40000,
                                        "feature_contract_sha256": sha(contract_path)})


class Distances:
    def __init__(self, values, method, node_count=492):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method; self.node_count = node_count
        self.values = torch.from_numpy(np.ascontiguousarray(values)).to(self.device)

    def one(self, index):
        if self.method == "G":
            # Each layer's count vector / (sqrt(2)*N): each categorical mismatch contributes 1/N^2.
            # Keep integer mismatch counts during LCMD so exact score ties stay exact.
            # Divide exported distances/scores by N^2; this common scale does not alter any selection.
            return (self.values != self.values[index]).sum(dim=1).cpu().numpy().astype(np.float64)
        # Direct differences avoid cancellation near identical features. Float64 distance arithmetic.
        return ((self.values - self.values[index])**2).sum(dim=1).cpu().numpy()


def lcmd(distance, candidate_count, ids, anchor_count, count=1400):
    require(count <= candidate_count, "Too many requested seeds")
    nearest = np.full(candidate_count, np.inf); owner = np.full(candidate_count, -1, dtype=np.int32)
    centers = []; active = np.ones(candidate_count, dtype=bool)
    lexrank = np.empty(len(ids), dtype=int); lexrank[np.argsort(ids, kind="stable")] = np.arange(len(ids))
    def add(center):
        d = distance.one(center)[:candidate_count]
        require(np.isfinite(d).all() and (d >= 0).all(), "Invalid representation distance")
        replace = d < nearest
        ties = d == nearest
        if centers:
            old_rank = lexrank[np.array(centers)[np.maximum(owner, 0)]]
            replace |= ties & (lexrank[center] < old_rank)
        nearest[replace] = d[replace]; owner[replace] = len(centers)
        centers.append(center)
    for index in range(candidate_count, candidate_count+anchor_count):
        add(index)
    chosen, trace = [], []
    for rank in range(count):
        scores = np.bincount(owner[active], weights=nearest[active], minlength=len(centers))
        occupied = np.bincount(owner[active], minlength=len(centers)) > 0
        scores[~occupied] = -np.inf
        regions = np.flatnonzero(scores == scores.max())
        region = min(regions, key=lambda j: ids[centers[j]])
        members = np.flatnonzero(active & (owner == region))
        far = members[nearest[members] == nearest[members].max()]
        index = min(far, key=lambda j: ids[j])
        factor = distance.node_count**2 if distance.method == "G" else 1
        trace.append({"candidate_index": int(index), "distance_squared": float(nearest[index]/factor),
                      "region_score": float(scores[region]/factor), "region_center_gene_id": ids[centers[region]]})
        chosen.append(index); active[index] = False; add(index)
        if (rank+1) % 200 == 0:
            print(f"LCMD {distance.method}: {rank+1}/{count}", flush=True)
    return chosen, trace


def display_partition(distance, n, ids):
    first = int(np.random.default_rng(STREAMS["audit"]).integers(n))
    centers = [first]; nearest = distance.one(first)[:n]; owner = np.zeros(n, dtype=int)
    for _ in range(9):
        available = np.ones(n, dtype=bool); available[centers] = False
        max_dist = nearest[available].max()
        options = np.flatnonzero(available & (nearest == max_dist))
        center = min(options, key=lambda j: ids[j])
        d = distance.one(center)[:n]
        old_ids = np.array([ids[i] for i in centers])[owner]
        mask = (d < nearest) | ((d == nearest) & (np.array(ids[center]) < old_ids))
        owner[mask] = len(centers); nearest[mask] = d[mask]; centers.append(center)
    return [int(c) for c in centers], owner+1


def select_seeds():
    generate(); graph = graph_build()
    for name, count in (("old_train", 40000), ("candidates", 20000)):
        meta = read(HERE / f"cache/features_{name}.json")
        require(meta["completed"] == count and meta["sha256"] == sha(HERE / f"cache/features_{name}.npy"), "F feature cache is incomplete or changed")
    candidates = csv_read(HERE / "candidates_train.csv"); anchors = csv_read(HERE / "old_anchors.csv")
    cb, ab = bits_of(candidates), bits_of(anchors)
    ids = [r["gene_id"] for r in candidates+anchors]
    scaler = read(HERE / "feature_scaler.json")
    center = np.array(scaler["center"]); scale = np.repeat(scaler["head_scale"], 256)
    _, split = old_data(); position = {int(index): i for i, index in enumerate(split["train"])}
    results = {}
    for method in "GF":
        out = HERE / f"train_{method}.csv"
        if out.exists() and (HERE / f"selection_{method}.json").exists():
            meta = read(HERE / f"selection_{method}.json")
            require(meta["train_csv_sha256"] == sha(out), "Selected manifest changed")
            results[method] = csv_read(out); continue
        started = time.perf_counter()
        if method == "G":
            values = wl_codes(np.vstack([cb, ab]), graph)
        else:
            raw_c = np.load(HERE / "cache/features_candidates.npy", mmap_mode="r")
            raw_old = np.load(HERE / "cache/features_old_train.npy", mmap_mode="r")
            values = (np.vstack([raw_c, raw_old[[position[int(r["old_index"])] for r in anchors]]]).astype(np.float64)-center) / scale / np.sqrt(2)
        distance = Distances(values, method)
        chosen, trace = lcmd(distance, len(cb), ids, len(ab))
        reps, region = display_partition(distance, len(cb), ids)
        selected = [{**candidates[i], "split_role": "train", "selection_method": method, "selection_rank": rank+1,
                     "selection_distance_squared": trace[rank]["distance_squared"], "display_region_id": int(region[i]),
                     "experiment_groups": f"{method}-S;{method}-E"} for rank, i in enumerate(chosen)]
        csv_write(out, selected, CSV_FIELDS)
        csv_write(HERE / f"display_regions_{method}.csv", [{"gene_id": ids[i], "region_id": int(region[i])} for i in range(len(cb))])
        import psutil
        save(HERE / f"selection_{method}.json", {"method": method, "elapsed_seconds": time.perf_counter()-started,
             "peak_process_memory_bytes": psutil.Process().memory_info().peak_wset if os.name == "nt" else psutil.Process().memory_info().rss,
             "peak_gpu_memory_bytes": int(distance.torch.cuda.max_memory_allocated()) if distance.device.type == "cuda" else 0,
             "trace": trace, "display_center_candidate_indices": reps, "train_csv_sha256": sha(out),
             "anchor_count": 512, "distance_exact": True, "no_candidate_prefilter": True,
             "G_category_note": "Integer categories encode exact tuples and are used only in equality comparisons."})
        results[method] = selected
        del distance, values
    return results


def nearest_old_hamming(bits, train):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Binary products and sums <=120 are exactly represented in FP32.
    base = torch.from_numpy(train.astype(np.float32)).to(device)
    ones = base.sum(dim=1); result = []
    with torch.inference_mode():
        for start in range(0, len(bits), 128):
            block = torch.from_numpy(bits[start:start+128].astype(np.float32)).to(device)
            d = block.sum(dim=1, keepdim=True) + ones[None, :] - 2*(block @ base.T)
            result.extend(d.min(dim=1).values.cpu().numpy().astype(int).tolist())
    return result


def preview_regions(graph, queue):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.collections import PolyCollection
    nodes, triangles, owners = parse_ans_mesh(REFERENCE_ANS)
    regions = graph["regions"]
    cmap = ListedColormap(["#f2f5f8", "#28364b", "#d99742", "#2678bd"])
    lookup = {r["gene_id"]: r for r in queue}
    for method in "GF":
        candidates = csv_read(HERE / "candidates_train.csv")
        reps = read(HERE / f"selection_{method}.json")["display_center_candidate_indices"]
        fig, axes = plt.subplots(5, 4, figsize=(15, 17))
        examples = []
        for k, i in enumerate(reps):
            row = candidates[i]; b = bits_of([row])[0]
            logical, actual = axes[k//2, (k % 2)*2:(k % 2)*2+2]
            logical.imshow(b.reshape(20, 6).T, origin="lower", cmap=ListedColormap(["#f2f5f8", "#2678bd"]), vmin=0, vmax=1, aspect="auto")
            logical.set_title(f"{method} region {k+1} | {row['source']} | PM {b.sum()}")
            logical.set_xlabel("Angular index (physical angle decreases)"); logical.set_ylabel("Radial index outward")
            colors = np.array([3 if r["gene_index"] >= 0 and b[r["gene_index"]] else
                               0 if r["gene_index"] >= 0 or r["fixed_material_class"] == 1 else
                               2 if r["fixed_material_class"] == 3 else 1 for r in regions])
            actual.add_collection(PolyCollection(nodes[triangles], facecolors=cmap(colors[owners]), edgecolors="none", rasterized=True))
            actual.set_xlim(0, 64); actual.set_ylim(0, 64); actual.set_aspect("equal")
            actual.set_title(f"Physical 90-deg sector | {row['gene_id'][:10]}")
            actual.set_xlabel("x (mm)"); actual.set_ylabel("y (mm)")
            examples.append({"region_id": k+1, "gene_id": row["gene_id"], "is_in_femm_queue": row["gene_id"] in lookup})
        fig.suptitle(f"{method}: ten display partitions, not physical classes; air=white, PM=blue, iron=dark, winding=gold")
        fig.tight_layout(rect=(0, 0, 1, .97))
        folder = HERE / "region_previews"; folder.mkdir(exist_ok=True)
        fig.savefig(folder / f"regions_{method}.png", dpi=135); plt.close(fig)
        save(folder / f"examples_{method}.json", examples)


def regression_reuse():
    """Read and hash the four previously solved examples; never start FEMM here."""
    run = read(HISTORY_RUN / "run.json"); summary = read(HISTORY_RUN / "summary.json")
    expected = physical_config()
    for key in ("current", "problem", "airgap", "inner_angles_deg", "initial_phases_deg", "torque_multiplier"):
        require(run["config"][key] == expected[key], f"Historical regression condition differs: {key}")
    records = []
    for name in ("G2", "G3", "G4", "G5"):
        prior = next(r for r in summary["rows"] if r["id"] == name)
        raw, angles = [], []
        for inner in expected["inner_angles_deg"]:
            folder = HISTORY_RUN / name / "phase_0" / f"angle_{inner:g}"
            result = read(folder / "result.json")
            for ext in ("fem", "ans"):
                require(sha(folder / f"model.{ext}") == result[f"{ext}_sha256"], f"Regression {name}/{inner} {ext} hash mismatch")
            raw.append(result["raw_torque_nm"])
            angles.append({"inner_angle_deg": inner, **result})
        metrics = physical.torque_metrics(raw, expected)
        require(np.allclose(raw, prior["raw_torques_nm"], atol=1e-6, rtol=1e-6), "Regression waveform mismatch")
        for a, b in ((metrics["tavg_nm"], prior["reference_tavg_nm"]), (metrics["peak_to_peak_nm"], prior["reference_delta_t"])):
            require(math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-6), "Historical regression summary mismatch")
        records.append({"historical_id": name, "state": prior["state"], "row": prior["row"],
                        "reference_tavg_nm": prior["reference_tavg_nm"], "reference_delta_t_nm": prior["reference_delta_t"],
                        "angles": angles, **metrics})
    result = {"status": "passed_by_verified_existing_results", "new_femm_calls": 0, "records": records,
              "source_run_sha256": sha(HISTORY_RUN / "run.json"), "source_summary_sha256": sha(HISTORY_RUN / "summary.json"),
              "note": "Reuses G2-G5 completed six-angle results after file-hash and condition checks. G1 is excluded."}
    save(HERE / "regression_validation.json", result)
    return result


def choose_pilot(queue):
    """Choose before labels: source, membership, then magnet-count extremes."""
    chosen = {}
    def add(rows):
        for row in rows:
            if row["gene_id"] not in chosen:
                chosen[row["gene_id"]] = row
                break
    train = [r for r in queue if r["split_role"] == "train"]
    for source in "ULBP":
        for method in ("G", "F", "G;F"):
            add(sorted([r for r in train if r["source"] == source and r["selection_method"] == method], key=lambda r: r["gene_id"]))
    for source in "ULBP":
        rows = [r for r in queue if r["source"] == source]
        add(sorted(rows, key=lambda r: (int(r["magnet_cells"]), r["gene_id"])))
        add(sorted(rows, key=lambda r: (-int(r["magnet_cells"]), r["gene_id"])))
    for row in sorted(queue, key=lambda r: r["gene_id"]):
        if len(chosen) >= 20:
            break
        chosen.setdefault(row["gene_id"], row)
    require(len(chosen) == 20, "Pilot must contain twenty queue members")
    return list(chosen.values())


def finalize_seeds():
    selected = select_seeds()
    dev, test = csv_read(HERE / "dev_common.csv"), csv_read(HERE / "test_common.csv")
    queue = {}
    for method in "GF":
        for r in selected[method]:
            if r["gene_id"] in queue:
                old = queue[r["gene_id"]]
                old["selection_method"] += ";" + method
                old["experiment_groups"] += ";" + r["experiment_groups"]
                old["selection_rank"] = f"G:{old['selection_rank']};F:{r['selection_rank']}"
                old["display_region_id"] = f"G:{old['display_region_id']};F:{r['display_region_id']}"
                old["selection_distance_squared"] = "see train_G.csv and train_F.csv"
            else:
                queue[r["gene_id"]] = dict(r)
    for r in dev+test:
        require(r["gene_id"] not in queue, "Train/holdout overlap")
        queue[r["gene_id"]] = dict(r)
    rows = [queue[k] for k in sorted(queue)]
    genes, split = old_data(); historical, _ = all_history_keys(genes)
    bits = bits_of(rows)
    validate_repaired(bits)
    require(all(np.packbits(b).tobytes() not in historical for b in bits), "New seed matches known historical gene")
    require(len(selected["G"]) == len(selected["F"]) == 1400 and len(dev) == 200 and len(test) == 400, "Wrong split sizes")
    for seq in [*selected.values(), dev, test]:
        require(len({r["gene_id"] for r in seq}) == len(seq), "Within-manifest duplicate")
    role_families = [{r["family_id"] for r in seq} for seq in [selected["G"]+selected["F"], dev, test]]
    require(all(not role_families[i] & role_families[j] for i in range(3) for j in range(i)), "Family leakage")
    intersection = len({r["gene_id"] for r in selected["G"]} & {r["gene_id"] for r in selected["F"]})
    require(len(rows) == 3400-intersection, "Queue is not the manifest union")
    graph = graph_build()
    areas = np.zeros(120)
    for region in graph["regions"]:
        if region["gene_index"] >= 0:
            areas[region["gene_index"]] += region["mesh_area_mm2"]
    nearest = nearest_old_hamming(bits, genes[split["train"]])
    for i, row in enumerate(rows):
        row.update(nearest_old_train_hamming=nearest[i], magnet_area_sector_mm2=float(bits[i] @ areas),
                   magnet_area_full_motor_mm2=float(4 * (bits[i] @ areas)), status="pending")
    csv_write(HERE / "femm_queue.csv", rows)
    memberships = [{"gene_id": r["gene_id"], "split_role": r["split_role"], "experiment_group": group}
                   for r in rows for group in r["experiment_groups"].split(";")]
    require(Counter(r["experiment_group"] for r in memberships) == {g: 2000 for g in ("G-S", "F-S", "G-E", "F-E")}, "Four group role counts differ")
    csv_write(HERE / "memberships.csv", memberships)
    pilot = choose_pilot(rows); csv_write(HERE / "pilot20.csv", pilot)
    regression_reuse()
    preview_regions(graph, rows)
    stats = {"status": "seeds_ready_no_new_femm_solve", "candidate_count": 20000,
             "train_G": 1400, "train_F": 1400, "dev": 200, "test": 400, "training_intersection": intersection,
             "unique_femm_queue": len(rows), "new_angle_solve_count": 6*len(rows), "historical_regression": "four verified existing cases reused",
             "source_counts": {name: dict(Counter(r["source"] for r in seq)) for name, seq in [*selected.items(), ("dev", dev), ("test", test)]},
             "nearest_old_train_hamming": {"minimum": min(nearest), "median": float(np.median(nearest)), "maximum": max(nearest)},
             "all_new_exactly_excluded_from_history": True, "encoding_and_480_label_mapping_verified": True,
             "isolated_single_cells_remaining": 0, "repaired_queue_genes": sum(int(r["repair_changed_cells"]) > 0 for r in rows),
             "families_disjoint": True, "four_group_counts": dict(Counter(r["experiment_group"] for r in memberships)),
             "pilot20_source_counts": dict(Counter(r["source"] for r in pilot)),
             "pilot20_membership_counts": dict(Counter(r["selection_method"] for r in pilot)),
             "graph_sha256": sha(HERE / "region_graph.json"), "pilot_source_sha256": sha(__file__)}
    protected = ["pilot_config.json", "provenance.json", "candidates_train.csv", "old_anchors.csv", "parent_families.json",
                 "train_G.csv", "train_F.csv", "dev_common.csv", "test_common.csv", "femm_queue.csv", "memberships.csv", "pilot20.csv",
                 "region_graph.json", "feature_scaler.json", "selection_G.json", "selection_F.json", "regression_validation.json"]
    stats["file_sha256"] = {name: sha(HERE / name) for name in protected}
    save(HERE / "seed_validation.json", stats)
    text = (f"# 03 四组试验种子审计\n\n已生成种子，尚未对新种子运行 FEMM。\n\n"
            f"候选 20000；G/F 各 1400；公共 dev 200、test 400。G/F 训练交集 {intersection}，去重队列 {len(rows)}，共 {6*len(rows)} 个角度任务。\n\n"
            f"四组各引用 2000 个基因；G-S/G-E 共享 G 种子，F-S/F-E 共享 F 种子。历史精确重复和新 split 家族交叉检查通过。\n\n"
            f"到完整旧 train 的最近 Hamming 位差：最小 {min(nearest)}，中位数 {np.median(nearest):g}，最大 {max(nearest)}。该值仅用于事后覆盖描述。\n\n"
            f"G 使用 {len(graph['regions'])} 个真实网格区域、{len(graph['shared_edge_adjacency'])} 条共享边界邻接，WL0/1 离散类别核；无周期补边。"
            "F 使用冻结 f0 两路 256 维特征，scaler 只拟合旧 train 40000；没有读取新 FEMM 标签或按预测性能过滤。\n\n"
            "G/F 都使用同一 512 个旧训练锚点的近似旧覆盖和精确 LCMD 距离更新；未构建全候选两两距离矩阵。"
            "十区图仅作展示，区域编号不能跨 G/F 对应。P 的新父代集合互斥，但父代本身仍属于旧 train。\n\n"
            "磁体面积是参考 90°网格的三角形面积之和，完整电机按四扇区换算，含曲线离散误差。结构映射通过不代表机械可制造性或性能验证。\n\n"
            "已核验并复用 G2–G5 的 24 个历史角度结果；G1 不作为标准。pilot20 在获取新标签前固定。"
            "这次不启动 FEMM、训练或路由调参。\n\n"
            "用户追加筛查：所有新候选/dev/test 在选样前按 8 邻域修正孤立单格（边缘只用网格内邻居），直到稳定；"
            "不作一般多数平滑。修正后重新去重、排除历史并补足配额。raw_bits/repair_changed_cells 保留变化证据。"
            "m、场尺度和矩形描述修正前生成器；最终磁体用量可能改变，不能把最终分布仍称原始均匀分布。\n\n"
            "| 清单 | U | L | B | P |\n|---|---:|---:|---:|---:|\n")
    for name, counts in stats["source_counts"].items():
        text += f"| {name} | " + " | ".join(str(counts.get(s, 0)) for s in "ULBP") + " |\n"
    (HERE / "selection_audit.md").write_text(text, encoding="utf8")
    print(json.dumps({k: v for k, v in stats.items() if k != "file_sha256"}, ensure_ascii=False, indent=2), flush=True)
    return stats


def check_seed_manifest():
    validation = read(HERE / "seed_validation.json")
    for name, value in validation["file_sha256"].items():
        require(sha(HERE / name) == value, f"Frozen seed file changed: {name}")
    for name, value in read(HERE / "provenance.json")["input_sha256"].items():
        require(sha(PROJECT / name) == value, f"Frozen source changed: {name}")
    return validation


def solver_identity():
    """Inspect installed FEMM, without creating a FEMM process."""
    import winreg
    command = None
    for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"femm.ActiveFEMM\CLSID", 0, winreg.KEY_READ | view) as key:
                clsid = winreg.QueryValue(key, None)
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, rf"CLSID\{clsid}\LocalServer32", 0, winreg.KEY_READ | view) as key:
                command = winreg.QueryValue(key, None)
            break
        except FileNotFoundError:
            continue
    require(command is not None, "FEMM COM server is not registered in either registry view")
    import re
    match = re.search(r'"([^"]+\.exe)"|(.+?\.exe)', command, re.I)
    require(match is not None, "Cannot locate registered FEMM binary")
    exe = Path(match.group(1) or match.group(2))
    import win32api
    info = win32api.GetFileVersionInfo(str(exe), "\\")
    version = [info["FileVersionMS"] >> 16, info["FileVersionMS"] & 65535, info["FileVersionLS"] >> 16, info["FileVersionLS"] & 65535]
    return {"executable": str(exe), "binary_sha256": sha(exe), "file_version": version}


def solve_contract():
    import importlib.metadata
    return {"physical": physical_config(), "template_sha256": sha(physical.TEMPLATE_FILE), "mat_sha256": sha(physical.MAT_FILE),
            "mapping_source_sha256": sha(mapping.__file__), "physical_source_sha256": sha(physical.__file__),
            "runner_source_sha256": hashlib.sha256("\n".join(inspect.getsource(fn) for fn in
                                      (prepare_cases, validate_model_settings, validate_success, solve_angle, solve_gene,
                                       worker_init, dispatch_genes, failure_limit, solve_queue, aggregate_results)).encode()).hexdigest(),
            "solver": solver_identity(), "pyfemm_version": importlib.metadata.version("pyfemm")}


def scoped_rows(scope):
    check_seed_manifest()
    require(scope in ("pilot", "all"), "Scope must be pilot or all")
    rows = csv_read(HERE / ("pilot20.csv" if scope == "pilot" else "femm_queue.csv"))
    bits_of(rows)
    validate_repaired(bits_of(rows))
    return rows


@contextmanager
def batch_lock(root, scope):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "active.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Batch lock exists; inspect its PID before removing a stale lock: {lock}")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as stream:
            json.dump({"pid": os.getpid(), "scope": scope, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, stream)
        yield
    finally:
        lock.unlink(missing_ok=True)


def prepare_cases(scope="pilot", lock_owner=None):
    rows = scoped_rows(scope); contract = solve_contract(); fingerprint = digest(contract)
    root = HERE / "femm_runs" / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    if lock_owner is None:
        with batch_lock(root, scope):
            return prepare_cases(scope, lock_owner=os.getpid())
    require(lock_owner == os.getpid() and read(root / "active.lock")["pid"] == lock_owner, "Prepare requires the coordinator's batch lock")
    manifest_path = root / "contract.json"
    if manifest_path.exists():
        require(read(manifest_path) == contract, "FEMM contract hash collision or corruption")
    else:
        save(manifest_path, contract)
    cfg = contract["physical"]
    template = physical.TEMPLATE_FILE.read_text(encoding="utf8")
    positions = loadmat(physical.MAT_FILE, variable_names=["MaterialPosition"], squeeze_me=True)["MaterialPosition"]
    mapped = mapping.match_material_positions(template, positions)
    available = shutil.disk_usage(HERE).free
    require(available > len(rows)*6*len(template.encode()) + 512*1024**2, "Insufficient disk space for FEM inputs")
    for row, b in zip(rows, bits_of(rows)):
        base = mapping.replace_cell_materials(template, b)
        mapping.validate_generated_model(base, mapped, b)
        for inner in cfg["inner_angles_deg"]:
            folder = root / row["gene_id"] / f"angle_{inner:g}"
            model = folder / "model.fem"; state_path = folder / "state.json"
            content = physical.configure_fem(base, cfg, inner, 0).encode("utf8")
            identity = {"gene_id": row["gene_id"], "bits": row["bits"], "inner_angle_deg": inner,
                        "rotor_travel_deg": inner-29, "currents_a": physical.phase_currents(cfg, inner, 0),
                        "condition_fingerprint": fingerprint, "prepared_sha256": hashlib.sha256(content).hexdigest()}
            if state_path.exists():
                state = read(state_path)
                require(all(state.get(k) == v for k, v in identity.items()), f"Existing angle identity differs: {folder}")
                if state["status"] == "succeeded":
                    validate_success(folder, identity)
                else:
                    # FEMM may normalize failed/running inputs: validate against recorded hash, never assume validity.
                    require(model.exists() and sha(model) in (state["prepared_sha256"], state.get("last_fem_sha256")), f"Changed pending/failed FEM input: {folder}")
                continue
            require(not model.exists(), f"Unregistered FEM input already exists: {model}")
            folder.mkdir(parents=True, exist_ok=True); model.write_bytes(content)
            save(state_path, {**identity, "status": "pending", "attempts": []})
    save(HERE / "femm_entry.json", {"run_directory": str(root.relative_to(HERE)), "condition_fingerprint": fingerprint,
                                    "prepared_scope": scope, "prepared_genes": len(rows), "default_workers": DEFAULT_WORKERS,
                                    "new_solve_performed_by_prepare": False})
    print(f"Prepared {len(rows)} genes x 6 angles; FEMM was not started: {root}", flush=True)
    return root, rows, contract


def validate_model_settings(model, cfg, inner):
    import re
    text = Path(model).read_text(encoding="utf8")
    for name, value in cfg["problem"].items():
        found = re.findall(rf"^\[{re.escape(name)}\]\s*=\s*([^\r\n]*)", text, re.M)
        require(len(found) == 1, f"FEM field missing/repeated: {name}")
        actual = found[0].strip().strip('"')
        require(math.isclose(float(actual), value, abs_tol=1e-15, rel_tol=1e-12) if isinstance(value, (int, float)) else actual == value,
                f"FEM setting changed: {name}")
    gaps = [s for s in re.findall(r"<BeginBdry>.*?<EndBdry>", text, re.S) if '<BdryName> = "sliding_airgap"' in s]
    require(len(gaps) == 1, "Missing/repeated sliding airgap")
    for field, value in (("innerangle", inner), ("outerangle", 0), ("BdryType", 6)):
        require(math.isclose(float(re.search(rf"<{field}>\s*=\s*([^\r\n]+)", gaps[0])[1]), value, abs_tol=1e-12, rel_tol=1e-12), "Airgap setting changed")
    actual_currents = {}
    for block in re.findall(r"<BeginCircuit>.*?<EndCircuit>", text, re.S):
        name = re.search(r'<CircuitName>\s*=\s*"([^"]+)"', block)[1]
        actual_currents[name] = float(re.search(r"<TotalAmps_re>\s*=\s*([^\r\n]+)", block)[1])
        require(float(re.search(r"<TotalAmps_im>\s*=\s*([^\r\n]+)", block)[1]) == 0, "Imaginary circuit current changed")
    expected = physical.phase_currents(cfg, inner, 0)
    require(actual_currents.keys() == expected.keys() and all(math.isclose(actual_currents[k], v, abs_tol=1e-12, rel_tol=1e-12) for k, v in expected.items()), "Phase current changed")


def validate_success(folder, expected):
    state = read(folder / "state.json"); result = read(folder / "result.json")
    require(state["status"] == "succeeded", "Angle is not successful")
    for key, value in expected.items():
        require(state.get(key) == value and result.get(key) == value, f"Cached angle identity mismatch: {key}")
    require(sha(folder / "result.json") == state["result_sha256"], "Result JSON hash mismatch")
    for ext in ("fem", "ans"):
        require(sha(folder / f"model.{ext}") == result[f"{ext}_sha256"], f"Cached {ext} hash mismatch")
    contract = read(folder.parents[1] / "contract.json")
    require(digest(contract) == expected["condition_fingerprint"], "Cached condition contract changed")
    validate_model_settings(folder / "model.fem", contract["physical"], expected["inner_angle_deg"])
    require(isinstance(result["raw_torque_nm"], (int, float)) and math.isfinite(result["raw_torque_nm"]), "Invalid cached torque")
    return result


def solve_angle(folder, contract):
    import femm
    import win32com.client
    state_path = folder / "state.json"; state = read(state_path)
    keys = ("gene_id", "bits", "inner_angle_deg", "rotor_travel_deg", "currents_a", "condition_fingerprint", "prepared_sha256")
    identity = {k: state[k] for k in keys}
    if state["status"] == "succeeded":
        return validate_success(folder, identity)
    # Two total recorded attempts per angle; crashes count as attempts. No endless implicit retries.
    while len(state["attempts"]) < 2:
        require(shutil.disk_usage(folder).free > 512*1024**2, "Disk space below 512 MiB; batch paused")
        model = folder / "model.fem"
        require(sha(model) in (state["prepared_sha256"], state.get("last_fem_sha256")), "FEM input changed before retry")
        b = bits_of([identity])[0]
        content = physical.configure_fem(mapping.replace_cell_materials(physical.TEMPLATE_FILE.read_text(encoding="utf8"), b),
                                          contract["physical"], identity["inner_angle_deg"], 0).encode("utf8")
        require(hashlib.sha256(content).hexdigest() == identity["prepared_sha256"], "Rebuilt retry input differs")
        # Preserve failed attempt input/output before FEMM overwrites its working filenames.
        if state["attempts"]:
            prior = folder / f"attempt_{len(state['attempts'])}"; prior.mkdir(exist_ok=True)
            for ext in ("fem", "ans"):
                old = folder / f"model.{ext}"
                if old.exists():
                    shutil.copy2(old, prior / old.name)
        model.write_bytes(content)
        started = time.perf_counter(); connected = False
        attempt = {"number": len(state["attempts"])+1, "status": "running", "worker_pid": os.getpid(),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        state["attempts"].append(attempt); state["status"] = "running"; save(state_path, state)
        try:
            femm.HandleToFEMM = win32com.client.DispatchEx("femm.ActiveFEMM"); connected = True
            femm.main_minimize(); femm.opendocument(str(model)); femm.mi_analyze(1); femm.mi_loadsolution()
            raw = complex(femm.mo_gapintegral("sliding_airgap", 0))
            require(abs(raw.imag) <= 1e-10 and math.isfinite(raw.real), f"Invalid FEMM torque: {raw}")
            validate_model_settings(model, contract["physical"], identity["inner_angle_deg"])
            result = {**identity, "raw_torque_nm": raw.real, "fem_sha256": sha(model), "ans_sha256": sha(model.with_suffix(".ans")),
                      "elapsed_seconds": time.perf_counter()-started, "attempt_number": attempt["number"], "worker_pid": os.getpid()}
            save(folder / "result.json", result)
            attempt.update(status="succeeded", elapsed_seconds=result["elapsed_seconds"])
            state.update(status="succeeded", result_sha256=sha(folder / "result.json")); save(state_path, state)
            return result
        except BaseException as exc:
            attempt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc), elapsed_seconds=time.perf_counter()-started)
            state.update(status=attempt["status"], last_fem_sha256=sha(model)); save(state_path, state)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if connected:
                try:
                    femm.closefemm()  # Only the DispatchEx instance created in this attempt.
                except Exception:
                    pass
    return None


def worker_init(stop_event):
    global WORKER_STOP
    WORKER_STOP = stop_event
    # Coordinator handles Ctrl+C; workers finish the current angle and save it before stopping.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def solve_gene(root_string, row, contract):
    """One process owns one gene directory; its six angles run sequentially."""
    import pythoncom
    root = Path(root_string)
    folder = root / row["gene_id"]
    started = time.perf_counter()
    outcome = {"gene_id": row["gene_id"], "worker_pid": os.getpid(), "status": "running"}
    save(folder / "worker.json", outcome)
    initialized = False
    try:
        pythoncom.CoInitialize(); initialized = True
        for inner in contract["physical"]["inner_angles_deg"]:
            if WORKER_STOP is not None and WORKER_STOP.is_set():
                outcome["status"] = "paused"
                break
            save(folder / "worker.json", {**outcome, "inner_angle_deg": inner})
            if solve_angle(folder / f"angle_{inner:g}", contract) is None:
                outcome.update(status="failed", error=f"angle {inner}: automatic attempts exhausted")
                break
        else:
            outcome["status"] = "succeeded"
        record = aggregate_results(root, [row], contract, write_tables=False)[0]
        outcome["successful_angles"] = record["successful_angles"]
        if record["status"] == "succeeded":
            outcome["status"] = "succeeded"
    except Exception as exc:
        outcome.update(status="failed", error=repr(exc))
    finally:
        if initialized:
            pythoncom.CoUninitialize()
    outcome["elapsed_seconds"] = time.perf_counter()-started
    save(folder / "worker.json", outcome)
    return outcome


def failure_limit(outcomes):
    # A paused/cancelled task is not a failed gene and does not reset a failure streak.
    finished = [r["status"] == "failed" for r in outcomes if r["status"] in ("succeeded", "failed")][-50:]
    return (len(finished) >= 3 and all(finished[-3:])) or sum(finished) >= 5


def dispatch_genes(rows, root, contract, workers, record, history=(), work=solve_gene):
    """Bounded spawn pool. Only the coordinator writes shared logs and scheduling state."""
    require(isinstance(workers, int) and workers >= 1, "workers must be a positive integer")
    require(len({r["gene_id"] for r in rows}) == len(rows), "Duplicate gene jobs are not allowed")
    context = mp.get_context("spawn")
    stop = context.Event()
    outcomes = list(history)
    if failure_limit(outcomes):
        raise RuntimeError("Recorded failure threshold already reached; investigate before resuming")
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=worker_init, initargs=(stop,))
    pending = {}; next_index = 0; reason = None; submitted = 0; peak_pending = 0
    started = time.perf_counter()

    def harvest(future):
        row = pending.pop(future)
        if future.cancelled():
            return
        try:
            result = future.result()
            require(result["gene_id"] == row["gene_id"] and result["status"] in ("succeeded", "failed", "paused"), "Invalid worker response")
        except Exception as exc:
            result = {"gene_id": row["gene_id"], "status": "failed", "error": repr(exc), "elapsed_seconds": None, "worker_pid": None}
        outcomes.append(result)
        record(result)

    try:
        while pending or (next_index < len(rows) and reason is None):
            while reason is None and next_index < len(rows) and len(pending) < workers:
                row = rows[next_index]
                pending[executor.submit(work, str(root), row, contract)] = row
                next_index += 1; submitted += 1; peak_pending = max(peak_pending, len(pending))
            if not pending:
                break
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda f: pending[f]["gene_id"]):
                harvest(future)
            if failure_limit(outcomes) and reason is None:
                reason = "failure_threshold"; stop.set()
                for future in pending:
                    future.cancel()
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"; stop.set()
    except BaseException:
        stop.set()
        raise
    finally:
        if stop.is_set():
            for future in pending:
                future.cancel()
        # Running FEMM calls are allowed to finish their current angle and close their own instance.
        executor.shutdown(wait=True, cancel_futures=stop.is_set())
        for future in list(pending):
            harvest(future)
    return {"status": "completed" if reason is None else "paused", "stop_reason": reason,
            "workers": workers, "submitted_genes": submitted, "peak_inflight_jobs": peak_pending,
            "not_submitted_genes": len(rows)-submitted, "elapsed_seconds": time.perf_counter()-started}


def aggregate_results(root, rows, contract, write_tables=True):
    output = []; waveforms = []
    for row in rows:
        results, errors = [], []
        gene_folder = root / row["gene_id"]
        for inner in contract["physical"]["inner_angles_deg"]:
            folder = gene_folder / f"angle_{inner:g}"
            if not (folder / "state.json").exists():
                errors.append(f"angle {inner}: not prepared"); continue
            state = read(folder / "state.json")
            identity = {k: state[k] for k in ("gene_id", "bits", "inner_angle_deg", "rotor_travel_deg", "currents_a", "condition_fingerprint", "prepared_sha256")}
            require(identity["gene_id"] == row["gene_id"] and identity["bits"] == row["bits"] and
                    identity["condition_fingerprint"] == digest(contract) and identity["inner_angle_deg"] == inner and
                    identity["currents_a"] == physical.phase_currents(contract["physical"], inner, 0), "Result identity differs from queue/condition")
            if state["status"] == "succeeded":
                result = validate_success(folder, identity); results.append(result)
                waveforms.append({"gene_id": row["gene_id"], "split_role": row["split_role"], "inner_angle_deg": inner,
                                  "rotor_travel_deg": result["rotor_travel_deg"], **result["currents_a"],
                                  "raw_torque_nm": result["raw_torque_nm"], "fem_sha256": result["fem_sha256"], "ans_sha256": result["ans_sha256"]})
            else:
                errors.append(f"angle {inner}: {state['status']}")
        record = {"gene_id": row["gene_id"], "split_role": row["split_role"], "source": row["source"], "experiment_groups": row["experiment_groups"],
                  "magnet_cells": row["magnet_cells"], "magnet_area_sector_mm2": row["magnet_area_sector_mm2"],
                  "magnet_area_full_motor_mm2": row["magnet_area_full_motor_mm2"], "condition_fingerprint": digest(contract),
                  "status": "succeeded" if len(results) == 6 else "incomplete", "successful_angles": len(results),
                  "tavg_nm": "", "delta_t_nm": "", "elapsed_seconds_successful_attempts": sum(r["elapsed_seconds"] for r in results), "errors": "; ".join(errors)}
        if len(results) == 6:
            metrics = physical.torque_metrics([r["raw_torque_nm"] for r in results], contract["physical"])
            record.update(tavg_nm=metrics["tavg_nm"], delta_t_nm=metrics["peak_to_peak_nm"])
            save(gene_folder / "label.json", {**record, "angles": results})
        output.append(record)
    if write_tables:
        csv_write(root / "labels.csv", output)
        csv_write(root / "waveforms.csv", waveforms, ["gene_id", "split_role", "inner_angle_deg", "rotor_travel_deg", "A", "B", "C", "raw_torque_nm", "fem_sha256", "ans_sha256"])
    return output


def solve_queue(scope="pilot", workers=DEFAULT_WORKERS):
    require(isinstance(workers, int) and workers >= 1, "workers must be a positive integer")
    # Lock before prepare: two coordinators cannot prepare or solve the same condition concurrently.
    scoped_rows(scope); contract = solve_contract()
    root = HERE / "femm_runs" / digest(contract)
    with batch_lock(root, scope):
        regression_reuse()
        if scope == "all":
            pilot = aggregate_results(root, csv_read(HERE / "pilot20.csv"), contract, write_tables=False)
            require(all(r["status"] == "succeeded" for r in pilot), "Complete and inspect solve --scope pilot before starting --scope all")
        root, rows, contract = prepare_cases(scope, lock_owner=os.getpid())
        outcomes_path = root / "gene_attempts.json"
        outcomes = read(outcomes_path) if outcomes_path.exists() else []
        pending = [r for r in rows if not all(read(root/r["gene_id"]/f"angle_{a:g}/state.json")["status"] == "succeeded"
                                             for a in contract["physical"]["inner_angles_deg"])]
        execution_id = f"{time.time_ns()}_{os.getpid()}"
        execution_path = root / "executions" / f"{execution_id}.json"
        started = time.perf_counter(); observed = []
        progress = {"execution_id": execution_id, "coordinator_pid": os.getpid(), "scope": scope,
                    "status": "running", "workers": workers, "scope_genes": len(rows), "pending_at_start": len(pending),
                    "reused_complete_genes": len(rows)-len(pending), "new_successes": 0, "new_failures": 0,
                    "paused_genes": 0, "worker_pids": [], "elapsed_seconds": 0.0}
        def record(result):
            observed.append(result)
            outcomes.append({**result, "execution_id": execution_id, "workers": workers})
            save(outcomes_path, outcomes)
            for field, status in (("new_successes", "succeeded"), ("new_failures", "failed"), ("paused_genes", "paused")):
                progress[field] = sum(r["status"] == status for r in observed)
            progress["worker_pids"] = sorted({r["worker_pid"] for r in observed if r.get("worker_pid")})
            progress["elapsed_seconds"] = time.perf_counter()-started
            save(root / "progress.json", progress); save(execution_path, progress)
            print(f"FEMM {result['gene_id'][:12]}: {result['status']} | {len(observed)}/{len(pending)} | worker PID {result.get('worker_pid')}", flush=True)
        save(root / "progress.json", progress); save(execution_path, progress)
        try:
            dispatched = dispatch_genes(pending, root, contract, workers, record, history=outcomes)
            progress.update(dispatched)
        except BaseException as exc:
            progress.update(status="failed", error=repr(exc))
            raise
        finally:
            progress["elapsed_seconds"] = time.perf_counter()-started
            progress["successful_genes_per_hour"] = 3600*progress["new_successes"]/progress["elapsed_seconds"] if progress["new_successes"] else None
            save(root / "progress.json", progress); save(execution_path, progress)
            aggregate_results(root, csv_read(HERE / "femm_queue.csv"), contract)
        print(json.dumps(progress, ensure_ascii=False, indent=2), flush=True)
        if progress["status"] != "completed":
            raise RuntimeError(f"Batch paused: {progress.get('stop_reason')}; completed angles were saved")


def report_results():
    check_seed_manifest()
    entry = read(HERE / "femm_entry.json"); root = HERE / entry["run_directory"]
    contract = read(root / "contract.json")
    require(digest(contract) == entry["condition_fingerprint"], "Result contract fingerprint mismatch")
    with batch_lock(root, "report"):
        return report_locked(root, contract)


def report_locked(root, contract):
    output = aggregate_results(root, csv_read(HERE / "femm_queue.csv"), contract)
    attempts = read(root / "gene_attempts.json") if (root / "gene_attempts.json").exists() else []
    success = [r for r in attempts if r["status"] == "succeeded"]
    complete = sum(r["status"] == "succeeded" for r in output)
    timings = [r["elapsed_seconds"] for r in success if r.get("elapsed_seconds") is not None]
    executions = [read(p) for p in sorted((root / "executions").glob("*.json"))]
    measured = [r for r in executions if r.get("status") != "running" and r.get("new_successes", 0) > 0]
    latest_workers = measured[-1]["workers"] if measured else None
    comparable = [r for r in measured if r["workers"] == latest_workers]
    wall = sum(r["elapsed_seconds"] for r in comparable)
    throughput = sum(r["new_successes"] for r in comparable)/wall if wall else None
    state = {"complete_genes": complete, "incomplete_genes": len(output)-complete,
             "attempted_genes": len(attempts), "failed_gene_attempts": sum(r["status"] == "failed" for r in attempts),
             "median_seconds_per_gene": float(np.median(timings)) if timings else None,
             "p90_seconds_per_gene": float(np.percentile(timings, 90)) if timings else None,
             "measured_workers": latest_workers, "successful_genes_per_hour": throughput*3600 if throughput else None,
             "remaining_seconds_estimate_from_throughput": (len(output)-complete)/throughput if throughput else None,
             "timing_note": "Throughput uses coordinator wall time at the same worker count, including worker startup and retries; not the sum of concurrent gene durations.",
             "no_time_estimate_before_measured_success": True}
    save(root / "report.json", state); print(json.dumps(state, indent=2), flush=True)


def _parallel_probe_gene(root_string, row, contract):
    """Synthetic scheduling probe only: never imports or invokes FEMM."""
    root = Path(root_string).resolve()
    require(root.is_relative_to((HERE / "cache").resolve()), "Probe files must stay in the isolated test cache")
    folder = root / row["gene_id"]; folder.mkdir()
    start = time.time_ns()
    save(folder / "started.json", {"pid": os.getpid(), "time_ns": start})
    deadline = time.monotonic()+20
    while len(list(root.glob("*/started.json"))) < contract["first_wave"]:
        require(time.monotonic() < deadline, "Six-process startup barrier timed out")
        time.sleep(.02)
    fail = contract["mode"] == "fail" and int(row["gene_id"], 16) < 3
    dwell = .1*(int(row["gene_id"], 16)+1) if fail else 5 if contract["mode"] == "fail" else .15
    deadline = time.monotonic()+dwell
    while time.monotonic() < deadline and not WORKER_STOP.is_set():
        time.sleep(.02)
    result = {"gene_id": row["gene_id"], "worker_pid": os.getpid(), "status": "failed" if fail else
              "paused" if WORKER_STOP.is_set() else "succeeded", "elapsed_seconds": (time.time_ns()-start)/1e9,
              "started_ns": start, "finished_ns": time.time_ns(), "synthetic_not_femm": True}
    save(folder / "probe.json", result)
    return result


def parallel_selftest():
    import tempfile
    cache = HERE / "cache"; cache.mkdir(exist_ok=True)
    checks = {}
    with tempfile.TemporaryDirectory(prefix="parallel_probe_", dir=cache) as name:
        root = Path(name).resolve()
        require(root.is_relative_to(cache.resolve()), "Temporary test cleanup outside workspace")
        rows = [{"gene_id": f"{i:064x}"} for i in range(20)]
        for mode in ("success", "fail"):
            output = root / mode; output.mkdir()
            observed = []
            stats = dispatch_genes(rows, output, {"mode": mode, "first_wave": 6}, 6, observed.append, work=_parallel_probe_gene)
            require(len({r["gene_id"] for r in observed}) == len(observed), "A job was reported twice")
            require(stats["peak_inflight_jobs"] == 6, "Pool did not hold six distinct jobs")
            if mode == "success":
                require(len(observed) == 20 and all(r["status"] == "succeeded" for r in observed), "Synthetic successful queue did not finish")
                require(len({r["worker_pid"] for r in observed}) == 6, "Expected six independent worker PIDs")
                spans = sorted([(r["started_ns"], 1) for r in observed]+[(r["finished_ns"], -1) for r in observed])
                active = peak = 0
                for _, change in spans:
                    active += change; peak = max(peak, active)
                require(peak == 6, "Six process lifetimes did not overlap")
                stats["observed_concurrent_processes"] = peak
            else:
                require(stats["status"] == "paused" and stats["stop_reason"] == "failure_threshold", "Failure threshold did not stop dispatch")
                require(stats["not_submitted_genes"] > 0 and any(r["status"] == "paused" for r in observed), "In-flight jobs did not observe the stop event")
            checks[mode] = stats
        duplicate = [rows[0], rows[0]]
        try:
            dispatch_genes(duplicate, root, {}, 6, lambda r: None, work=_parallel_probe_gene)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate jobs accepted")
        lockroot = root / "lock_test"
        with batch_lock(lockroot, "test"):
            try:
                with batch_lock(lockroot, "second_coordinator"):
                    raise AssertionError("Two coordinators acquired the same lock")
            except RuntimeError:
                pass
        require(not (lockroot / "active.lock").exists(), "Batch lock leaked")
        make = lambda statuses: [{"status": s} for s in statuses]
        require(failure_limit(make(["failed", "paused", "failed", "failed"])), "Paused jobs incorrectly reset failure streak")
        require(failure_limit(make(["failed", "succeeded"]*5)), "Rolling-window failure limit failed")
        require(not failure_limit(make(["failed"]*5+["succeeded"]*50)), "Old failures did not leave the fifty-gene window")
    save(HERE / "parallel_validation.json", {"status": "passed_mock_execution_only", "default_workers": DEFAULT_WORKERS,
         "new_femm_calls": 0, "checks": checks, "duplicate_jobs_rejected": True, "exclusive_coordinator_lock": True,
         "failure_window_tests": True, "pilot_source_sha256": sha(__file__)})
    print("Parallel selftest passed: six real Python processes, separate directories, no duplicate jobs, failure pause and exclusive lock; zero FEMM calls.", flush=True)


def selftest():
    # Independent explicit WL count-vector kernel versus compressed positional category equality.
    from collections import defaultdict
    graph = {"regions": [{"gene_index": 0}, {"gene_index": 1}, {"gene_index": -1}, {"gene_index": 2}],
             "shared_edge_adjacency": [(0, 1), (1, 2), (2, 3)]}
    bits = np.zeros((8, 120), dtype=np.uint8)
    bits[:, :3] = np.array([[int(c) for c in f"{i:03b}"] for i in range(8)])
    compressed = wl_codes(bits, graph)
    vectors = []
    for b in bits:
        initial = [(i, int(b[r["gene_index"]]) if r["gene_index"] >= 0 else "iron") for i, r in enumerate(graph["regions"])]
        neighbors = defaultdict(list)
        for i, j in graph["shared_edge_adjacency"]:
            neighbors[i].append(j); neighbors[j].append(i)
        features = Counter((0, label) for label in initial)
        features.update((1, initial[i], tuple(initial[j] for j in sorted(neighbors[i]))) for i in range(4))
        vectors.append(features)
    for i in range(8):
        for j in range(8):
            explicit = sum((vectors[i].get(k, 0)-vectors[j].get(k, 0))**2 for k in vectors[i].keys() | vectors[j].keys()) / (2*4**2)
            fast = np.count_nonzero(compressed[i] != compressed[j]) / 4**2
            require(explicit == fast, "WL explicit count kernel differs")
    # Exact LCMD example checked against a full pairwise distance implementation.
    rng = np.random.default_rng(41); v = rng.normal(size=(15, 5)); ids = [f"{i:064x}" for i in range(15)]
    class Table:
        method = "test"
        def one(self, i):
            return ((v-v[i])**2).sum(axis=1)
    actual, _ = lcmd(Table(), 12, ids, 3, count=7)
    full = ((v[:, None]-v[None, :])**2).sum(axis=2); centers = [12, 13, 14]; expected = []; remaining = set(range(12))
    for _ in range(7):
        groups = defaultdict(list)
        for i in remaining:
            c = min(centers, key=lambda j: (full[i, j], ids[j])); groups[c].append(i)
        c = min(groups, key=lambda j: (-sum(full[i, j] for i in sorted(groups[j])), ids[j]))
        i = min(groups[c], key=lambda j: (-full[j, c], ids[j])); expected.append(i); centers.append(i); remaining.remove(i)
    require(actual == expected, "LCMD differs from independent dense reference")
    for background in (0, 1):
        for pos in ((0, 0), (0, 10), (3, 10), (5, 19)):
            grid = np.full((6, 20), background, dtype=np.uint8); grid[pos] ^= 1
            repaired, passes = repair_isolated(grid)
            require(np.all(repaired == background) and passes == 1, "Isolated cell repair failed")
        grid = np.full((6, 20), background, dtype=np.uint8); grid[2, 5] ^= 1; grid[3, 6] ^= 1
        require(np.array_equal(repair_isolated(grid)[0], grid), "Diagonal same-material neighbors should prevent 8-neighbor isolation")
    for _ in range(100):
        grid = rng.integers(0, 2, (6, 20), dtype=np.uint8)
        expected_mask = np.zeros((6, 20), dtype=bool)
        for r in range(6):
            for a in range(20):
                ring = [grid[y, x] for y in range(max(0, r-1), min(6, r+2)) for x in range(max(0, a-1), min(20, a+2)) if (y, x) != (r, a)]
                expected_mask[r, a] = all(v != grid[r, a] for v in ring)
        require(np.array_equal(isolated_mask(grid), expected_mask), "Vectorized isolation differs from explicit neighbor enumeration")
        repaired, _ = repair_isolated(grid)
        require(not isolated_mask(repaired).any() and np.array_equal(repair_isolated(repaired)[0], repaired), "Repair not stable")
    cfg = physical_config()
    require(np.allclose(list(physical.phase_currents(cfg, 29, 0).values()), [3.5, -1.75, -1.75]), "Initial phase/current mismatch")
    require(np.allclose(list(physical.phase_currents(cfg, 44, 0).values()), [1.75, 1.75, -3.5]), "Final current mismatch")
    print("Selftest passed: exact WL kernel, independent LCMD, isolated-cell repair, encoding/current checks.", flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="status", choices=["status", "selftest", "parallel-selftest", "generate", "features", "select", "seeds", "prepare", "solve", "report"])
    parser.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="FEMM independent processes, default 6; solve only")
    parser.add_argument("--batch-size", type=int, default=16, help="f0 FP32 feature inference batch; fixed in cache contract")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.command == "status":
        status = {"default_workers": DEFAULT_WORKERS, "default_does_not_start_femm": True}
        if (HERE / "seed_validation.json").exists():
            v = read(HERE / "seed_validation.json")
            status["seeds"] = {k: v[k] for k in ("train_G", "train_F", "dev", "test", "unique_femm_queue")}
        if (HERE / "femm_entry.json").exists():
            root = HERE / read(HERE / "femm_entry.json")["run_directory"]
            if (root / "progress.json").exists():
                status["execution"] = read(root / "progress.json")
            else:
                status["execution"] = {"status": "prepared_no_solve"}
        print(json.dumps(status, ensure_ascii=False, indent=2))
    elif args.command == "selftest":
        selftest()
    elif args.command == "parallel-selftest":
        parallel_selftest()
    elif args.command == "generate":
        generate(); graph_build()
    elif args.command == "features":
        extract_features(args.batch_size)
    elif args.command == "select":
        bootstrap(); finalize_seeds()
    elif args.command == "seeds":
        selftest(); extract_features(args.batch_size); finalize_seeds()
    elif args.command == "prepare":
        prepare_cases(args.scope)
    elif args.command == "solve":
        solve_queue(args.scope, args.workers)
    elif args.command == "report":
        report_results()


if __name__ == "__main__":
    main()
