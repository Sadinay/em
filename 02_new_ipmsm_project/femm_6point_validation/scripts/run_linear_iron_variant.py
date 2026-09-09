"""Test the corrected topology with MAT's linear mu_Eisen=4000 value."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import run_corrected_3deg_angles as experiment
from run_validation import atomic_json, validation_lock


ROOT = experiment.VALIDATION_ROOT / "corrected_pm_direction" / "linear_iron_4000"
BASE = ROOT / "base" / "model.fem"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_linear_iron_base() -> None:
    source = experiment.VALIDATION_ROOT / "corrected_pm_direction" / "base" / "model.fem"
    text = source.read_text(encoding="utf-8")
    pattern = re.compile(r"<BeginBlock>.*?<BlockName>\s*=\s*\"Pure Iron\".*?<EndBlock>", re.S)
    match = pattern.search(text)
    if not match:
        raise ValueError("Pure Iron material block not found")
    block = match.group(0)
    block = re.sub(r"(<Mu_x>\s*=\s*)[^\r\n]+", r"\g<1>4000", block, count=1)
    block = re.sub(r"(<Mu_y>\s*=\s*)[^\r\n]+", r"\g<1>4000", block, count=1)
    # Remove the nonlinear BH rows and retain an explicitly linear material.
    block = re.sub(r"<BHPoints>\s*=\s*21.*?<EndBlock>", "<BHPoints> = 0\n  <EndBlock>", block, count=1, flags=re.S)
    generated = text[: match.start()] + block + text[match.end() :]
    BASE.parent.mkdir(parents=True, exist_ok=True)
    BASE.write_text(generated, encoding="utf-8", newline="")
    topology_manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    topology_manifest["iron_material_variant"] = "linear_mu_r_4000_no_bh_curve"
    topology_manifest["model_sha256_after_iron_variant"] = sha256(BASE)
    atomic_json(BASE.parent / "manifest.json", topology_manifest)
    atomic_json(
        BASE.parent / "variant_manifest.json",
        {
            "source_model": str(source),
            "source_sha256": sha256(source),
            "generated_model": str(BASE),
            "generated_sha256": sha256(BASE),
            "only_intended_change": "Pure Iron: Mu_x/Mu_y=4000 and BHPoints=0; Sigma remains 10.44",
            "evidence": "workspace_600.mat inp.mu_Eisen=4000",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use 1..3 workers")
    make_linear_iron_base()
    experiment.BASE = BASE
    experiment.ROOT = ROOT / "angles_3deg"
    experiment.CHECKPOINT = experiment.ROOT / "checkpoint.json"
    experiment.SUMMARY = experiment.ROOT / "analysis_summary.json"
    experiment.CURVE = experiment.ROOT / "candidate_002_air0_linear_iron_angles_0_to_15.csv"
    experiment.REUSE_15_PATH = None
    experiment.VARIANT_DESCRIPTION = "linear Pure Iron: mu_r=4000, Sigma=10.44, no BH curve"
    with validation_lock():
        experiment.run(args.workers, args.timeout_seconds, args.maximum_attempts)


if __name__ == "__main__":
    main()
