from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fem_raster import rasterize_fem
from model import StructureCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict J directly from a FEM file and T_min.")
    parser.add_argument("--fem", type=Path, required=True)
    parser.add_argument("--t-min", type=float, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_config = checkpoint["config"]["image"]
    image, raster_metadata = rasterize_fem(
        args.fem.resolve(),
        height=int(image_config["height"]),
        width=int(image_config["width"]),
        bbox=tuple(float(value) for value in image_config["bbox"]),
    )
    model = StructureCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    t_scaled = (
        args.t_min - float(checkpoint["t_min_mean"])
    ) / float(checkpoint["t_min_std"])
    t_tensor = torch.tensor([[t_scaled]], dtype=torch.float32)
    with torch.no_grad():
        prediction_scaled = float(model(image_tensor, t_tensor).item())
    prediction = (
        prediction_scaled * float(checkpoint["target_std"])
        + float(checkpoint["target_mean"])
    )
    print(json.dumps({
        "fem_path": str(args.fem.resolve()),
        "t_min": args.t_min,
        "predicted_j": prediction,
        "raster_metadata": raster_metadata,
        "warning": "流程验证模型仅有11个独立训练结构，预测值不能用于工程决策。",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
