import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from models.lfa_mamba import LFAMamba


SUPPORTED_EXTENSIONS = {".pcd", ".ply"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run point cloud completion inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more input point clouds or directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the completed point cloud(s).",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Filename glob used when an input path is a directory. Default: *",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Inference device. pointnet2_ops in this project requires CUDA.",
    )
    return parser.parse_args()


def validate_paths(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")


def collect_input_files(inputs, pattern):
    files = []
    for input_path in inputs:
        if input_path.is_file():
            if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported point cloud format: {input_path}. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                )
            files.append(input_path)
            continue

        if input_path.is_dir():
            matched = sorted(
                path
                for path in input_path.rglob(pattern)
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not matched:
                raise FileNotFoundError(
                    f"No supported point clouds found under directory: {input_path}"
                )
            files.extend(matched)
            continue

        raise FileNotFoundError(f"Input point cloud or directory not found: {input_path}")

    deduped = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def build_model(device):
    model = LFAMamba().to(device)
    model.eval()
    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model_keys = model.state_dict().keys()
    ckpt_has_module = next(iter(state_dict)).startswith("module.")
    model_has_module = next(iter(model_keys)).startswith("module.")

    if ckpt_has_module and not model_has_module:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif model_has_module and not ckpt_has_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint keys do not match model.\n"
            f"Missing keys: {len(missing)}\n"
            f"Unexpected keys: {len(unexpected)}\n"
            f"First missing keys: {missing[:5]}\n"
            f"First unexpected keys: {unexpected[:5]}"
        )


def load_point_cloud(input_path, device):
    point_cloud = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Point cloud is empty: {input_path}")
    return torch.from_numpy(points).unsqueeze(0).to(device)


def run_inference(model, partial):
    with torch.no_grad():
        output = model(partial)
    if isinstance(output, (list, tuple)):
        output = output[-1]
    return output.squeeze(0).detach().cpu().numpy()


def save_point_cloud(points, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if not o3d.io.write_point_cloud(str(output_path), point_cloud):
        raise IOError(f"Failed to write point cloud: {output_path}")


def resolve_output_path(output_path, input_path, multiple_inputs):
    if not multiple_inputs:
        if output_path.suffix:
            return output_path
        return output_path / f"{input_path.stem}.ply"

    suffix = input_path.suffix or ".pcd"
    parent_name = input_path.parent.name if input_path.parent.name else "root"
    return output_path / parent_name / f"{input_path.stem}_completed{suffix}"


def main():
    args = parse_args()
    validate_paths(args)

    if args.device == "cpu":
        raise RuntimeError(
            "This project depends on CUDA-only pointnet2_ops during inference. "
            "Please run infer.py on a machine with an available NVIDIA GPU, "
            "or rebuild the custom ops with CPU support."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")

    input_files = collect_input_files(args.input, args.pattern)
    multiple_inputs = len(input_files) > 1 or any(path.is_dir() for path in args.input)

    if multiple_inputs and args.output.suffix:
        raise ValueError(
            "When inferring multiple files, --output must be a directory, not a file path."
        )

    device = torch.device(args.device)
    model = build_model(device)
    load_checkpoint(model, args.checkpoint, device)

    saved_outputs = []
    for input_path in input_files:
        partial = load_point_cloud(input_path, device)
        completed = run_inference(model, partial)
        output_path = resolve_output_path(args.output, input_path, multiple_inputs)
        save_point_cloud(completed, output_path)
        saved_outputs.append(output_path)
        print(f"Saved {input_path} -> {output_path}")

    print(f"Inference finished. Saved {len(saved_outputs)} file(s).")


if __name__ == "__main__":
    main()
