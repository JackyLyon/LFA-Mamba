# LFA-Mamba: Local-frame adaptive state-space network for point cloud completion of femoral surface defect

This repository provides the official implementation of LFA-Mamba for point cloud completion. It includes training, evaluation, and inference pipelines for:

- `PCN`
- `FSD`
- `ShapeNet55`

The main model implementation lives in [models/lfa_mamba.py](models/lfa_mamba.py), and the training / evaluation entrypoints are [train.py](train.py), [test.py](test.py), and [infer.py](infer.py).

This repository is intended to contain source code, lightweight dataset split files, and documentation. Full datasets, generated point clouds, training outputs, and model checkpoints should be distributed separately and placed in the paths described below.

## Overview

<p align="center">
  <img src="figs/LFA-Mamba.jpg" alt="FSD qualitative comparison" width="900">
  <br>
  <em>Architecture and information flow of LFA-Mamba.</em>
</p>

## Environment

- Python `3.9` is recommended
- NVIDIA GPU + CUDA are required for training, evaluation, and inference
- `pointnet2_ops` and `Chamfer3D` are custom CUDA extensions and must be compiled locally

Install dependencies:

```bash
pip install -r requirements.txt
bash install.sh
```

`install.sh` builds:

- `pointnet2_ops_lib`
- `Chamfer3D`

Install a CUDA-matched PyTorch build first from the official PyTorch instructions, then run `pip install -r requirements.txt`.

## Repository Layout

```text
.
├── train.py                # training config + entrypoint
├── test.py                 # evaluation config + entrypoint
├── infer.py                # single point-cloud inference
├── tools/
│   ├── trainer.py
│   └── tester.py
├── models/
├── utils/
├── datasets/
│   ├── PCN/
│   ├── FSD/
│   └── ShapeNet55-34/      # expected for ShapeNet55
├── pointnet2_ops_lib/
└── Chamfer3D/
```

## Dataset Format

### FSD

The default FSD config in [train.py](train.py) expects:

```text
datasets/FSD/
├── FSD.json
├── FSD_complete/
│   └── <model_id>.pcd
└── FSD_partial/
    └── <model_id>/
        └── <view_id>.pcd
```

Download sources:

- The femoral surface dataset (FSD) download link: <https://drive.google.com/drive/folders/1xlEOMSqH7ucFeAWtRmZbCUmrKSVTfJsD?usp=drive_link>
- `FSD.json` should contain `train`, `val`, and `test` splits.

### PCN

The default PCN config expects the standard PCN layout:

```text
datasets/PCN/
├── PCN.json
├── train/
├── val/
└── test/
```

Download sources:

- Official PCN project page: <https://wentaoyuan.github.io/pcn/>
- Processed dataset link commonly used by PoinTr: <https://github.com/yuxumin/PoinTr/blob/master/DATASET.md>

### ShapeNet55

The ShapeNet55 loader expects:

```text
datasets/ShapeNet55-34/ShapeNet-55/
├── train.txt
├── test.txt
└── shapenet_pc/
```

Download sources:

- PoinTr dataset instructions and processed ShapeNet55/34 links: <https://github.com/yuxumin/PoinTr/blob/master/DATASET.md>

## Checkpoints

Pretrained checkpoints are not committed to the Git repository because they are large binary artifacts. Place externally provided checkpoints under paths such as:

```text
/path/to/checkpoints/<dataset_name>/ckpt-best.pth
```

Pretrained weights and the FSD dataset are available from Google Drive:

| Resource | Link |
| --- | --- |
| FSD pretrained weight | [Download](https://drive.google.com/file/d/1x5KbjtYUBfkkJnphVa5APev8XqYTXzXw/view?usp=drive_link) |
| PCN pretrained weight | [Download](https://drive.google.com/file/d/12VIm3Zl2tUcYDsE5-Qu5FSC4BgOchQLS/view?usp=drive_link) |
| ShapeNet55 pretrained weight | [Download](https://drive.google.com/file/d/1ldhDr492svy3Hu4yPgjJri737vprCdK0/view?usp=drive_link) |


## Training

Edit the `CFG` block in [train.py](train.py), then run:

```bash
python train.py
```

Key fields:

- `data`: `pcn`, `fsd`, or `shapenet55`
- `weights`: checkpoint path for resume / finetune, or `None`
- `batch`
- `epochs`
- `workers`
- `output`
- `device`
- `mamba_dim`

Training outputs are written under `results/...` and typically include:

- `training.txt`
- `testing.txt`
- `config.json`
- `metrics_curves.png`
- `training_metrics.xlsx`
- `checkpoints/ckpt-best.pth`

## Evaluation

Edit the `CFG` block in [test.py](test.py), then run:

```bash
python test.py
```

This will load a checkpoint, evaluate on the configured dataset split, and optionally save completed point clouds under the test output directory.

## Inference

`infer.py` does not define default file paths. Provide the checkpoint, input point cloud or directory, and output path explicitly:

```bash
python infer.py \
  --checkpoint /path/to/checkpoint.pth \
  --input /path/to/input_point_cloud.pcd \
  --output /path/to/output_point_cloud.ply
```

For directory input, `--output` must be a directory:

```bash
python infer.py \
  --checkpoint /path/to/checkpoint.pth \
  --input /path/to/input_point_cloud_directory \
  --output /path/to/output_directory \
  --pattern "*.pcd"
```

`infer.py` requires CUDA because the current custom ops are built for GPU use.


## Notes

- The FSD training pipeline currently validates on the `val` split rather than reusing `test`.
- Validation sampling is deterministic to keep per-epoch metrics comparable.
- `testing.txt` only records accepted validation epochs, so anomaly rollback points are not mixed into the final validation curve.

## License

This code is released under the [LFA-Mamba Academic Research License](LICENSE). It permits non-commercial academic and research use only. Commercial use requires prior written permission from the copyright holder.

## Troubleshooting

- If `pointnet2_ops` or `Chamfer3D` import fails, rebuild them with `bash install.sh`.
- If `mamba_ssm` import fails, make sure `mamba-ssm` is installed in the active environment.
- If Open3D fails to load `.pcd` files, check that your dataset files are valid point clouds and that the installed Open3D version matches your Python version.
