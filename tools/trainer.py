import json
import os
import time

import numpy as np
import torch
import utils.data_loaders
from pprint import pprint

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from manager import Manager, Manager_shapenet55
from tools.checkpoint import filter_compatible_state_dict, print_skipped_checkpoint_keys
from models.lfa_mamba import LFAMamba
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_model_type(args):
    if args.model_type is not None:
        aliases = {
            "tna_mamba": "lfa_mamba",
            "model": "lfa_mamba",
        }
        return aliases.get(args.model_type, args.model_type)
    defaults = {
        "pcn": "lfa_mamba",
        "fsd": "lfa_mamba",
        "shapenet55": "lfa_mamba",
    }
    return defaults[args.dataset]


def build_model(args, model_type):
    if args.dataset not in {"pcn", "fsd", "shapenet55"}:
        raise ValueError(
            f"LFAMamba currently supports only PCN/FSD/ShapeNet55; dataset={args.dataset!r} is unsupported."
        )
    builders = {
        "lfa_mamba": lambda: LFAMamba(
            mamba_dim=args.mamba_dim,
            use_mamba=getattr(args, "use_mamba", True),
            mamba_axes=getattr(args, "mamba_axes", ("x", "y", "z")),
            mamba_bidirectional=getattr(args, "mamba_bidirectional", True),
            mamba_restore_order=getattr(args, "mamba_restore_order", True),
            use_local_frame=getattr(args, "use_local_frame", True),
            use_tangent_normal=getattr(args, "use_tangent_normal", True),
            use_anisotropic_scale=getattr(args, "use_anisotropic_scale", True),
            use_gate=getattr(args, "use_gate", True),
        ),
    }

    if model_type not in builders:
        raise ValueError(
            f"Unsupported model_type={model_type!r} for dataset={args.dataset!r}"
        )
    return builders[model_type]()


def prepare_environment(cfg):
    if getattr(cfg.CONST, "DEVICE", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.CONST.DEVICE)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training because this project uses CUDA-only "
            "pointnet2_ops and Chamfer3D extensions. Please run on a machine "
            "with an available NVIDIA GPU and a CUDA-matched PyTorch build."
        )


def prepare_output_dirs(cfg, args):
    timestr = time.strftime("_Log_%Y_%m_%d_%H_%M_%S", time.gmtime())
    run_tag = f"train_{args.dataset}{timestr}"
    cfg.DIR.OUT_PATH = os.path.join(cfg.DIR.OUT_PATH, run_tag)
    cfg.DIR.CHECKPOINTS = os.path.join(cfg.DIR.OUT_PATH, "checkpoints")
    cfg.DIR.LOGS = cfg.DIR.OUT_PATH
    os.makedirs(cfg.DIR.CHECKPOINTS, exist_ok=True)


def build_generic_dataloaders(cfg):
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[
        cfg.DATASET.TRAIN_DATASET
    ](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[
        cfg.DATASET.TEST_DATASET
    ](cfg)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.VAL),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=max(1, cfg.CONST.NUM_WORKERS // 2),
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False,
    )
    return train_data_loader, val_data_loader


def maybe_wrap_cuda(model):
    if torch.cuda.is_available():
        return torch.nn.DataParallel(model).cuda()
    return model


def maybe_register_grad_hooks(model, model_type):
    if model_type != "lfa_mamba":
        return
    for name, param in model.named_parameters():
        def make_hook(param_name):
            def hook(grad):
                if grad is not None and (
                    torch.isnan(grad).any() or torch.isinf(grad).any()
                ):
                    print(f"[Warning] Abnormal gradient detected: {param_name}")
                    return torch.zeros_like(grad)
                return grad

            return hook

        param.register_hook(make_hook(name))


def maybe_resume_weights(model, cfg):
    if "WEIGHTS" not in cfg.CONST:
        return
    print(f"Recovering from {cfg.CONST.WEIGHTS} ...")
    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=map_location)
    checkpoint_state, skipped = filter_compatible_state_dict(model, checkpoint["model"])
    print_skipped_checkpoint_keys(skipped)
    msg = model.load_state_dict(checkpoint_state, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(
            "Recover with non-strict loading:",
            f"missing={len(msg.missing_keys)}",
            f"unexpected={len(msg.unexpected_keys)}",
        )
    if "epoch_index" in checkpoint and "best_metrics" in checkpoint:
        print(
            f"Recover complete. epoch=#{checkpoint['epoch_index']} "
            f"best={checkpoint['best_metrics']}"
        )


def save_run_metadata(cfg, args):
    pprint(cfg)
    with open(os.path.join(cfg.DIR.LOGS, "config.json"), "w") as file:
        json.dump(cfg, file, indent=4, sort_keys=True)
    torch.save(args, os.path.join(cfg.DIR.LOGS, "args_training.pth"))


def select_manager(args, model, cfg):
    if args.dataset == "shapenet55":
        return Manager_shapenet55(model, cfg)
    return Manager(model, cfg)


def main(args, cfg):
    model_type = resolve_model_type(args)
    prepare_environment(cfg)
    require_cuda()
    prepare_output_dirs(cfg, args)
    save_run_metadata(cfg, args)

    train_data_loader, val_data_loader = build_generic_dataloaders(cfg)

    model = build_model(args, model_type)
    print(
        "Using model:",
        model.__class__.__name__,
        f"| dataset={args.dataset}",
        f"| model_type={model_type}",
        f"| CAU={getattr(model, 'use_cau', 'N/A')}",
        f"| Mamba={getattr(model, 'use_mamba', 'N/A')}",
        f"| MambaDim={getattr(model, 'mamba_dim', 'N/A')}",
        f"| MambaAxes={getattr(model, 'mamba_axes', 'N/A')}",
    )
    model = maybe_wrap_cuda(model)
    maybe_register_grad_hooks(model, model_type)
    maybe_resume_weights(model, cfg)

    manager = select_manager(args, model, cfg)
    print("cuda available", torch.cuda.is_available())
    print(f"Saving outdir: {cfg.DIR.OUT_PATH}")
    manager.train(model, train_data_loader, val_data_loader, cfg)


if __name__ == "__main__":
    raise RuntimeError("Use train.py as the training entrypoint.")
