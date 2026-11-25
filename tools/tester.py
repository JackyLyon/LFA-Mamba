import json
import os

import numpy as np
import torch
import utils.data_loaders

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
            "improved_cau_only": "lfa_mamba",
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
            "CUDA is required for testing because this project uses CUDA-only "
            "pointnet2_ops and Chamfer3D extensions. Please run on a machine "
            "with an available NVIDIA GPU and a CUDA-matched PyTorch build."
        )


def maybe_wrap_cuda(model):
    if torch.cuda.is_available():
        return torch.nn.DataParallel(model).cuda()
    return model


def load_checkpoint(model, cfg):
    if not getattr(cfg.CONST, "WEIGHTS", None):
        raise ValueError(
            "No checkpoint was configured. Set CFG.weights in test.py to the "
            "checkpoint path you want to evaluate."
        )
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
    return checkpoint


def prepare_test_dirs(cfg, args):
    if args.out_path:
        cfg.DIR.TEST_PATH = args.out_path
    cfg.DIR.RESULTS = os.path.join(
        cfg.DIR.TEST_PATH, "outputs" if args.dataset == "shapenet55" else "results"
    )
    cfg.DIR.LOGS = cfg.DIR.TEST_PATH
    os.makedirs(cfg.DIR.RESULTS, exist_ok=True)
    os.makedirs(cfg.DIR.LOGS, exist_ok=True)


def build_generic_test_loader(cfg):
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    return torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
        batch_size=1,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False,
    )


def print_run_info(model, args, model_type, cfg):
    print(
        "Using model:",
        model.__class__.__name__,
        f"| dataset={args.dataset}",
        f"| model_type={model_type}",
        f"| CAU={getattr(model, 'use_cau', 'N/A')}",
        f"| Mamba={getattr(model, 'use_mamba', 'N/A')}",
        f"| MambaDim={getattr(model, 'mamba_dim', 'N/A')}",
    )
    print("cuda available", torch.cuda.is_available())
    print(f"Saving outdir: {cfg.DIR.TEST_PATH}")


def maybe_save_test_metadata(cfg, args):
    metadata = {k: v for k, v in vars(args).items()}
    with open(os.path.join(cfg.DIR.LOGS, "test_args.json"), "w") as file:
        json.dump(metadata, file, indent=4, sort_keys=True)


def run_standard_test(args, cfg, model_type):
    test_data_loader = build_generic_test_loader(cfg)
    model = build_model(args, model_type)
    print_run_info(model, args, model_type, cfg)
    model = maybe_wrap_cuda(model)
    load_checkpoint(model, cfg)

    if args.dataset == "shapenet55":
        manager = Manager_shapenet55(model, cfg)
        manager.test(
            cfg,
            model,
            test_data_loader,
            outdir=cfg.DIR.RESULTS if args.output else None,
            mode="median",
        )
        return

    manager = Manager(model, cfg)
    manager.test(
        cfg,
        model,
        test_data_loader,
        outdir=cfg.DIR.RESULTS if args.output else None,
    )


def main(args, cfg):
    set_seed(args.seed)
    prepare_environment(cfg)
    require_cuda()
    prepare_test_dirs(cfg, args)
    maybe_save_test_metadata(cfg, args)

    model_type = resolve_model_type(args)
    run_standard_test(args, cfg, model_type)


if __name__ == "__main__":
    raise RuntimeError("Use test.py as the testing entrypoint.")
