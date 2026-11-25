from types import SimpleNamespace
from easydict import EasyDict as edict
from tools.trainer import main as trainer_main


CFG = edict(
    data="pcn",                 # "pcn" | "fsd" | "shapenet55"
    model="lfa_mamba",          # keep "lfa_mamba"
    desc="Training LFA-Mamba",
    weights=None,                  # resume checkpoint path, or None
    batch=8,                  # None -> dataset default
    epochs=300,                 # None -> dataset default
    workers=0,                # use 0 for portable smoke tests; increase for full training
    output=None,                 # custom output root, or None
    seed=0,
    device=0,                    # e.g. 0 or "0,1"
    mamba_dim=256,               # must match encoder feature dim
    use_mamba=True,
    mamba_axes=("x", "y", "z"),
    mamba_bidirectional=True,
    mamba_restore_order=True,
    use_local_frame=True,
    use_tangent_normal=True,
    use_anisotropic_scale=True,
    use_gate=True,
    fscore_loss_weight=0.05,
)


COMMON_TRAIN = dict(
    N_EPOCHS=300,
    SAVE_FREQ=1,
    LEARNING_RATE=0.0003,
    LR_MILESTONES=[50, 100, 150, 200, 250],
    LR_DECAY_STEP=50,
    WARMUP_STEPS=200,
    WARMUP_EPOCHS=10,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=0,
    LR_DECAY=200,
    GRAD_CLIP=0.3,
    ANOMALY_MIN_EPOCHS=10,
    ANOMALY_CD_FACTOR=3.0,
    ANOMALY_F_SCORE_RATIO=0.25,
    ANOMALY_F_SCORE_CD_FACTOR=1.5,
    FSCORE_LOSS_WEIGHT=0.05,
    FSCORE_LOSS_THRESHOLD=0.01,
    FSCORE_LOSS_SOFTNESS=0.002,
)


DATASET_CFG = {
    "pcn": {
        "category_file": "./datasets/PCN/PCN.json",
        "partial_path": "./datasets/PCN/%s/partial/%s/%s/%02d.pcd",
        "complete_path": "./datasets/PCN/%s/complete/%s/%s.pcd",
        "train_dataset": "PCN",
        "test_dataset": "PCN",
        "batch_size": 32,
        "device": "0,1",
        "out_path": "results/PCN",
        "test_path": "results/PCN",
        "network": dict(
            UPSAMPLE_FACTORS=[2, 2, 1, 8],
            KP_EXTENTS=[0.1, 0.1, 0.05, 0.025],
        ),
        "train": {},
        "datasets": lambda spec: edict(
            COMPLETION3D=edict(
                CATEGORY_FILE_PATH="./datasets/Completion3D.json",
                PARTIAL_POINTS_PATH="./datasets/Completion3D/%s/partial/%s/%s.h5",
                COMPLETE_POINTS_PATH="./datasets/Completion3D/%s/gt/%s/%s.h5",
            ),
            SHAPENET=edict(
                CATEGORY_FILE_PATH=spec["category_file"],
                N_RENDERINGS=8,
                N_POINTS=2048,
                PARTIAL_POINTS_PATH=spec["partial_path"],
                COMPLETE_POINTS_PATH=spec["complete_path"],
            ),
        ),
    },
    "fsd": {
        "category_file": "./datasets/FSD/FSD.json",
        "partial_path": "./datasets/FSD/FSD_partial/%s/%d.pcd",
        "complete_path": "./datasets/FSD/FSD_complete/%s.pcd",
        "train_dataset": "FSD",
        "test_dataset": "FSD",
        "batch_size": 32,
        "device": "0,1",
        "out_path": "results/FSD",
        "test_path": "results/FSD",
        "network": dict(
            UPSAMPLE_FACTORS=[2, 2, 1, 8],
            KP_EXTENTS=[0.1, 0.1, 0.05, 0.025],
        ),
        "train": {},
        "datasets": lambda spec: edict(
            COMPLETION3D=edict(
                CATEGORY_FILE_PATH="./datasets/Completion3D.json",
                PARTIAL_POINTS_PATH="./datasets/Completion3D/%s/partial/%s/%s.h5",
                COMPLETE_POINTS_PATH="./datasets/Completion3D/%s/gt/%s/%s.h5",
            ),
            SHAPENET=edict(
                CATEGORY_FILE_PATH=spec["category_file"],
                N_RENDERINGS=8,
                N_POINTS=2048,
                PARTIAL_POINTS_PATH=spec["partial_path"],
                COMPLETE_POINTS_PATH=spec["complete_path"],
            ),
        ),
    },
    "shapenet55": {
        "train_dataset": "ShapeNet55",
        "test_dataset": "ShapeNet55",
        "batch_size": 32,
        "device": "0,1,2,3",
        "out_path": "results/shapenet55",
        "test_path": "test/shapenet55",
        "network": dict(
            UPSAMPLE_FACTORS=[2, 2, 1, 8],
            KP_EXTENTS=[0.1, 0.1, 0.05, 0.025],
        ),
        "train": {},
        "datasets": lambda spec: edict(
            SHAPENET55=edict(
                CATEGORY_FILE_PATH="./datasets/ShapeNet55-34/ShapeNet-55/",
                N_POINTS=2048,
                COMPLETE_POINTS_PATH="./datasets/ShapeNet55-34/ShapeNet-55/shapenet_pc/%s",
            ),
        ),
    },
}


def build_cfg(args):
    args.dataset = str(args.dataset).strip().lower()
    if args.dataset not in DATASET_CFG:
        raise ValueError(
            f"Unsupported dataset={args.dataset!r}. "
            f"Expected one of: {', '.join(sorted(DATASET_CFG))}"
        )
    spec = DATASET_CFG[args.dataset]
    train_cfg = edict(COMMON_TRAIN.copy())
    train_cfg.update(spec["train"])
    train_cfg.BATCH_SIZE = spec["batch_size"]

    cfg = edict(
        DATASETS=spec["datasets"](spec),
        DATASET=edict(
            TRAIN_DATASET=spec["train_dataset"],
            TEST_DATASET=spec["test_dataset"],
        ),
        CONST=edict(NUM_WORKERS=8, N_INPUT_POINTS=2048, DEVICE=spec["device"]),
        DIR=edict(
            OUT_PATH=spec["out_path"],
            TEST_PATH=spec["test_path"],
        ),
        NETWORK=edict(spec["network"]),
        TRAIN=train_cfg,
        TEST=edict(METRIC_NAME="ChamferDistance", CD_TYPE="L1"),
    )

    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        cfg.TRAIN.N_EPOCHS = args.epochs
    if args.num_workers is not None:
        cfg.CONST.NUM_WORKERS = args.num_workers
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.pretrained:
        cfg.CONST.WEIGHTS = args.pretrained
    if args.device is not None:
        cfg.CONST.DEVICE = args.device
    if hasattr(args, "fscore_loss_weight") and args.fscore_loss_weight is not None:
        cfg.TRAIN.FSCORE_LOSS_WEIGHT = args.fscore_loss_weight

    return cfg


def get_args():
    return SimpleNamespace(
        dataset=CFG["data"],
        model_type=CFG.model,
        desc=CFG.desc,
        pretrained=CFG.weights,
        batch_size=CFG.batch,
        epochs=CFG.epochs,
        num_workers=CFG.workers,
        out_path=CFG.output,
        seed=CFG.seed,
        device=CFG.device,
        mamba_dim=CFG.mamba_dim,
        use_mamba=CFG.use_mamba,
        mamba_axes=CFG.mamba_axes,
        mamba_bidirectional=CFG.mamba_bidirectional,
        mamba_restore_order=CFG.mamba_restore_order,
        use_local_frame=CFG.use_local_frame,
        use_tangent_normal=CFG.use_tangent_normal,
        use_anisotropic_scale=CFG.use_anisotropic_scale,
        use_gate=CFG.use_gate,
        fscore_loss_weight=CFG.fscore_loss_weight,
    )


def get_cfg():
    return build_cfg(get_args())


def main():
    args = get_args()
    cfg = build_cfg(args)
    trainer_main(args, cfg)


if __name__ == "__main__":
    main()
