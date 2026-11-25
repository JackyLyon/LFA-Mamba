from types import SimpleNamespace
from easydict import EasyDict as edict
from tools.tester import main as tester_main
from train import build_cfg

CFG = edict(
    data="pcn",                 # "pcn" | "fsd" | "shapenet55"
    model="lfa_mamba",          # keep "lfa_mamba"
    desc="Testing LFA-Mamba",
    weights=None,                  # pretrained checkpoint path, or None
    batch=None,                  # only used by some test loaders
    workers=0,                   # use 0 for portable smoke tests; increase for full evaluation
    project=None,                # custom test output dir, or None
    seed=1128,
    device=None,                 # e.g. 0 or "0,1"
    save=1,                      # 1 to save outputs, 0 to disable
    mamba_dim=256,
)


def build_test_cfg(args):
    cfg = build_cfg(args)
    if args.dataset == "shapenet55":
        cfg.DIR.OUT_PATH = "results/shapenet55"
        cfg.DIR.TEST_PATH = "test/shapenet55"
    elif args.dataset == "pcn":
        cfg.DIR.OUT_PATH = "results/PCN/train/cau_mamba"
        cfg.DIR.TEST_PATH = "results/PCN/test/cau_mamba"
    else:
        cfg.DIR.OUT_PATH = "results/FSD/train/cau_mamba"
        cfg.DIR.TEST_PATH = "results/FSD/test/cau_mamba"
    cfg.TEST.CD_TYPE = "L2"
    cfg.CONST.WEIGHTS = args.pretrained
    return cfg


def get_args():
    return SimpleNamespace(
        dataset=CFG["data"],
        model_type=CFG.model,
        desc=CFG.desc,
        pretrained=CFG.weights,
        batch_size=CFG.batch,
        epochs=None,
        num_workers=CFG.workers,
        out_path=CFG.project,
        seed=CFG.seed,
        device=CFG.device,
        output=CFG.save,
        mamba_dim=CFG.mamba_dim,
    )


def get_cfg():
    return build_test_cfg(get_args())


def main():
    args = get_args()
    cfg = build_test_cfg(args)
    tester_main(args, cfg)


if __name__ == "__main__":
    main()
