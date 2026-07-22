# -*- coding: utf-8 -*-
"""
inference_fast.py

EoMT inference - pred mask 저장만 (metric 계산 없음).

Usage:
    python inference_fast.py \
        --config configs/dinov2/tep/semantic/eomt_large_504.yaml \
        --ckpt logs/tep_dinov2_vitl/ori_aug/checkpoints/best.ckpt \
        --split int_val \
        --output_dir results/seed1/int_val
"""

import argparse
import importlib
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from lightning import seed_everything
from tqdm import tqdm

warnings.filterwarnings("ignore")
seed_everything(0, verbose=False)


# ---------------------------------------------------------------
# Model 로드
# ---------------------------------------------------------------

def load_model(config_path: str, ckpt_path: str, device: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = tuple(config["data"]["init_args"]["img_size"])
    num_classes = config["data"]["init_args"]["num_classes"]

    enc_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_cls = getattr(
        importlib.import_module(enc_cfg["class_path"].rsplit(".", 1)[0]),
        enc_cfg["class_path"].rsplit(".", 1)[1],
    )
    enc_init = {k: v for k, v in enc_cfg.get("init_args", {}).items() if k != "ckpt_path"}
    encoder = enc_cls(img_size=img_size, **enc_init)

    net_cfg = config["model"]["init_args"]["network"]
    net_cls = getattr(
        importlib.import_module(net_cfg["class_path"].rsplit(".", 1)[0]),
        net_cfg["class_path"].rsplit(".", 1)[1],
    )
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = net_cls(masked_attn_enabled=False, num_classes=num_classes,
                      encoder=encoder, **net_kwargs)

    lit_cfg = config["model"]
    lit_cls = getattr(
        importlib.import_module(lit_cfg["class_path"].rsplit(".", 1)[0]),
        lit_cfg["class_path"].rsplit(".", 1)[1],
    )
    model_kwargs = {
        k: v for k, v in lit_cfg["init_args"].items()
        if k not in ("network", "ckpt_path", "freeze_strategy", "num_classes", "img_size")
    }
    model = lit_cls(img_size=img_size, num_classes=num_classes, network=network,
                    ckpt_path=None, **model_kwargs)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)
    print(f"Model loaded: {ckpt_path}")
    return model, img_size


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------

@torch.no_grad()
def infer_single(model, img_np: np.ndarray, img_size: tuple, device: str) -> np.ndarray:
    orig_h, orig_w = img_np.shape[:2]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    img_resized = F.interpolate(
        img_tensor.unsqueeze(0), size=img_size, mode="bilinear", align_corners=False
    )[0]
    img_batch = img_resized.unsqueeze(0).to(device)

    with torch.cuda.amp.autocast():
        mask_logits_list, class_logits_list = model.network(img_batch)

    mask_logits = F.interpolate(mask_logits_list[-1], img_size, mode="bilinear")
    logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_list[-1])

    pred = logits[0].argmax(dim=0).cpu()
    pred = F.interpolate(
        pred.unsqueeze(0).unsqueeze(0).float(),
        size=(orig_h, orig_w),
        mode="nearest",
    )[0, 0].long().numpy().astype(np.int64)
    return pred


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_path", default="./TEP_dataset/ro")
    parser.add_argument("--split", default="int_val",
                        choices=["int_val", "ext_val", "train", "tune"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    device = f"cuda:{args.device}"
    model, img_size = load_model(args.config, args.ckpt, device)

    img_dir = Path(args.data_path) / args.split / "images"
    img_paths = sorted(img_dir.glob("*.png"))
    print(f"[{args.split}] {len(img_paths)}장")

    save_dir = Path(args.output_dir) / "pred_masks"
    save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(img_paths, desc=f"Inference [{args.split}]"):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        pred_mask = infer_single(model, img_np, img_size, device)
        Image.fromarray(pred_mask.astype(np.uint8)).save(save_dir / img_path.name)

    print(f"\n완료! pred masks → {save_dir}")


if __name__ == "__main__":
    main()
