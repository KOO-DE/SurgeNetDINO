"""
convert_surgenetdino_weights.py

SurgeNetDINO pretrained weights (.pth)를 EoMT가 ckpt_path로 로드할 수 있는
형식으로 변환합니다.

SurgeNetDINO .pth 키 구조 (timm backbone):
    blocks.0.attn.qkv.weight
    blocks.0.norm1.weight
    ...
    norm.weight
    patch_embed.proj.weight

EoMT lightning_module state_dict 키 구조:
    network.encoder.backbone.blocks.0.attn.qkv.weight
    network.encoder.backbone.blocks.0.norm1.weight
    ...
    network.encoder.backbone.norm.weight
    network.encoder.backbone.patch_embed.proj.weight

따라서 prefix "network.encoder.backbone." 를 붙여주기만 하면 됩니다.
EoMT의 _load_ckpt + load_state_dict(strict=False) 덕분에
class_head, mask_head, q 등 EoMT-only 키는 missing으로 처리되어 무시됩니다.

Usage:
    python convert_surgenetdino_weights.py \
        --src /path/to/DINOv2_ViTb14_size336_SurgeNetXL.pth \
        --dst /path/to/DINOv2_ViTb14_size336_SurgeNetXL_eomt.pth

    또는 여러 파일 한 번에:
    python convert_surgenetdino_weights.py --all --src_dir /path/to/weights/
"""

import argparse
from pathlib import Path
import torch


EOMT_BACKBONE_PREFIX = "network.encoder.backbone."


def convert(src_path: Path, dst_path: Path, dry_run: bool = False):
    print(f"\n[Convert] {src_path.name}")

    state_dict = torch.load(src_path, map_location="cpu", weights_only=True)

    # 혹시 Lightning checkpoint 형식이면 state_dict 추출
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # 키 확인
    sample_keys = list(state_dict.keys())[:5]
    print(f"  Sample original keys: {sample_keys}")

    # 이미 변환된 파일인지 확인
    if any(k.startswith(EOMT_BACKBONE_PREFIX) for k in state_dict.keys()):
        print("  ⚠️  Already converted (prefix already present). Skipping.")
        return

    # prefix 추가
    new_state_dict = {
        EOMT_BACKBONE_PREFIX + k: v
        for k, v in state_dict.items()
    }

    sample_new_keys = list(new_state_dict.keys())[:5]
    print(f"  Sample converted keys: {sample_new_keys}")
    print(f"  Total keys: {len(new_state_dict)}")

    if dry_run:
        print("  [dry_run] Not saving.")
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, dst_path)
    print(f"  ✅ Saved → {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert SurgeNetDINO weights for EoMT")
    parser.add_argument("--src", type=str, help="Source .pth file path")
    parser.add_argument("--dst", type=str, help="Destination .pth file path")
    parser.add_argument("--all", action="store_true",
                        help="Convert all .pth files in --src_dir")
    parser.add_argument("--src_dir", type=str,
                        help="Directory containing SurgeNetDINO .pth files (used with --all)")
    parser.add_argument("--dst_dir", type=str, default=None,
                        help="Output directory for converted files (default: same as src_dir)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print key mapping without saving")
    args = parser.parse_args()

    if args.all:
        src_dir = Path(args.src_dir)
        dst_dir = Path(args.dst_dir) if args.dst_dir else src_dir / "eomt_converted"
        pth_files = sorted(src_dir.glob("*.pth"))
        print(f"Found {len(pth_files)} .pth files in {src_dir}")
        for src_path in pth_files:
            dst_path = dst_dir / (src_path.stem + "_eomt.pth")
            convert(src_path, dst_path, dry_run=args.dry_run)
    else:
        if not args.src or not args.dst:
            parser.error("--src and --dst are required unless --all is used")
        convert(Path(args.src), Path(args.dst), dry_run=args.dry_run)


if __name__ == "__main__":
    main()