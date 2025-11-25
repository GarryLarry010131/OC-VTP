import os
import json
from typing import List, Dict, Any
from pathlib import Path
import argparse
import random

from tqdm import tqdm
from PIL import Image

import torch
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

import re
from datetime import datetime

from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

import lmdb
import pickle as pkl
import io


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True, )
    parser.add_argument("--layer_id", type=int, default=8, help="The layer id to be hooked.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of samples to process.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lmdb_path", type=str, required=True,
                        help="Path to LMDB file for storing features and tokens.")
    parser.add_argument("--write_freq", type=int, default=64, help="Commit to LMDB every N samples.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device=device
    )
    model.eval()

    # pinpoints = [[672, 672]]
    # if hasattr(model, "config") and hasattr(model.config, "image_grid_pinpoints"):
    #     model.config.image_grid_pinpoints = pinpoints
    #
    # if getattr(tokenizer, "pad_token_id", None) is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    if hasattr(model, "config") and hasattr(model.config, "image_aspect_ratio"):
        model.config.image_aspect_ratio = None

    # Register hook to get object features
    object_features = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            object_features.append(output.detach().to(torch.float16).cpu())
        elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
            object_features.append(output[0].detach().to(torch.float16).cpu())

    vision_tower = model.get_vision_tower()
    vision = vision_tower.vision_tower
    target_layer = vision.vision_model.encoder.layers
    handle = target_layer[args.layer_id].register_forward_hook(hook_fn)

    # Load data
    samples = [os.path.join(args.image_root, f) for f in os.listdir(args.image_root) if
               f.endswith(".jpg") or f.endswith(".png")]
    print(f"Number of samples: {len(samples)}")
    if args.limit > 0:
        samples = samples[:args.limit]

    lmdb_env = lmdb.open(
        args.lmdb_path,
        map_size=1024 ** 4,
        subdir=False,
        readonly=False,
        meminit=False,
    )
    txn = lmdb_env.begin(write=True)
    lmdb_keys = []

    try:
        save_bar = tqdm(samples, desc="LLaVA@GQA dump -> LMDB")
        for idx, img_fp in enumerate(save_bar):
            object_features.clear()
            # Load images
            try:
                image = Image.open(img_fp).convert("RGB")
            except Exception as e:
                print(f"[WARN] fail open {img_fp}: {e}")
                continue

            image = image.resize((336, 336), Image.BICUBIC)

            # Image preprocessing
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [t.to(device, dtype=getattr(model, "dtype", torch.float16)) for t in image_tensor]
            else:
                if image_tensor.ndim == 5:  # (B, T, 3, H, W) -> (B*T, 3, H, W)
                    B, T, C, H, W = image_tensor.shape
                    image_tensor = image_tensor.view(B * T, C, H, W)
                image_tensor = image_tensor.to(device, dtype=getattr(model, "dtype", torch.float16))

            save_shape = image_tensor.shape
            save_bar.set_postfix(shape=save_shape)

            # Vision Tower activate to get features
            with torch.inference_mode():
                _ = model.encode_images(image_tensor)

            if len(object_features) == 0:
                feats_np = None
            else:
                feats = object_features[0]  # Tensor: [B, tokens, hidden] torch.Size([1, 577, 1024])
                feats_np = feats.numpy()

            # To LMDB
            sample_key = f"{idx:08d}".encode("ascii")
            lmdb_keys.append(sample_key)
            sample_dict = {
                "idx": idx,
                "feats": feats_np,  # bytes or None
            }
            txn.put(sample_key, pkl.dumps(sample_dict))

            # Commit
            if (idx + 1) % args.write_freq == 0:
                txn.commit()
                txn = lmdb_env.begin(write=True)

        txn.commit()
        txn = lmdb_env.begin(write=True)
        txn.put(b"__keys__", pkl.dumps(lmdb_keys))
        txn.commit()
    finally:
        handle.remove()
        lmdb_env.close()

    print(f"[OK] LMDB -> {args.lmdb_path}")
    print(f"[OK] total={len(lmdb_keys)}, write_freq={args.write_freq}")
    print("[TIP] LMDB includes: feats(np.ndarray) only, plus idx.")


if __name__ == "__main__":
    main()
