import os
import io
import re
import lmdb
import math
import json
import time
import pickle as pkl
import warnings
import random
import string
import argparse
import pathlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import unicodedata

from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from timm.scheduler import CosineLRScheduler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.utils as vutils

from model import (
    Sequential,
    Identity,
    MLP,
    NormalSeparat,
    NormalShared,
    SlotAttention,
    LearntPositionalEmbedding,
    Project,
    TransformerDecoderLayer,
    TransformerDecoder,
    ARRandTransformerDecoder,
    DINOSAUR,
    build_ocl_model,
)
from utils import set_seed, AlphaWeightedMSELoss, interpolat_argmax_attent, draw_segmentation_np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token


def remove_and_remember_cls(feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    feats: [1, T, D] or [T, D]; assume first token is CLS if present.
    Returns (feats_wo_cls, cls_token)
    """
    if feats.dim() == 2:
        feats = feats.unsqueeze(0)
    cls_tok = feats[:, :1, :].contiguous().clone()
    body = feats[:, 1:, :].contiguous().clone()
    return body, cls_tok


def compress_tokens_by_slots(
        attn_qk: torch.Tensor,
        target_num: int,
        top_k: int = 1,
        has_cls: bool = True,
        pad_mode: str = 'att_score',
        keep_always: Optional[List[int]] = None) -> List[int]:
    """
    Select token indices by argmax over tokens per slot.
    attn_qk: [1, S, T]  -> argmax/topk over token dim (last dim).
    keep_always: Choose to save a list of exact indices of the tokens.
    pad_mode: 'random' | 'att_score' | 'None'
    Returns a sorted list of INDICES IN THE "WITH-CLS" SPACE if has_cls=True (i.e., 0 is CLS).
    """
    # assert attn_qk.shape[0] == 1, "Batch size must be 1."
    assert pad_mode in ['random', 'att_score', 'None'], "pad_mode must be 'random' or 'att_score' or 'None'."
    if attn_qk.dim() == 2:
        attn_qk = attn_qk.unsqueeze(0)
    S = attn_qk.size(1)
    Np = attn_qk.size(2)

    _, top_idx = torch.topk(attn_qk, dim=2, k=top_k)  # [1, S, top_k]
    top_idx = top_idx.view(-1).tolist()

    if has_cls:
        top_idx = [i + 1 for i in top_idx]
        pick = set(top_idx)
        pick.add(0)
    else:
        pick = set(top_idx)

    # force keep
    if keep_always:
        pick.update(keep_always)

    # pad if less than target_num
    if len(pick) < target_num and pad_mode != 'None':
        if has_cls:
            pool = list(set(range(1, Np + 1)) - pick)  # 1..Np
        else:
            pool = list(set(range(Np)) - pick)  # 0..Np-1
        if pool:
            if pad_mode == 'random':
                random.shuffle(pool)
                need = target_num - len(pick)
                pick.update(pool[:need])

    out = list(pick)
    if len(out) > target_num:
        out = out[:target_num]
    out.sort()
    return out


def index_to_bool_masks(seg_idx_hw: torch.Tensor | np.ndarray, S: int) -> np.ndarray:
    """(H,W) -> (H,W,S)"""
    if isinstance(seg_idx_hw, torch.Tensor):
        seg_idx_hw = seg_idx_hw.cpu().numpy()
    return (seg_idx_hw[..., None] == np.arange(S)[None, None, :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True, )
    parser.add_argument("--layer_id", type=int, default=8, help="The layer id to be hooked.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of samples to process.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocl = build_ocl_model()
    ocl.load_state_dict(torch.load(args.weight_path, map_location=device), strict=True)
    ocl.to(device)
    ocl.eval()

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

    vision_tower = model.get_vision_tower()
    vision = vision_tower.vision_tower
    if hasattr(vision, "vision_model"):
        vision.vision_model.config.output_attentions = True
    target_layer = vision.vision_model.encoder.layers

    # Load data
    samples = [os.path.join(args.image_root, f) for f in os.listdir(args.image_root) if
               f.endswith(".jpg") or f.endswith(".png")]
    print(f"Number of samples: {len(samples)}")
    if args.limit > 0:
        samples = samples[:args.limit]

    for idx, img_fp in enumerate(samples):
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

        print(f"[INFO] original image shape: {image_tensor.shape}")

        # Register hook to get object features
        state = {
            "compressed_ids": None,
            "attn_qk": None,
            "busy": False,
            "handle_get_features": None,

            "handle_get_attnScores": None,
            "handle_alternate_tokens": None,
        }

        target_num = 65
        att_selection_mode = "meanq"

        def hook_gf(module, input, output):
            if state["busy"] or state["compressed_ids"] is not None or state["attn_qk"] is not None:
                return

            if isinstance(output, torch.Tensor):
                input_features = output.detach()
            elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
                input_features = output[0].detach()
            else:
                raise TypeError("Unsupported object feature type!")

            state["busy"] = True
            if state["handle_get_features"] is not None:
                state["handle_get_features"].remove()
                state["handle_get_features"] = None
            try:
                with torch.inference_mode():
                    input_features, _ = remove_and_remember_cls(
                        input_features.to(device=input_features.device, dtype=torch.float32))
                    attn_qk = ocl(input_features.to(device=input_features.device, dtype=torch.float32),
                                  64 // 1)[1].detach()
                    compressed_ids = compress_tokens_by_slots(
                        attn_qk,
                        target_num,
                        1,
                        True,
                        "att_score",
                    )
                    state["compressed_ids"] = compressed_ids
                    state["attn_qk"] = attn_qk
            finally:
                state["busy"] = False

        def hook_att(module, input, output):
            if state["compressed_ids"] is None:
                return

            if len(state["compressed_ids"]) == target_num:
                return
            elif len(state["compressed_ids"]) > target_num:
                raise ValueError(
                    "[!ERROR!] The number of compressed tokens should not exceed the target number of tokens.")
            num_gap = target_num - len(state["compressed_ids"])

            if isinstance(output, (tuple, list)) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
                attn_scores = output[1].detach()  # [B, H, N, N]
            else:
                return

            attn_scores = attn_scores.mean(dim=1)  # [B, N, N]

            if att_selection_mode == "cls":
                attn_scores = attn_scores[:, 0, :]  # [B, N]
            else:
                attn_scores = attn_scores.mean(dim=1)  # [B, N]

            N = attn_scores.numel()

            cur = state["compressed_ids"]
            order = torch.argsort(-attn_scores[0])  # [N]，B=1
            pool = [i for i in range(N) if i not in set(cur)]
            att_rank = [int(i) for i in order.tolist() if i in pool]
            add = att_rank[:max(0, num_gap)]
            cur += add

            if len(cur) < target_num:
                rest = [i for i in pool if i not in set(add)]
                extra = rest[:(target_num - len(cur))]
                cur += extra

            state["compressed_ids"] = sorted(cur[:target_num])

            if state["handle_get_attnScores"] is not None:
                state["handle_get_attnScores"].remove()
                state["handle_get_attnScores"] = None

        state["handle_get_features"] = target_layer[args.layer_id].register_forward_hook(hook_gf)
        state["handle_get_attnScores"] = target_layer[-2].register_forward_hook(hook_att)

        # Vision Tower activate to get features
        with torch.inference_mode():
            _ = model.encode_images(image_tensor)

        retained_indices = state["compressed_ids"][1:]
        kept_idx_wo_cls = torch.tensor([i - 1 for i in retained_indices],
                                       device=image_tensor.device, dtype=torch.long)
        print(f"[INFO] retained indices: {kept_idx_wo_cls}")
        attent_qk = state["attn_qk"]

        B = image_tensor.size(0)
        H, W = 24, 24               # patch grid
        H_out, W_out = 336, 336

        num_kept = kept_idx_wo_cls.numel()   # kept token

        keep_mask = torch.zeros(
            B, num_kept, H, W, device=image_tensor.device, dtype=torch.float32
        )

        b = 0
        for j, t_idx in enumerate(kept_idx_wo_cls.tolist()):
            h = t_idx // W
            w = t_idx % W
            keep_mask[b, j, h, w] = 1.0

        keep_mask_up = torch.nn.functional.interpolate(
            keep_mask, size=(H_out, W_out), mode="nearest"
        )

        vis_images = []
        for b in range(B):
            mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
            std = torch.tensor(image_processor.image_std).view(3, 1, 1)
            img = (
                image_tensor[b].detach().cpu() * std + mean
            ).clamp(0, 1).permute(1, 2, 0)

            # keep_mask_up[b]: (num_kept, H_out, W_out)
            masks_hws = keep_mask_up[b].detach().cpu().numpy() > 0.5
            masks_hws = np.transpose(masks_hws, (1, 2, 0))  # (H_out,W_out,num_kept)

            vis = draw_segmentation_np(img.numpy(), masks_hws, alpha=0.5)
            vis_images.append(vis)

        for i, vis in enumerate(vis_images):
            Image.fromarray(vis).save(f"{out_dir}/{idx:05d}-kept-{i:03d}.png")


if __name__ == "__main__":
    main()
