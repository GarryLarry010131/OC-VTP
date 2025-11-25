# -*- coding: utf-8 -*-
"""
OCL training with DINOSAUR on LMDB features, and step-wise LLaVA validation via LMMs-Eval.

Data in LMDB (value is pickled dict):
sample_dict = {
    "idx": idx,
    "image_id": ex["image"],
    "gt": gt,                   # ground-truth answer string (may be empty {})
    "feat_npz": buf_feat,       # bytes or None  -> object features (sequence tokens x dim)
    "feat_shape": feat_shape,   # list or None   -> reshape hint
    "tok_npz": buf_tok,         # bytes          -> text tokens only (no attn_mask), batch=1
}

Key points:
- Remove CLS token before OCL training; always add it back at the end and keep it.
- Validation uses single-case generation (no batch), consuming tok_npz only.
- Vision tokens are compressed by slot-attention argmax; deduplicate and pad to target = num_slots.
- Each n steps: write predictions.jsonl / references.jsonl / questions.jsonl and evaluate with LMMs-Eval.
"""

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
)
from utils import set_seed, AreaWeightedMSELoss
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token


# ------------- Dataset from LMDB -------------
class LMDBReader(Dataset):
    def __init__(self, lmdb_path: str, max_spare: Optional[int] = 16, max_samples: Optional[int] = None):
        self.lmdb_path = lmdb_path
        self.max_spare = max_spare
        self.env = None
        env_ = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            readahead=False,
            meminit=False,
            max_spare_txns=max_spare,
            lock=False,
        )

        # Collect keys (assume keys are stringified indices or similar)
        with env_.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))
        env_.close()

        # Limits
        if max_samples is not None and max_samples > 0:
            self.idxs = self.idxs[:max_samples]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self.env_open()
        key = self.idxs[idx]
        with self.env.begin(buffers=True) as txn:
            sample_ = txn.get(key)
        sample = pkl.loads(bytes(sample_))

        feat = sample["feats"]

        # If NeXT
        if len(feat.shape) == 3:
            cls_token = np.mean(feat[:, :1, :], axis=0)
            feat = feat[:, 1:, :].reshape(-1, 1024)
            feat = np.concatenate([cls_token, feat], axis=0)

        feat = np.array(feat, dtype=np.float32, copy=True, order="C")
        feat = torch.tensor(feat, dtype=torch.float32).contiguous()

        # decode
        return {
            "idx": sample["idx"],
            "feat": feat,
        }

    def env_open(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=False,
                readonly=True,
                readahead=True,
                meminit=False,
                max_spare_txns=self.max_spare,
                lock=False,
            )


class FeatsDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        feat = self.base[idx]["feat"]
        if feat.shape[0] == 1:
            feat = self.base[idx]["feat"][0]
        return feat.to(dtype=torch.float32)


# ------------- OCL wrapper -------------
"""
1. Input features
2. Encode positional embeddings
3. Encode projection
4. Initialize slots
5. Slot attention
6. Decode and reconstruct features
"""


class OCLTrainer:
    def __init__(
            self,
            args,
            enc_in_dim: int = 1024,
            enc_dims: list = [1024 * 2, 512],
            enc_ln: str = 'pre',
            enc_dropout: float = 0.0,
            init_num: int = 64,
            init_dim: int = 512,
            agg_num_iter: int = 3,
            agg_embed_dim: int = 512,
            agg_ffn_dim: int = 512 * 4,
            agg_dropout: float = 0.01,
            agg_trunc_bp: str = 'bi-level',
            pos_resolut: list = [24 * 24],
            pos_embed_dim: int = 1024,
            proj1_in_dim: int = 1024,
            proj1_out_dim: int = 1024,
            proj1_norm_dim: int = 1024,
            proj1_bias: bool = False,
            proj2_in_dim: int = 512,
            proj2_out_dim: int = 1024,
            proj2_norm_dim: int = 1024,
            proj2_bias: bool = False,
            transformer_d_model: int = 1024,
            transformer_nhead: int = 4,
            transformer_feedforward: int = 1024 * 4,
            transformer_dropout: float = 0.0,
            transformer_activation: str = "gelu",
            transformer_batch_first: bool = True,
            transformer_norm_first: bool = True,
            transformer_bias: bool = False,
            backbone_nlayer: int = 4,
            dec_dim: int = 1024,
            # dec_readout: nn.Module = Identity(),
            lr: float = 1e-3,
            device: torch.device = torch.device('cpu'),
    ):
        """
        Trainer, contains the model, the loss function, and the optimizer.

        :param enc_in_dim: `Encoder` param, refers to input dimension.
        :param enc_dims: `Encoder` param, refers to hidden and output dimensions, which should be a list.
        :param enc_ln: `Encoder` param, refers to the position of layer norm, only available between 'pre' and 'post'.
        :param enc_dropout: `Encoder` param, refers to the dropout probability.
        :param init_num: `Initialize` param, refers to the number of slots.
        :param init_dim: `Initialize` param, refers to the projection dimension of slots.
        :param agg_num_iter: `SlotAttention` param, refers to the number of loops of the slot attention.
        :param agg_embed_dim: `SlotAttention` param, refers to the embedding dimension of q, k, v projection.
        :param agg_ffn_dim: `SlotAttention` param, refers to the MLP hidden dimension.
        :param agg_dropout: `SlotAttention` param, refers to the dropout probability.
        :param agg_trunc_bp: `SlotAttention` param, refers to detach from initial slot.
        :param pos_resolut: `PositionalEmbedding` param, refers to positional embeddings shape.
        :param pos_embed_dim: `PositionalEmbedding` param, refers to positional embeddings dimension.
        :param proj1_in_dim: 'Decoder Projection1' param, refers to input dimension.
        :param proj1_out_dim: 'Decoder Projection1' param, refers to output dimension.
        :param proj1_norm_dim: 'Decoder Projection1' param, refers to normalization dimension.
        :param proj1_bias: 'Decoder Projection1' param, refers to if using bias, default False.
        :param proj2_in_dim: 'Decoder Projection2' param, refers to input dimension.
        :param proj2_out_dim: 'Decoder Projection2' param, refers to output dimension.
        :param proj2_norm_dim: 'Decoder Projection2' param, refers to normalization dimension.
        :param proj2_bias: 'Decoder Projection2' param, refers to if using bias, default False.
        :param transformer_d_model: 'Backbone Layer' param, refers to transformer layer dimension.
        :param transformer_nhead: 'Backbone Layer' param, refers to transformer layer num of heads.
        :param transformer_feedforward: 'Backbone Layer' param, refers to transformer feedforward dimension.
        :param transformer_dropout: 'Backbone Layer' param, refers to transformer dropout.
        :param transformer_activation: 'Backbone Layer' param, refers to transformer activation, default 'gelu'.
        :param transformer_batch_first: 'Backbone Layer' param, refers to transformer batch first, default True.
        :param transformer_norm_first: 'Backbone Layer' param, refers to transformer normLayer first, default True.
        :param transformer_bias: 'Backbone Layer' param, refers to if using bias, default False.
        :param backbone_nlayer: 'Backbone' param, refers to backbone number of transformer layers.
        :param dec_dim: 'Decoder' param, refers to decoder dimension.
        :param lr: Learning rate
        """
        self.device = device
        self.K_slots = init_num

        # ------------- Model Initialization -------------
        encode_posit_embed = Identity()
        encode_project = MLP(in_dim=enc_in_dim, dims=enc_dims, ln=enc_ln, dropout=enc_dropout)
        # initializ = NormalSeparat(num=init_num, dim=init_dim)
        initializ = NormalShared(dim=init_dim)
        aggregat = SlotAttention(num_iter=agg_num_iter, embed_dim=agg_embed_dim, ffn_dim=agg_ffn_dim,
                                 dropout=agg_dropout, trunc_bp=agg_trunc_bp)
        # Decoder
        posit_embed = LearntPositionalEmbedding(resolut=pos_resolut, embed_dim=pos_embed_dim)
        proj1 = Project(in_features=proj1_in_dim, out_features=proj1_out_dim, normalized_shape=proj1_norm_dim,
                        bias=proj1_bias)
        proj2 = Project(in_features=proj2_in_dim, out_features=proj2_out_dim, normalized_shape=proj2_norm_dim,
                        bias=proj2_bias)
        decoder_layer = TransformerDecoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation,
            batch_first=transformer_batch_first,
            norm_first=transformer_norm_first,
            bias=transformer_bias,
        )
        backbone = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=backbone_nlayer,
        )
        readout = Identity()

        decode = ARRandTransformerDecoder(
            vfm_dim=dec_dim,
            posit_embed=posit_embed,
            project1=proj1,
            project2=proj2,
            backbone=backbone,
            readout=readout,
        )
        self.model = DINOSAUR(
            encode_posit_embed=encode_posit_embed,
            encode_project=encode_project,
            initializ=initializ,
            aggregat=aggregat,
            decode=decode,
        ).to(self.device)

        self.criterion = AreaWeightedMSELoss().to(self.device)
        # self.criterion = nn.MSELoss().to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineLRScheduler(
            optimizer=self.opt,
            t_initial=args.epochs,
            lr_min=args.lr_min,
            warmup_t=args.warmup_t,
            warmup_lr_init=args.warmup_lr_init,
        )
        self.scheduler_step = False
        print(f"Trainer Initialization Completed!")

    def step(self, feats: np.ndarray, epoch, has_cls: bool = True) -> Dict[str, torch.Tensor]:
        """
        Train on a single sample of object features (np: [N, D] or [1,N,D]).
        Returns dict with tensors for logging if needed.
        """
        assert feats is not None, "feat_npz is None; cannot train OCL."
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        x = feats.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, D]

        # Remove CLS before OCL
        if has_cls:
            x_main, _ = OCLTrainer.remove_and_remember_cls(x)
        else:
            x_main = x

        self.model.train()

        self.opt.zero_grad()

        # Forward
        slots, attn_qk, decoder_alpha, x_rec = self.model(x_main, self.K_slots)  # Shapes: [B, S, T], [B, K, T], [B, K, T], [B, T, C]

        # Compute loss
        loss = self.criterion(
            pred=x_rec,
            target=x_main.detach(),
            alpha=attn_qk
        )
        # loss = self.criterion(x_rec, x_main.detach())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)

        self.opt.step()

        if self.scheduler_step:
            self.scheduler.step(epoch)
            self.scheduler_step = False

        return {
            "loss": loss.detach().cpu().item(),
            "lr": self.opt.param_groups[0]["lr"],
        }

    def validate(self, feats: np.ndarray, has_cls: bool = True) -> Dict[str, torch.Tensor]:
        assert feats is not None, "feat_npz is None; cannot train OCL."
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        x = feats.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, D]

        # Remove CLS before OCL
        if has_cls:
            x_main, _ = OCLTrainer.remove_and_remember_cls(x)
        else:
            x_main = x

        self.model.eval()

        with torch.no_grad():
            # Forward
            slots, attn_qk, decoder_alpha, x_rec = self.model(
                x_main, self.K_slots)  # Shapes: [B, S, T], [B, K, T], [B, K, T], [B, T, C]

            # Compute loss
            loss = self.criterion(
                pred=x_rec,
                target=x_main.detach(),
                alpha=attn_qk
            )
            # loss = self.criterion(x_rec, x_main.detach())

        return {
            "loss": loss.detach().cpu().item(),
        }

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        return self.model.load_state_dict(sd)

    def save_model(self, save_dir):
        torch.save(self.state_dict(), save_dir)

    @staticmethod
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

    @staticmethod
    def add_back_cls(compressed_feats: torch.Tensor, cls_tok: torch.Tensor) -> torch.Tensor:
        """
        compressed_feats: [1, M, D], cls_tok: [1,1,D] -> concat to [1, M+1, D] with CLS at front.
        """
        return torch.cat([cls_tok, compressed_feats], dim=1)


# ------------- Main training + validation loop -------------
def main():
    parser = argparse.ArgumentParser()
    # ------------- Basis -------------
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_folder", type=str, required=True, help="output folder.")
    # ------------- Data -------------
    parser.add_argument("--lmdb_path", type=str, required=True, help="Lmdb path.")
    parser.add_argument("--max_spare", type=int, default=16, help="Lmdb max spare.")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of samples to load.")
    # ------------- OCL Trainer -------------
    # Encoder
    parser.add_argument("--enc_in_dim", type=int, default=1024, help="`OCL encoder`, input dimension.")
    parser.add_argument("--enc_dims", type=list, default=[1024 * 2, 512], help="`OCL encoder`, hidden and out layers.")
    parser.add_argument("--enc_ln", type=str, default="pre", choices=["pre", "post"],
                        help="`OCL encoder`, position of the layer norm, can be only `pre` or `post`.")
    parser.add_argument("--enc_dropout", type=float, default=0.0, help="`OCL encoder`, dropout probability.")
    # Slots initialization
    parser.add_argument("--init_num", type=int, default=64,
                        help="`OCL slots initialization`, should be target tokens num - 1 if existing a cls token.")
    parser.add_argument("--init_dim", type=int, default=512,
                        help="`OCL slots initialization`, the hidden dimension of slots.")
    # Slot attention
    parser.add_argument("--agg_num_iter", type=int, default=3,
                        help="`OCL slot attention`, the number of loops for slot attention.")
    parser.add_argument("--agg_embed_dim", type=int, default=512,
                        help="`OCL slot attention`, projection embedding dimension.")
    parser.add_argument("--agg_ffn_dim", type=int, default=512 * 4,
                        help="`OCL slot attention`, FFN dimension after each attention.")
    parser.add_argument("--agg_dropout", type=float, default=0.01, help="`OCL slot attention`, dropout probability.")
    parser.add_argument("--agg_trunc_bp", type=str, default="bi-level",
                        help="`OCL slot attention`, if the gradient back-propagate to the variance `sigma` at the last attention loop.")
    # Decoder
    parser.add_argument("--pos_resolut", type=list, default=[24 * 24],
                        help="`OCL positional embedding`, the number of tokens to be embedded.")
    parser.add_argument("--pos_embed_dim", type=int, default=1024,
                        help="`OCL embedding dimension`, embedding dimension.")
    parser.add_argument("--proj1_in_dim", type=int, default=1024,
                        help="`OCL decoder projector`, the input dimension.")
    parser.add_argument("--proj1_out_dim", type=int, default=1024,
                        help="`OCL decoder projector`, the output dimension.")
    parser.add_argument("--proj1_norm_dim", type=int, default=1024,
                        help="`OCL decoder projector`, the normalization dimension.")
    parser.add_argument("--proj1_bias", action="store_true", help="`OCL decoder projector`, the bias parameter.")
    parser.add_argument("--proj2_in_dim", type=int, default=512,
                        help="`OCL decoder projector`, the input dimension.")
    parser.add_argument("--proj2_out_dim", type=int, default=1024,
                        help="`OCL decoder projector`, the output dimension.")
    parser.add_argument("--proj2_norm_dim", type=int, default=1024,
                        help="`OCL decoder projector`, the normalization dimension.")
    parser.add_argument("--proj2_bias", action="store_true", help="`OCL decoder projector`, the bias parameter.")
    parser.add_argument("--transformer_d_model", type=int, default=1024,
                        help="`Decoder backbone transformer layer`, the input dimension.")
    parser.add_argument("--transformer_nhead", type=int, default=4,
                        help="`Decoder backbone transformer layer`, the number of attention heads.")
    parser.add_argument("--transformer_feedforward", type=int, default=1024 * 4,
                        help="`OCL decoder transformer layer`, the hidden dimension.")
    parser.add_argument("--transformer_dropout", type=float, default=0.0,
                        help="`OCL decoder transformer layer`, dropout probability.")
    parser.add_argument("--transformer_activation", type=str, default="gelu",
                        help="`OCL decoder transformer layer`, the activation function.")
    parser.add_argument("--transformer_batch_first", action="store_true",
                        help="`OCL decoder transformer layer`, batch first.")
    parser.add_argument("--transformer_norm_first", action="store_true",
                        help="`OCL decoder transformer layer`, normalization first.")
    parser.add_argument("--transformer_bias", action="store_true",
                        help="`OCL decoder transformer layer`, bias parameter.")
    parser.add_argument("--backbone_nlayer", type=int, default=4,
                        help="`OCL backbone`, the number of transformer layers.")
    parser.add_argument("--dec_dim", type=int, default=1024, help="`Decoder`, input dimensions of the decoder.")
    # ------------- Learner -------------
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--lr_min", type=float, default=1e-8, help="Minimum learning rate while Cosine annealing.")
    parser.add_argument("--warmup_t", type=int, default=4, help="Warmup periods.")
    parser.add_argument("--warmup_lr_init", type=float, default=1e-6, help="Warmup start learning rate.")
    parser.add_argument("--train_set_prob", type=float, default=0.95,
                        help="The proportion of the whole set will be training data.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--val_interval", type=int, default=1, help="Interval between validation epochs.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs without improvement before early stopping.")
    # ------------- Evaluation -------------
    # LLaVA predict generator
    parser.add_argument("--model_path", type=str, default=r"liuhaotian/llava-v1.5-7b", help="LLaVA model path.")
    parser.add_argument("--model_base", type=str, default=None, help="LLaVA model base.")
    parser.add_argument("--image_folder", type=str, required=True, help="Image base folder.")
    parser.add_argument("--process_layer_idx", type=int, default=8,
                        help="From which layer the object features will be processed.")
    parser.add_argument("--target_num", type=int, default=64, help="Target number of tokens.")
    parser.add_argument("--top_k", type=int, default=1,
                        help="The number of top k tokens will be saved as compressed tokens (cls tokens counted).")
    parser.add_argument("--has_cls", action="store_true", help="If the VLM output has any cls tokens.")
    parser.add_argument("--pad_mode", type=str, default="att_score", choices=["att_score", "random"],
                        help="Choose how the tokens will be padded to target num, if there are not enough tokens selected by OCL (Output same indices exist).")
    parser.add_argument("--att_selection_mode", type=str, default="meanq", choices=["meanq", "cls"],
                        help="Choose how the attention score will be computed, `cls` refers to only consider the att scores from cls tokens.")
    parser.add_argument("--num_beams", type=int, default=1, help="The number of beams.")
    parser.add_argument("--temperature", type=float, default=0.2, help="The temperature.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="The maximum number of tokens.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    if args.process_layer_idx == -2 and args.pad_mode == "att_score":
        args.pad_mode = "random"

    # Dataset
    lmdb_set = LMDBReader(args.lmdb_path, max_samples=args.max_samples)
    train_len = int(args.train_set_prob * len(lmdb_set))
    val_len = len(lmdb_set) - train_len
    lmdb_set_tr, lmdb_set_val = random_split(lmdb_set, [train_len, val_len],
                                             generator=torch.Generator().manual_seed(args.seed))
    # Training set
    feature_set = FeatsDataset(lmdb_set_tr)
    train_loader = DataLoader(feature_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    feature_set_val = FeatsDataset(lmdb_set_val)
    val_loader = DataLoader(feature_set_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    print(f"[info] LMDB loaded: {train_len} training samples, {val_len} validation samples.")

    # OCL trainer
    trainer = OCLTrainer(
        args=args,
        enc_in_dim=args.enc_in_dim,
        enc_dims=args.enc_dims,
        enc_ln=args.enc_ln,
        enc_dropout=args.enc_dropout,
        init_num=args.init_num,
        init_dim=args.init_dim,
        agg_num_iter=args.agg_num_iter,
        agg_embed_dim=args.agg_embed_dim,
        agg_ffn_dim=args.agg_ffn_dim,
        agg_dropout=args.agg_dropout,
        agg_trunc_bp=args.agg_trunc_bp,
        pos_resolut=args.pos_resolut,
        pos_embed_dim=args.pos_embed_dim,

        lr=args.lr,
        device=device,
    )

    # Training details
    epochs = args.epochs
    patience_counter = 0
    val_counter = 0
    # For saving best validated model weights
    val_best_loss = float(np.inf)
    best_epoch = -1  # From 0 ~ epochs
    # For generate training loss plots
    loss_tr = []
    # For generate validate loss plots
    loss_val = []
    for epoch in range(args.epochs):
        step = 0
        accumulated_loss = 0.0
        epoch_bar = tqdm(train_loader, desc=f"[info] Epoch {epoch}/{epochs}", unit="batch")
        for i, batch in enumerate(epoch_bar):
            trainer.K_slots = random.choice([16, 32, 64, 128])
            # trainer.K_slots = random.choice([32, 64, 128, 192])
            # trainer.K_slots = random.choice([640, 320, 160, 80])
            # =============================== training ===============================
            input_features = batch.to(device)
            output_dict = trainer.step(input_features, epoch=epoch, has_cls=args.has_cls)
            epoch_bar.set_postfix(loss=f"{output_dict['loss']:.8f}", lr=f"{output_dict['lr']:.8f}")
            accumulated_loss += output_dict['loss']
            step += 1
        loss_tr.append(accumulated_loss / step)
        trainer.scheduler_step = True

        # if epoch > epochs // 2:
        if epoch >= 0:
            if epoch % args.val_interval == 0:
                trainer.K_slots = args.init_num
                print(f"[val] epoch={epoch} ...")
                # =============================== Validating ===============================
                val_step = 0
                accu_val_loss = 0.0
                val_bar = tqdm(val_loader, desc=f"[val] epoch={epoch}", unit="data")
                for i, data in enumerate(val_bar):
                    val_dict = trainer.validate(
                        input_features,
                        has_cls=args.has_cls
                    )
                    accu_val_loss += val_dict['loss']
                    val_step += 1

                val_avg_loss = accu_val_loss / val_step
                loss_val.append(val_avg_loss)
                val_counter += 1
                print(f"[val] val_loss={val_avg_loss:.4f} / best_score={val_best_loss:.4f}")

                # Save best metric model
                if val_avg_loss <= val_best_loss:
                    val_best_loss = val_avg_loss
                    patience_counter = 0
                    trainer.save_model(os.path.join(output_folder, f"best_metric_weights.pth"))
                    print(f"Model saved!")

                patience_counter += 1
                if patience_counter >= args.patience:
                    trainer.save_model(os.path.join(output_folder, f"last_epoch_weights.pth"))
                    print(f"[info] last epoch model saved!")
                    break
                if epoch == args.epochs - 1:
                    trainer.save_model(os.path.join(output_folder, f"last_epoch_weights.pth"))
                    print(f"[info] last epoch model saved!")

    df = pd.DataFrame({"val_loss": loss_val})
    df.to_csv(os.path.join(output_folder, f"val_loss_history_{val_counter}.csv"), index=False, encoding="utf-8")

    df = pd.DataFrame({"loss": loss_tr})
    df.to_csv(os.path.join(output_folder, f"train_loss_history_{epochs}.csv"), index=False, encoding="utf-8")
    print(f"[info] done!")


if __name__ == "__main__":
    main()
