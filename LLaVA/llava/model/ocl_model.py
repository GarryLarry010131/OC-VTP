# import re
# import numpy as np
# import torch as pt
# import torch.nn as nn
# import torch.nn.functional as ptnf
# from einops import rearrange, repeat, einsum
# import timm
#
# Identity = nn.Identity
#
#
# def build_ocl_model(
#         enc_in_dim: int = 1024,
#         enc_dims: list = [1024 * 2, 512],
#         enc_ln: str = 'pre',
#         enc_dropout: float = 0.0,
#         init_num: int = 64,
#         init_dim: int = 512,
#         agg_num_iter: int = 3,
#         agg_embed_dim: int = 512,
#         agg_ffn_dim: int = 512 * 4,
#         agg_dropout: float = 0.01,
#         agg_trunc_bp: str = 'bi-level',
#         pos_resolut: list = [24 * 24],
#         pos_embed_dim: int = 1024,
#         proj1_in_dim: int = 1024,
#         proj1_out_dim: int = 1024,
#         proj1_norm_dim: int = 1024,
#         proj1_bias: bool = False,
#         proj2_in_dim: int = 512,
#         proj2_out_dim: int = 1024,
#         proj2_norm_dim: int = 1024,
#         proj2_bias: bool = False,
#         transformer_d_model: int = 1024,
#         transformer_nhead: int = 4,
#         transformer_feedforward: int = 1024 * 4,
#         transformer_dropout: float = 0.0,
#         transformer_activation: str = "gelu",
#         transformer_batch_first: bool = True,
#         transformer_norm_first: bool = True,
#         transformer_bias: bool = False,
#         backbone_nlayer: int = 4,
#         dec_dim: int = 1024,
# ):
#     encode_posit_embed = Identity()
#     encode_project = MLP(in_dim=enc_in_dim, dims=enc_dims, ln=enc_ln, dropout=enc_dropout)
#     # initializ = NormalSeparat(num=init_num, dim=init_dim)
#     initializ = NormalShared(dim=init_dim)
#     aggregat = SlotAttention(num_iter=agg_num_iter, embed_dim=agg_embed_dim, ffn_dim=agg_ffn_dim,
#                              dropout=agg_dropout, trunc_bp=agg_trunc_bp)
#     # Decoder
#     # posit_embed = LearntPositionalEmbedding(resolut=pos_resolut, embed_dim=pos_embed_dim)
#     # proj1 = Project(in_features=proj1_in_dim, out_features=proj1_out_dim, normalized_shape=proj1_norm_dim,
#     #                 bias=proj1_bias)
#     # proj2 = Project(in_features=proj2_in_dim, out_features=proj2_out_dim, normalized_shape=proj2_norm_dim,
#     #                 bias=proj2_bias)
#     # decoder_layer = TransformerDecoderLayer(
#     #     d_model=transformer_d_model,
#     #     nhead=transformer_nhead,
#     #     dim_feedforward=transformer_feedforward,
#     #     dropout=transformer_dropout,
#     #     activation=transformer_activation,
#     #     batch_first=transformer_batch_first,
#     #     norm_first=transformer_norm_first,
#     #     bias=transformer_bias,
#     # )
#     # backbone = TransformerDecoder(
#     #     decoder_layer=decoder_layer,
#     #     num_layers=backbone_nlayer,
#     # )
#     # readout = Identity()
#     #
#     # decode = ARRandTransformerDecoder(
#     #     vfm_dim=dec_dim,
#     #     posit_embed=posit_embed,
#     #     project1=proj1,
#     #     project2=proj2,
#     #     backbone=backbone,
#     #     readout=readout,
#     # )
#     model = DINOSAUR(
#         encode_posit_embed=encode_posit_embed,
#         encode_project=encode_project,
#         initializ=initializ,
#         aggregat=aggregat,
#         # decode=decode,
#         decode=None,
#     )
#     return model
#
#
# class Sequential(nn.Sequential):
#     def __init__(self, modules: list):
#         super().__init__(*modules)
#
#     def forward(self, input):
#         for module in self:
#             if isinstance(input, (list, tuple)):
#                 input = module(*input)
#             else:
#                 input = module(input)
#         return input
#
#
# class Interpolate(nn.Module):
#
#     def __init__(self, size=None, scale_factor=None, interp="bilinear"):
#         super().__init__()
#         self.size = size
#         self.scale_factor = scale_factor
#         self.interp = interp
#
#     def forward(self, input):
#         return ptnf.interpolate(input, self.size, self.scale_factor, self.interp)
#
#
# class DINO2ViT(nn.Module):
#     def __init__(self,
#                  model_name="vit_small_patch14_reg4_dinov2.lvd142m",
#                  in_size=224,
#                  rearrange=True,
#                  norm_out=True):
#         """Backbone DINOViTv2, the results can be used both as the teacher and the inputs of the slots attention."""
#         super().__init__()
#         self.in_size = in_size
#         self.rearrange = rearrange
#
#         # Create a DINOViTv2 model
#         self.DINOv2 = timm.create_model(model_name, pretrained=True, img_size=self.in_size)
#         if not norm_out:
#             self.DINOv2.norm = nn.Identity()
#
#         # Get necessary configs
#         self.patch_size = self.DINOv2.patch_embed.patch_size[0]
#         self.num_prefix_tokens = self.DINOv2.num_prefix_tokens
#         # Ensure successful patching
#         assert self.in_size % self.patch_size == 0
#         self.out_size = in_size // self.patch_size
#
#     def forward(self, input):
#         features = self.DINOv2.forward_features(input)
#         features = features[:, self.num_prefix_tokens:, :]
#         if self.rearrange:
#             features = rearrange(features, "b (h w) c -> b c h w", h=self.out_size, w=self.out_size)
#         return features
#
#
# class MLP(nn.Module):
#     def __init__(self, in_dim, dims, ln=None, dropout=0):
#         super().__init__()
#         """
#         DINOViT -> Slots Attention Input Projection, [
#
#         in_dim: input dimension=384
#         dims: channels of each layer [384*2, 256]
#         ln should be selected from [None, 'pre', 'post']
#         """
#         assert ln in [None, 'pre', 'post']
#
#         ch_in = in_dim
#         layers = []
#
#         if ln == 'pre':
#             layers.append(nn.LayerNorm(ch_in))
#
#         depth = len(dims)
#         for i, ch in enumerate(dims):
#             if i + 1 < depth:
#                 layers.append(nn.Linear(ch_in, ch))
#                 layers.append(nn.GELU())
#                 layers.append(nn.Dropout(dropout))
#             else:
#                 layers.append(nn.Linear(ch_in, ch))
#             ch_in = ch
#
#         if ln == 'post':
#             layers.append(nn.LayerNorm(ch_in))
#
#         self.mlp = nn.Sequential(*layers)
#
#     def forward(self, input):
#         return self.mlp(input)
#
#
# class NormalSeparat(nn.Module):
#     def __init__(self, num, dim):
#         """Slots initialization"""
#         super().__init__()
#         self.num = num
#         self.dim = dim
#         self.mean = nn.Parameter(pt.empty(1, self.num, self.dim))
#         # self.std = nn.Parameter(pt.ones(1, self.num, self.dim))
#         self.logstd = nn.Parameter((pt.ones(1, self.num, self.dim) * dim ** -0.5).log())
#         # nn.init.xavier_uniform_(self.mean)
#         nn.init.xavier_uniform_(self.mean[0, :, :])
#
#     def forward(self, input):
#         slots = pt.zeros(input.shape[0], self.num, self.dim).to(input.device)
#         slots = slots + self.mean
#         # slots = slots + pt.randn_like(slots) * self.std
#         if self.training:
#             slots = slots + pt.randn_like(slots) * self.logstd.exp()
#
#         return slots
#
#
# class NormalShared(nn.Module):
#     """Shared gaussian as queries."""
#
#     # TODO new trick: Conditional Random Initialization
#
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.mean = nn.Parameter(pt.empty(1, 1, dim))
#         self.logstd = nn.Parameter(pt.empty(1, 1, dim))
#         nn.init.xavier_uniform_(self.mean)
#         nn.init.xavier_uniform_(self.logstd)
#
#     def forward(self, b, num):
#         smpl = self.mean.expand(b, num, -1)
#         randn = pt.randn_like(smpl)
#         smpl = smpl + randn * self.logstd.exp()
#         return smpl
#
#
# class SlotAttention(nn.Module):
#     def __init__(self, num_iter, embed_dim, ffn_dim, dropout=0., trunc_bp=None):
#         super().__init__()
#         self.num_iter = num_iter
#         self.embed_dim = embed_dim
#         self.norm_in = nn.LayerNorm(embed_dim)
#         self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.norm_slots_0 = nn.LayerNorm(embed_dim)
#         self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.gru = nn.GRUCell(embed_dim, embed_dim)
#         self.norm_slots_1 = nn.LayerNorm(embed_dim)
#         self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], ln=None, dropout=dropout)
#
#         self.trunc_bp = trunc_bp
#
#     def forward(self, input, slots, smask=None):
#         B, K, C = slots.shape
#         input = self.norm_in(input)
#         k = self.proj_k(input)
#         v = self.proj_v(input)
#
#         ########
#         slots_ = slots
#         ########
#         attn = ...
#         for i in range(self.num_iter):
#             ########
#             if i + 1 == self.num_iter:
#                 if self.trunc_bp == "bi-level":
#                     slots_ = slots_.detach() + slots - slots.detach()
#             ########
#             slots_prev = slots_
#             slots_ = self.norm_slots_0(slots_)
#             q = self.proj_q(slots_)
#
#             product = pt.einsum("bqc, bkc -> bqk", q, k)
#             attn = product * (q.shape[-1] ** -0.5)
#             ########
#             if smask is not None:
#                 attn = attn.where(smask[:, :, None], -pt.inf)
#             ########
#             attn = pt.softmax(attn, dim=1)  # [b, q, k]: [b, K, h*w]
#
#             updates = attn / (attn.sum(dim=-1, keepdim=True) + 1e-5)
#             updates = pt.einsum("bqv, bvc -> bqc", updates, v)
#
#             slots_ = self.gru(updates.flatten(0, 1), slots_prev.flatten(0, 1))
#             slots_ = slots_.view(B, K, -1)
#
#             slots_ = slots_ + self.ffn(self.norm_slots_1(slots_))
#
#         return slots_, attn
#
#
# # class LearntPositionalEmbedding(nn.Module):
# #     def __init__(self, resolut, embed_dim):
# #         super().__init__()
# #         self.resolut = resolut
# #         self.embed_dim = embed_dim
# #         # self.pos_embed = nn.Parameter(pt.randn(1, *resolut, embed_dim), requires_grad=True)
# #         self.pos_embed = nn.Parameter(pt.zeros(1, *resolut, embed_dim), requires_grad=True)
# #         nn.init.trunc_normal_(self.pos_embed)
# #
# #     def forward(self, input, retp=False):
# #         # return x + self.pos_embed
# #         ########
# #         max_r = ", ".join([f":{_}" for _ in input.shape[1:-1]])
# #         # TODO XXX support variant length
# #         # pe = timm.layers.pos_embed.resample_abs_pos_embed(self.pe, ...)
# #         # pe = self.pe[:, :max_resolut, :]
# #         pe = eval(f"self.pos_embed[:, {max_r}, :]")
# #         output = input + pe
# #         if retp:
# #             return output, pe
# #         return output
# #         ########
#
#
# class LearntPositionalEmbedding(nn.Module):
#     """Support any dimension. Must be channel-last.
#     PositionalEncoding: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#     """
#
#     def __init__(self, resolut: list, embed_dim: int, in_dim: int = 0):
#         super().__init__()
#         self.resolut = resolut
#         self.embed_dim = embed_dim
#         if in_dim:
#             self._pe = nn.Parameter(pt.zeros(1, *resolut, in_dim), requires_grad=True)
#             self._project = nn.Linear(in_dim, embed_dim)
#         else:
#             self._pe = nn.Parameter(
#                 pt.zeros(1, *resolut, embed_dim), requires_grad=True
#             )
#         nn.init.trunc_normal_(self._pe)
#
#     @property
#     def pe(self):
#         if hasattr(self, "_project"):
#             return self._project(self._pe)
#         return self._pe
#
#     def forward(self, input, retp=False):
#         """
#         input: in shape (b,*r,c)
#         output: in shape (b,*r,c)
#         """
#         max_r = ", ".join([f":{_}" for _ in input.shape[1:-1]])
#         # TODO XXX support variant length
#         # pe = timm.layers.pos_embed.resample_abs_pos_embed(self.pe, ...)
#         # pe = self.pe[:, :max_resolut, :]
#         pe = eval(f"self.pe[:, {max_r}, :]")
#         output = input + pe
#         if retp:
#             return output, pe
#         return output
#
#     def extra_repr(self):
#         return f"{self.resolut}, {self.embed_dim}"
#         ########
#
#
# class Project(nn.Module):
#     def __init__(self, in_features, out_features, normalized_shape, bias=False):
#         super().__init__()
#         self.project = nn.Linear(in_features, out_features, bias=bias)
#         self.norm = nn.LayerNorm(normalized_shape=normalized_shape)
#
#     def forward(self, input):
#         input = self.project(input)
#         input = self.norm(input)
#         return input
#
#
# TransformerDecoderLayer = nn.TransformerDecoderLayer
# TransformerDecoder = nn.TransformerDecoder
#
#
# class ARRandTransformerDecoder(nn.Module):
#     """GeneralZ's new OCL decoder.
#     Auto-regressive Transformer decoder with random token permutations.
#     """
#
#     def __init__(
#             self,
#             vfm_dim,
#             posit_embed,
#             # posit_embed_hw,
#             project1,
#             project2,
#             backbone,
#             readout,
#     ):
#         super().__init__()
#         self.mask_token = nn.Parameter(pt.randn(1, 1, vfm_dim) * vfm_dim ** -0.5)
#         assert hasattr(posit_embed, "pe")
#         self.posit_embed = posit_embed  # 1d
#         # self.posit_embed_hw = posit_embed_hw  # 2d
#         self.project1 = project1
#         self.project2 = project2
#
#         assert isinstance(backbone, nn.TransformerDecoder)
#         self.norm0 = backbone.layers[0].norm1  # very beneficial
#         backbone.layers[0].norm1 = nn.Identity()  # very beneficial
#         self.backbone = backbone
#         self.readout = readout
#
#         def attent_hook_forward_pre(module, args, kwargs):
#             kwargs["need_weights"] = True  # obtain the attention weights
#
#         def attent_hook_forward(module, args, output):
#             self._attent = output[1]
#
#         self.backbone.layers[-1].multihead_attn.register_forward_pre_hook(
#             attent_hook_forward_pre, with_kwargs=True
#         )
#         self.backbone.layers[-1].multihead_attn.register_forward_hook(
#             attent_hook_forward
#         )
#
#         ### interaction asymmetry
#
#         self._interact = [None for _ in range(len(self.backbone.layers[:-1]))]
#         for l, layer in enumerate(self.backbone.layers[:-1]):
#             def interact_hook_forward(module, args, output):
#                 self._interact[l] = output[1]
#
#             layer.multihead_attn.register_forward_pre_hook(
#                 attent_hook_forward_pre, with_kwargs=True
#             )
#             layer.multihead_attn.register_forward_hook(interact_hook_forward)
#
#     def forward(self, input, slots, p=0.5):
#         """
#         input: target to be destructed, shape=(b,m=h*w,c)
#         slots: slots, shape=(b,n,c)
#         """
#         b, m, c = input.shape
#         assert m == self.posit_embed.pe.size(1)
#         _, n, _ = slots.shape
#         device = input.device
#         tokens = self.project1(input)  # (b,m,c)
#
#         # TODO XXX disable masking in val for attent2 !!!
#
#         # mim-predict-all-masked-tokens
#         # seg1:
#         # "ari": 0.20348355174064636, "ari_fg": 0.34435588121414185, "mbo": 0.29168349504470825, "miou": 0.2779198884963989
#         # seg2:  # TODO disable masking in val for attent2 !!!
#         # 'ari': 0.2038770616054535, 'ari_fg': 0.3444632291793823, 'mbo': 0.29167482256889343, 'miou': 0.27789679169654846
#
#         if self.training:
#             idxs = pt.vmap(  # (b,m)
#                 lambda _: pt.randperm(m, device=device), randomness="different"
#             )(tokens)
#             idxs_expanded = idxs[:, :, None].expand(-1, -1, c)
#
#             idxs0 = pt.arange(0, m, device=device)[None, :]  # (1,m)
#             keep1 = pt.randint(0, m - 1, [b, 1], device=device)  # (b,1)
#             keep2 = (
#                     pt.ones(b, 1, dtype=pt.long, device=device) * int(256 * 0.1) - 1
#             )  # TODO
#             # TODO XXX realize a Poisson: when in [0, 1], it is Poisson; when out, then uniformly re-distribute in [0, 1]
#             cond = pt.rand(b, 1, device=device) < p
#             keep = pt.where(cond, keep1, keep2)
#             # XXX 论文论述 XXX
#             # keep@0: SlotMixerDecoder
#             # 只预测下一个：ARTransformerDecoder
#             # 其中9种只预测下一个，AR9TransformerDecoder
#             mask = idxs0 < keep  # (b,m)
#
#             # shuffle tokens
#             tokens_shuffled = tokens.gather(1, idxs_expanded)  # (b,m,c)
#             # mask tokens
#             mask_token_expanded = self.mask_token.expand(b, m, -1)
#             tokens_masked = tokens_shuffled.where(mask[:, :, None], mask_token_expanded)
#
#             # shuffle pe
#             pe_expanded = self.posit_embed.pe[:, :m, :].expand(b, -1, -1)  # (b,m,c)
#             # pe_hw_expanded = self.posit_embed_hw.pe.flatten(1, -2)[:, :m, :].expand(
#             #     b, -1, -1
#             # )  # (b,m,c)
#             pe_shuffled = pe_expanded.gather(1, idxs_expanded)  # (b,m,c)
#             # pe_hw_shuffled = pe_hw_expanded.gather(1, idxs_expanded)  # (b,m,c)
#
#             query = tokens_masked + pe_shuffled  # + pe_hw_shuffled
#
#         else:
#             query = (
#                     tokens
#                     + self.posit_embed.pe[:, :m, :]
#                 # + self.posit_embed_hw.pe.flatten(1, -2)[:, :m, :]
#             )
#
#         memory = self.project2(slots)
#         autoreg = self.backbone(self.norm0(query), memory=memory)
#         recon = self.readout(autoreg)  # (b,m,c)
#         _, _, d = recon.shape
#
#         if self.training:
#             idxs_inverse = idxs.argsort(1)[:, :, None]
#             recon = recon.gather(1, idxs_inverse.expand(-1, -1, d))
#             attent = self._attent.gather(1, idxs_inverse.expand(-1, -1, n))
#         else:
#             attent = self._attent
#
#         attent = rearrange(attent, "b m n -> b n m")
#         return recon, attent
#
#
# class BroadcastMLPDecoder(nn.Module):
#     def __init__(self, posit_embed, backbone):
#         super().__init__()
#         self.posit_embed = posit_embed
#         self.backbone = backbone
#
#     def forward(self, slots, n_tokens):
#         B, K, C = slots.shape
#
#         # Apply slots on each patch
#         broad_slots = slots.reshape(-1, 1, C)
#         broad_slots = broad_slots.repeat(1, n_tokens, 1)
#
#         embeded_slots = self.posit_embed(broad_slots)
#
#         recon = self.backbone(embeded_slots)  # [B*K, n, 384 + 1]
#         recon, alpha = recon[:, :, :-1], recon[:, :, -1:]  # [B*K, n, 384], [B*K, n, 1]
#
#         recon = recon.reshape(B, K, n_tokens, -1)
#         alpha = alpha.reshape(B, K, n_tokens, 1)
#         alpha = alpha.softmax(dim=1)
#
#         recon = (recon * alpha).sum(dim=1)
#
#         return recon, alpha[:, :, :, 0]
#
#
# # class DINOSAUR(nn.Module):
# #     def __init__(self, encode_posit_embed, encode_project, initializ, aggregat, decode):
# #         super().__init__()
# #         # self.encode_backbone = encode_backbone
# #         self.encode_posit_embed = encode_posit_embed
# #         self.encode_project = encode_project
# #         self.initializ = initializ
# #         self.aggregat = aggregat
# #         self.decode = decode
# #
# #         ################
# #         self.reset_parameters(
# #             [self.encode_posit_embed, self.encode_project, self.aggregat]
# #         )
# #
# #     @staticmethod
# #     def reset_parameters(modules):
# #         for module in modules:
# #             for m in module.modules():
# #                 if isinstance(m, nn.Conv2d):
# #                     if m.bias is not None:
# #                         nn.init.zeros_(m.bias)
# #                 elif isinstance(m, nn.Linear):
# #                     if m.bias is not None:
# #                         nn.init.zeros_(m.bias)
# #                 elif isinstance(m, nn.GRUCell):
# #                     if m.bias:
# #                         nn.init.zeros_(m.bias_ih)
# #                         nn.init.zeros_(m.bias_hh)
# #
# #     ################
# #
# #     def forward(self, features):
# #         # features = self.encode_backbone(input).detach()  # Freeze backbone, gradients not pass
# #         b, t, c = features.shape  # [b, tokens, channels]
# #
# #         encode = self.encode_project(features)  # [b, tokens, proj_dims]: [b, 576, 512]
# #
# #         slots_init = self.initializ(encode)
# #
# #         slots, attn = self.aggregat(encode, slots_init)  # slots: [b, K, h_dim=512] attn: [b, K, tokens]
# #         recon, alpha = self.decode(slots, t)  # recon: [b, tokens, c] alpha: [b, K, tokens]
# #
# #         return slots, attn, alpha, recon
#
#
# class DINOSAUR(nn.Module):
#     def __init__(self, encode_posit_embed, encode_project, initializ, aggregat, decode):
#         super().__init__()
#         # self.encode_backbone = encode_backbone
#         self.encode_posit_embed = encode_posit_embed
#         self.encode_project = encode_project
#         self.initializ = initializ
#         self.aggregat = aggregat
#         self.decode = decode
#
#         ################
#         self.reset_parameters(
#             [self.encode_posit_embed, self.encode_project, self.aggregat]
#         )
#
#     @staticmethod
#     def reset_parameters(modules):
#         for module in modules:
#             for m in module.modules():
#                 if isinstance(m, nn.Conv2d):
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, nn.Linear):
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, nn.GRUCell):
#                     if m.bias:
#                         nn.init.zeros_(m.bias_ih)
#                         nn.init.zeros_(m.bias_hh)
#
#     ################
#
#     def forward(self, features, n_slots):
#         # features = self.encode_backbone(input).detach()  # Freeze backbone, gradients not pass
#         b, t, c = features.shape  # [b, tokens, channels]
#
#         encode = self.encode_project(features)  # [b, tokens, proj_dims]: [b, 576, 512]
#
#         slots_init = self.initializ(encode.shape[0], n_slots)
#
#         slots, attn = self.aggregat(encode, slots_init)  # slots: [b, K, h_dim=512] attn: [b, K, tokens]
#
#         if self.decode is not None:
#             recon, alpha = self.decode(features, slots)  # recon: [b, tokens, c] alpha: [b, K, tokens]
#             return slots, attn, alpha, recon
#         else:
#             return slots, attn
#
#
# import torch
# import random
# from typing import Tuple, List, Optional
# import numpy as np
# import torch.nn as nn
#
#
# class OC_LLaVA(nn.Module):
#     def __init__(
#             self,
#             ocl_model,
#             llava_model,
#             mode="pruning",
#             process_layer_idx=8,
#             target_num: int = 64,
#             top_k: int = 1,
#             has_cls: bool = True,
#             pad_mode: str = "att_score",
#             att_selection_mode: str = "meanq",
#     ):
#         super().__init__()
#         # ------------- Model Initialization -------------
#         self.ocl = ocl_model
#         self.llava = llava_model
#
#         assert mode in ["pruning", "merging"], f"Unsupported mode {mode}!"
#         # Configs for hook
#         self.mode = mode
#         self.process_layer_idx = process_layer_idx
#
#         # Merging should happen on the last second layer
#         # if self.mode == "merging":
#         #     self.process_layer_idx = -2
#
#         self.has_cls = has_cls
#
#         if self.mode == "pruning" and self.has_cls:
#             self.target_num = target_num + 1
#         else:
#             self.target_num = target_num
#         self.top_k = top_k
#
#         # While last second layer can only use random as padding mode
#         if process_layer_idx == -2:
#             pad_mode = "random"
#         self.pad_mode = pad_mode
#         self.att_selection_mode = att_selection_mode
#
#         # Define a hook and register
#         self.vision = self.llava.get_vision_tower().vision_tower
#         # vision.output_attentions = True
#         if hasattr(self.vision, "vision_model"):
#             self.vision.vision_model.config.output_attentions = True
#
#         self.target_layer = self.vision.vision_model.encoder.layers
#         self.last_layer = self.target_layer[-2]
#
#         # Get features
#         self.state = {
#             "compressed_ids": None,
#             "busy": False,
#             "handle_get_features": None,
#
#             "handle_get_attnScores": None,
#             "handle_alternate_tokens": None,
#
#             "handle_get_merging_attn": None,
#             "merging_attn": None,
#             "handle_merging": None,
#         }
#
#         self.config = self.llava.config
#
#     def forward(self, *args, **kwargs):
#         # self.state["compressed_ids"] = None
#         # self.state["busy"] = False
#         #
#         # self.state["handle_get_features"] = self.target_layer[self.process_layer_idx].register_forward_hook(
#         #     self.hook_gf)
#         #
#         # if self.pad_mode == "att_score":
#         #     self.state["handle_get_attnScores"] = getattr(self.last_layer, "self_attn",
#         #                                                   getattr(self.last_layer, "attn", None)).register_forward_hook(
#         #         self.hook_att)
#         #
#         # # Select indices and alternate the final tokens
#         # self.state["handle_alternate_tokens"] = self.target_layer[-1].register_forward_hook(
#         #     self.hook_al)
#         #
#         # out = self.llava(*args, **kwargs)
#         #
#         # for key in ("handle_get_features", "handle_alternate_tokens"):
#         #     if self.state.get(key) is not None:
#         #         try:
#         #             self.state[key].remove()
#         #         except Exception:
#         #             raise ValueError(f"{key} has no attribute remove.")
#         #         self.state[key] = None
#         # if self.pad_mode == "att_score":
#         #     if self.state["handle_get_attnScores"] is not None:
#         #         self.state["handle_get_attnScores"].remove()
#         #         self.state["handle_get_attnScores"] = None
#         #
#         # self.state["compressed_ids"] = None
#         # self.state["busy"] = False
#         # return out
#         pass
#
#     def hook_gf(self, module, input, output):
#         if self.state["busy"] or self.state["compressed_ids"] is not None:
#             return
#
#         if isinstance(output, torch.Tensor):
#             input_features = output.detach()
#         elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
#             input_features = output[0].detach()
#         else:
#             raise TypeError("Unsupported object feature type!")
#
#         self.state["busy"] = True
#         if self.state["handle_get_features"] is not None:
#             self.state["handle_get_features"].remove()
#             self.state["handle_get_features"] = None
#         try:
#             with torch.inference_mode():
#                 # ======== LLaVA-NeXT ========
#                 if input_features.ndim == 3:
#                     if self.has_cls:
#                         cls_token = torch.mean(input_features[:, :1, :].detach(), dim=0)
#                         feat = input_features[:, 1:, :].detach().reshape(-1, 1024)
#                         input_features = torch.cat((cls_token, feat), dim=0)
#                     else:
#                         input_features = input_features.detach().reshape(-1, 1024)
#                 # ======== LLaVA-NeXT ========
#                 if self.has_cls:
#                     input_features, _ = OC_LLaVA.remove_and_remember_cls(
#                         input_features.to(device=input_features.device, dtype=torch.float32))
#                 attn_qk = self.ocl(input_features.to(device=input_features.device, dtype=torch.float32),
#                                    self.target_num // self.top_k)[1]
#                 compressed_ids = OC_LLaVA.compress_tokens_by_slots(
#                     attn_qk,
#                     self.target_num,
#                     self.top_k,
#                     self.pad_mode
#                 )
#                 if self.has_cls:
#                     compressed_ids = [idx + 1 for idx in compressed_ids]
#                     compressed_ids.insert(0, 0)
#                 self.state["compressed_ids"] = compressed_ids
#         finally:
#             self.state["busy"] = False
#
#     def hook_att(self, module, input, output):
#         if self.state["compressed_ids"] is None:
#             return
#
#         if len(self.state["compressed_ids"]) == self.target_num:
#             return
#         elif len(self.state["compressed_ids"]) > self.target_num:
#             raise ValueError("[!ERROR!] The number of compressed tokens should not exceed the target number of tokens.")
#
#         if self.has_cls:
#             num_gap = self.target_num - len(self.state["compressed_ids"])
#         else:
#             self.target_num - len(self.state["compressed_ids"])
#
#         # ======== LLaVA-NeXT ========
#         if output[1].ndim == 4:
#             if self.has_cls:
#                 output_ = output[1].detach()  # [5, H, N, N]
#                 cls_token = torch.mean(output_[:, :, :, :1], dim=0).unsqueeze(0)  # [1, H, N, 1]
#                 att = output_[:, :, :, 1:].reshape(1, output_.shape[1], output_.shape[2], -1)  # [1, H, N, 5*576]
#                 attn_scores = torch.cat((cls_token, att), dim=-1)
#             else:
#                 attn_scores = output[1].detach().reshape(1, output[1].shape[1], output[1].shape[2],
#                                                          -1)  # [1, H, N, 5*N]
#         # ======== LLaVA-NeXT ========
#         else:
#             if isinstance(output, (tuple, list)) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
#                 attn_scores = output[1].detach()  # [B, H, N, N]
#             else:
#                 return
#
#         attn_scores = attn_scores.mean(dim=1)  # [B, N, N] / [1, N, 5N]
#
#         if self.att_selection_mode == "cls":
#             attn_scores = attn_scores[:, 0, :]  # [B, N] / [1, 5N]
#         else:
#             attn_scores = attn_scores.mean(dim=1)  # [B, N] / [1, 5N]
#
#         # attn_scores = attn_scores.view(-1)  # [N]
#         # attn_ids = torch.argsort(-attn_scores)
#         #
#         N = attn_scores.numel()
#         # attn_ids_set = set(range(N)) - set(self.state["compressed_ids"])
#         #
#         # attn_ids = [int(i) for i in attn_ids.tolist() if i in attn_ids_set]
#         #
#         # # Update compress ids
#         # self.state["compressed_ids"] = self.state["compressed_ids"] + attn_ids[:num_gap]
#         # self.state["compressed_ids"].sort()
#
#         cur = self.state["compressed_ids"]
#         order = torch.argsort(-attn_scores[0])  # [N]，B=1
#         pool = [i for i in range(N) if i not in set(cur)]
#         att_rank = [int(i) for i in order.tolist() if i in pool]
#         add = att_rank[:max(0, num_gap)]
#         cur += add
#
#         if len(cur) < self.target_num:
#             rest = [i for i in pool if i not in set(add)]
#             extra = rest[:(self.target_num - len(cur))]
#             cur += extra
#
#         self.state["compressed_ids"] = sorted(cur[:self.target_num])
#
#         if self.state["handle_get_attnScores"] is not None:
#             self.state["handle_get_attnScores"].remove()
#             self.state["handle_get_attnScores"] = None
#
#     def hook_al(self, module, input, output):
#         output_, aux = output  # [1, 577, 1024] / [5, 577, 1024]
#
#         if output_.ndim == 3 and output_.shape[0] > 1:
#             if self.has_cls:
#                 cls_token = torch.mean(output_[:, :1, :].detach(), dim=0)  # [1, 1024]
#                 feat = output_[:, 1:, :].detach().reshape(-1, 1024)  # [5*576, 1024]
#                 output_ = torch.cat((cls_token, feat), dim=0).unsqueeze(0)  # [1, 2881, 1024]
#             else:
#                 output_ = output_.detach().reshape(-1, 1024).unsqueeze(0)
#
#         idx = torch.as_tensor(self.state["compressed_ids"], dtype=torch.long, device=output_.device)
#         T = output_.size(1)
#         assert int(idx.min()) >= 0 and int(idx.max()) < T, f"compressed_ids out of range: max={int(idx.max())}, T={T}"
#
#         output_ = torch.index_select(
#             output_, dim=1,
#             index=torch.tensor(self.state["compressed_ids"], dtype=torch.long, device=output_.device)
#         )
#         if self.state["handle_alternate_tokens"] is not None:
#             self.state["handle_alternate_tokens"].remove()
#             self.state["handle_alternate_tokens"] = None
#         return output_, aux
#
#     def hook_get_merging_attn(self, module, input, output):
#         if self.state["busy"] or self.state["merging_attn"] is not None:
#             return
#
#         if isinstance(output, torch.Tensor):
#             input_features = output.detach()
#         elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
#             input_features = output[0].detach()
#         else:
#             raise TypeError("Unsupported object feature type!")
#
#         self.state["busy"] = True
#         if self.state["handle_get_merging_attn"] is not None:
#             self.state["handle_get_merging_attn"].remove()
#             self.state["handle_get_merging_attn"] = None
#         try:
#             with torch.inference_mode():
#                 # ======== LLaVA-NeXT ========
#                 if input_features.ndim == 3:
#                     if self.has_cls:
#                         cls_token = torch.mean(input_features[:, :1, :].detach(), dim=0)
#                         feat = input_features[:, 1:, :].detach().reshape(-1, 1024)
#                         input_features = torch.cat((cls_token, feat), dim=0)
#                     else:
#                         input_features = input_features.detach().reshape(-1, 1024)
#                 # ======== LLaVA-NeXT ========
#                 if self.has_cls:
#                     input_features, _ = OC_LLaVA.remove_and_remember_cls(
#                         input_features.to(device=input_features.device, dtype=torch.float32))
#                 attn_qk = self.ocl(input_features.to(device=input_features.device, dtype=torch.float32),
#                                    self.target_num // self.top_k)[1]  # [slots, tokens]
#                 flag = attn_qk == attn_qk.max(dim=0, keepdim=True)
#                 attn_qk[~flag] = 0
#                 attn_qk = attn_qk / attn_qk.sum(dim=1, keepdim=True)
#                 self.state["merging_attn"] = attn_qk
#         finally:
#             self.state["busy"] = False
#
#     def hook_merging(self, module, input, output):
#         output_, aux = output  # [1, 577, 1024] / [5, 577, 1024]
#
#         if output_.ndim == 3 and output_.shape[0] > 1:
#             if self.has_cls:
#                 cls_token = torch.mean(output_[:, :1, :].detach(), dim=0)  # [1, 1024]
#                 feat = output_[:, 1:, :].detach().reshape(-1, 1024)  # [5*576, 1024]
#                 output_ = torch.cat((cls_token, feat), dim=0).unsqueeze(0)  # [1, 2881, 1024]
#             else:
#                 output_ = output_.detach().reshape(-1, 1024).unsqueeze(0)
#
#         if self.state["merging_attn"] is None:
#             raise ValueError("Attn not accessible!")
#
#         # Get features' cls token
#         if self.has_cls:
#             cls_token = output_.detach()[:, :1, :]
#             output_ = output_.detach()[:, 1:, :]
#
#         merging_attn = self.state["merging_attn"]
#         if merging_attn.ndim == 2:
#             merging_attn = merging_attn.unsqueeze(0)
#
#         if merging_attn.dtype != output_.dtype:
#             merging_attn = merging_attn.to(output_.dtype)
#         output_ = pt.einsum("bsn, bnc -> bsc", merging_attn, output_)
#
#         if self.has_cls:
#             output_ = torch.cat((cls_token, output_), dim=1)
#         if self.state["handle_merging"] is not None:
#             self.state["handle_merging"].remove()
#             self.state["handle_merging"] = None
#         return output_, aux
#
#     @staticmethod
#     def remove_and_remember_cls(feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         feats: [1, T, D] or [T, D]; assume first token is CLS if present.
#         Returns (feats_wo_cls, cls_token)
#         """
#         if feats.dim() == 2:
#             feats = feats.unsqueeze(0)
#         cls_tok = feats[:, :1, :].contiguous().clone()
#         body = feats[:, 1:, :].contiguous().clone()
#         return body, cls_tok
#
#     @staticmethod
#     def add_back_cls(compressed_feats: torch.Tensor, cls_tok: torch.Tensor) -> torch.Tensor:
#         """
#         compressed_feats: [1, M, D], cls_tok: [1,1,D] -> concat to [1, M+1, D] with CLS at front.
#         """
#         return torch.cat([cls_tok, compressed_feats], dim=1)
#
#     @staticmethod
#     def compress_tokens_by_slots(
#             attn_qk: torch.Tensor,
#             target_num: int,
#             top_k: int = 1,
#             pad_mode: str = 'att_score',
#             keep_always: Optional[List[int]] = None) -> List[int]:
#         """
#         Select token indices by argmax over tokens per slot.
#         attn_qk: [1, S, T]  -> argmax/topk over token dim (last dim).
#         keep_always: Choose to save a list of exact indices of the tokens.
#         pad_mode: 'random' | 'att_score' | 'None'
#         Returns a sorted list of INDICES IN THE "WITH-CLS" SPACE if has_cls=True (i.e., 0 is CLS).
#         """
#         # assert attn_qk.shape[0] == 1, "Batch size must be 1."
#         assert pad_mode in ['random', 'att_score', 'None'], "pad_mode must be 'random' or 'att_score' or 'None'."
#         if attn_qk.dim() == 2:
#             attn_qk = attn_qk.unsqueeze(0)
#         S = attn_qk.size(1)
#         Np = attn_qk.size(2)
#
#         _, top_idx = torch.topk(attn_qk, dim=2, k=top_k)  # [1, S, top_k]
#         top_idx = top_idx.view(-1).tolist()
#
#         pick = set(top_idx)
#
#         # force keep
#         if keep_always:
#             pick.update(keep_always)
#
#         # pad if less than target_num
#         if len(pick) < target_num and pad_mode != 'None':
#             pool = list(set(range(Np)) - pick)  # 0..Np-1
#             if pool:
#                 if pad_mode == 'random':
#                     random.shuffle(pool)
#                     need = target_num - len(pick)
#                     pick.update(pool[:need])
#
#                     # if more, trim
#         out = list(pick)
#         if len(out) > target_num:
#             out = out[:target_num]
#         out.sort()
#         return out
#
#     @staticmethod
#     def _to_batched_tensor(x, *, dtype=None, device=None):
#         import torch
#         if isinstance(x, (list, tuple)):
#             # 支持元素是 1D [T] 或 2D [1,T]
#             rows = []
#             for t in x:
#                 assert torch.is_tensor(t)
#                 if t.dim() == 2 and t.size(0) == 1:
#                     t = t[0]
#                 assert t.dim() == 1, f"expect 1D tensor per sample, got {tuple(t.shape)}"
#                 rows.append(t)
#             # 需要等长（应已被 pad），否则请先自行 pad
#             out = torch.stack(rows, dim=0)
#         elif torch.is_tensor(x):
#             out = x
#             if out.dim() == 1:  # [T] -> [1,T]
#                 out = out.unsqueeze(0)
#             assert out.dim() == 2, f"expect [B,T], got {tuple(out.shape)}"
#         else:
#             raise TypeError("Unsupported type for ids/mask")
#
#         if dtype is not None and out.dtype != dtype:
#             out = out.to(dtype)
#         if device is not None and out.device != device:
#             out = out.to(device)
#         return out
#
#     def generate(self, *args, **kwargs):
#         tok = getattr(self.llava, 'tokenizer', None) or getattr(self, 'tokenizer', None)
#
#         # ---- Tensor [B,T] ----
#         ids = kwargs.get("input_ids", None)
#         am = kwargs.get("attention_mask", None)
#
#         if ids is None and len(args) > 0 and isinstance(args[0], dict):
#             ids = args[0].get("input_ids", None)
#             am = args[0].get("attention_mask", am)
#
#         if ids is not None:
#             ids = OC_LLaVA._to_batched_tensor(ids, dtype=torch.long)
#             if am is None:
#                 pad_id = (getattr(tok, 'pad_token_id', None) or getattr(tok, 'eos_token_id', 0)) if tok else 0
#                 am = ids.ne(pad_id)
#             else:
#                 am = OC_LLaVA._to_batched_tensor(am)
#                 am = (am != 0)  # -> bool
#             assert ids.shape == am.shape, f"ids {tuple(ids.shape)} vs mask {tuple(am.shape)}"
#             kwargs["input_ids"] = ids
#             kwargs["attention_mask"] = am.to(torch.bool)
#
#             # 若 args[0] 是 batch dict，也一并回写
#             if len(args) > 0 and isinstance(args[0], dict):
#                 a0 = dict(args[0])
#                 a0["input_ids"] = ids
#                 a0["attention_mask"] = am.to(torch.bool)
#                 args = (a0, *args[1:])
#
#         if self.mode == "pruning":
#             self.state["compressed_ids"] = None
#             self.state["busy"] = False
#             self.state["handle_get_features"] = self.target_layer[self.process_layer_idx].register_forward_hook(
#                 self.hook_gf)
#             if self.pad_mode == "att_score":
#                 self.state["handle_get_attnScores"] = getattr(self.last_layer, "self_attn",
#                                                               getattr(self.last_layer, "attn",
#                                                                       None)).register_forward_hook(
#                     self.hook_att)
#             self.state["handle_alternate_tokens"] = self.last_layer.register_forward_hook(self.hook_al)
#
#         elif self.mode == "merging":
#             self.state["merging_attn"] = None
#             self.state["busy"] = False
#             self.state["handle_get_merging_attn"] = self.target_layer[self.process_layer_idx].register_forward_hook(
#                 self.hook_get_merging_attn)
#             self.state["handle_merging"] = self.target_layer[self.process_layer_idx].register_forward_hook(self.hook_merging)
#         else:
#             raise ValueError("Unsupported mode '{}'".format(self.mode))
#
#         # mm_proj = getattr(self.llava.get_model(), "mm_projector", None)
#         # mm_in_hook = None
#         # mm_out_hook = None
#         # if mm_proj is not None:
#         #     def _mm_in_hook(m, i, o):
#         #         x = i[0] if isinstance(i, (tuple, list)) else i
#         #         if isinstance(x, torch.Tensor):
#         #             print(f"[OCL][probe] mm_projector IN: {tuple(x.shape)}", flush=True)
#         #             assert x.dim() == 3, "mm_projector input must be (B, M, C)"
#         #
#         #     def _mm_out_hook(m, i, o):
#         #         y = o[0] if isinstance(o, (tuple, list)) else o
#         #         if isinstance(y, torch.Tensor):
#         #             print(f"[OCL][probe] mm_projector OUT: {tuple(y.shape)}", flush=True)
#         #
#         #     mm_in_hook = mm_proj.register_forward_hook(_mm_in_hook)
#         #     mm_out_hook = mm_proj.register_forward_hook(_mm_out_hook)
#
#         return self.llava.generate(*args, **kwargs)
#
#     def tie_weights(self, *args, **kwargs):
#         if hasattr(self.llava, "tie_weights"):
#             return self.llava.tie_weights(*args, **kwargs)
#
#     def get_output_embeddings(self):
#         if hasattr(self.llava, "get_output_embeddings"):
#             return self.llava.get_output_embeddings()
#
#     def set_output_embeddings(self, new_emb):
#         if hasattr(self.llava, "set_output_embeddings"):
#             return self.llava.set_output_embeddings(new_emb)

import re
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf
from einops import rearrange, repeat
import timm

Identity = nn.Identity


def build_ocl_model(
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
):
    encode_posit_embed = Identity()
    encode_project = MLP(in_dim=enc_in_dim, dims=enc_dims, ln=enc_ln, dropout=enc_dropout)
    # initializ = NormalSeparat(num=init_num, dim=init_dim)
    initializ = NormalShared(dim=init_dim)
    aggregat = SlotAttention(num_iter=agg_num_iter, embed_dim=agg_embed_dim, ffn_dim=agg_ffn_dim,
                             dropout=agg_dropout, trunc_bp=agg_trunc_bp)
    # Decoder
    # posit_embed = LearntPositionalEmbedding(resolut=pos_resolut, embed_dim=pos_embed_dim)
    # proj1 = Project(in_features=proj1_in_dim, out_features=proj1_out_dim, normalized_shape=proj1_norm_dim,
    #                 bias=proj1_bias)
    # proj2 = Project(in_features=proj2_in_dim, out_features=proj2_out_dim, normalized_shape=proj2_norm_dim,
    #                 bias=proj2_bias)
    # decoder_layer = TransformerDecoderLayer(
    #     d_model=transformer_d_model,
    #     nhead=transformer_nhead,
    #     dim_feedforward=transformer_feedforward,
    #     dropout=transformer_dropout,
    #     activation=transformer_activation,
    #     batch_first=transformer_batch_first,
    #     norm_first=transformer_norm_first,
    #     bias=transformer_bias,
    # )
    # backbone = TransformerDecoder(
    #     decoder_layer=decoder_layer,
    #     num_layers=backbone_nlayer,
    # )
    # readout = Identity()
    #
    # decode = ARRandTransformerDecoder(
    #     vfm_dim=dec_dim,
    #     posit_embed=posit_embed,
    #     project1=proj1,
    #     project2=proj2,
    #     backbone=backbone,
    #     readout=readout,
    # )
    model = DINOSAUR(
        encode_posit_embed=encode_posit_embed,
        encode_project=encode_project,
        initializ=initializ,
        aggregat=aggregat,
        # decode=decode,
        decode=None,
    )
    return model


class Sequential(nn.Sequential):
    def __init__(self, modules: list):
        super().__init__(*modules)

    def forward(self, input):
        for module in self:
            if isinstance(input, (list, tuple)):
                input = module(*input)
            else:
                input = module(input)
        return input


class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, interp="bilinear"):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.interp = interp

    def forward(self, input):
        return ptnf.interpolate(input, self.size, self.scale_factor, self.interp)


class DINO2ViT(nn.Module):
    def __init__(self,
                 model_name="vit_small_patch14_reg4_dinov2.lvd142m",
                 in_size=224,
                 rearrange=True,
                 norm_out=True):
        """Backbone DINOViTv2, the results can be used both as the teacher and the inputs of the slots attention."""
        super().__init__()
        self.in_size = in_size
        self.rearrange = rearrange

        # Create a DINOViTv2 model
        self.DINOv2 = timm.create_model(model_name, pretrained=True, img_size=self.in_size)
        if not norm_out:
            self.DINOv2.norm = nn.Identity()

        # Get necessary configs
        self.patch_size = self.DINOv2.patch_embed.patch_size[0]
        self.num_prefix_tokens = self.DINOv2.num_prefix_tokens
        # Ensure successful patching
        assert self.in_size % self.patch_size == 0
        self.out_size = in_size // self.patch_size

    def forward(self, input):
        features = self.DINOv2.forward_features(input)
        features = features[:, self.num_prefix_tokens:, :]
        if self.rearrange:
            features = rearrange(features, "b (h w) c -> b c h w", h=self.out_size, w=self.out_size)
        return features


class MLP(nn.Module):
    def __init__(self, in_dim, dims, ln=None, dropout=0):
        super().__init__()
        """
        DINOViT -> Slots Attention Input Projection, [

        in_dim: input dimension=384
        dims: channels of each layer [384*2, 256]
        ln should be selected from [None, 'pre', 'post']
        """
        assert ln in [None, 'pre', 'post']

        ch_in = in_dim
        layers = []

        if ln == 'pre':
            layers.append(nn.LayerNorm(ch_in))

        depth = len(dims)
        for i, ch in enumerate(dims):
            if i + 1 < depth:
                layers.append(nn.Linear(ch_in, ch))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(ch_in, ch))
            ch_in = ch

        if ln == 'post':
            layers.append(nn.LayerNorm(ch_in))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp(input)


class NormalSeparat(nn.Module):
    def __init__(self, num, dim):
        """Slots initialization"""
        super().__init__()
        self.num = num
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, self.num, self.dim))
        # self.std = nn.Parameter(pt.ones(1, self.num, self.dim))
        self.logstd = nn.Parameter((pt.ones(1, self.num, self.dim) * dim ** -0.5).log())
        # nn.init.xavier_uniform_(self.mean)
        nn.init.xavier_uniform_(self.mean[0, :, :])

    def forward(self, input):
        slots = pt.zeros(input.shape[0], self.num, self.dim).to(input.device)
        slots = slots + self.mean
        # slots = slots + pt.randn_like(slots) * self.std
        if self.training:
            slots = slots + pt.randn_like(slots) * self.logstd.exp()

        return slots


class NormalShared(nn.Module):
    """Shared gaussian as queries."""

    # TODO new trick: Conditional Random Initialization

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, 1, dim))
        self.logstd = nn.Parameter(pt.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.mean)
        nn.init.xavier_uniform_(self.logstd)

    def forward(self, b, num):
        smpl = self.mean.expand(b, num, -1)
        randn = pt.randn_like(smpl)
        smpl = smpl + randn * self.logstd.exp()
        return smpl


class SlotAttention(nn.Module):
    def __init__(self, num_iter, embed_dim, ffn_dim, dropout=0., trunc_bp=None):
        super().__init__()
        self.num_iter = num_iter
        self.embed_dim = embed_dim
        self.norm_in = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm_slots_0 = nn.LayerNorm(embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gru = nn.GRUCell(embed_dim, embed_dim)
        self.norm_slots_1 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], ln=None, dropout=dropout)

        self.trunc_bp = trunc_bp

    def forward(self, input, slots, smask=None):
        B, K, C = slots.shape
        input = self.norm_in(input)
        k = self.proj_k(input)
        v = self.proj_v(input)

        ########
        slots_ = slots
        ########
        attn = ...
        for i in range(self.num_iter):
            ########
            if i + 1 == self.num_iter:
                if self.trunc_bp == "bi-level":
                    slots_ = slots_.detach() + slots - slots.detach()
            ########
            slots_prev = slots_
            slots_ = self.norm_slots_0(slots_)
            q = self.proj_q(slots_)

            product = pt.einsum("bqc, bkc -> bqk", q, k)
            attn = product * (q.shape[-1] ** -0.5)
            ########
            if smask is not None:
                attn = attn.where(smask[:, :, None], -pt.inf)
            ########
            attn = pt.softmax(attn, dim=1)  # [b, q, k]: [b, K, h*w]

            updates = attn / (attn.sum(dim=-1, keepdim=True) + 1e-5)
            updates = pt.einsum("bqv, bvc -> bqc", updates, v)

            slots_ = self.gru(updates.flatten(0, 1), slots_prev.flatten(0, 1))
            slots_ = slots_.view(B, K, -1)

            slots_ = slots_ + self.ffn(self.norm_slots_1(slots_))

        return slots_, attn


# class LearntPositionalEmbedding(nn.Module):
#     def __init__(self, resolut, embed_dim):
#         super().__init__()
#         self.resolut = resolut
#         self.embed_dim = embed_dim
#         # self.pos_embed = nn.Parameter(pt.randn(1, *resolut, embed_dim), requires_grad=True)
#         self.pos_embed = nn.Parameter(pt.zeros(1, *resolut, embed_dim), requires_grad=True)
#         nn.init.trunc_normal_(self.pos_embed)
#
#     def forward(self, input, retp=False):
#         # return x + self.pos_embed
#         ########
#         max_r = ", ".join([f":{_}" for _ in input.shape[1:-1]])
#         # TODO XXX support variant length
#         # pe = timm.layers.pos_embed.resample_abs_pos_embed(self.pe, ...)
#         # pe = self.pe[:, :max_resolut, :]
#         pe = eval(f"self.pos_embed[:, {max_r}, :]")
#         output = input + pe
#         if retp:
#             return output, pe
#         return output
#         ########


class LearntPositionalEmbedding(nn.Module):
    """Support any dimension. Must be channel-last.
    PositionalEncoding: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, resolut: list, embed_dim: int, in_dim: int = 0):
        super().__init__()
        self.resolut = resolut
        self.embed_dim = embed_dim
        if in_dim:
            self._pe = nn.Parameter(pt.zeros(1, *resolut, in_dim), requires_grad=True)
            self._project = nn.Linear(in_dim, embed_dim)
        else:
            self._pe = nn.Parameter(
                pt.zeros(1, *resolut, embed_dim), requires_grad=True
            )
        nn.init.trunc_normal_(self._pe)

    @property
    def pe(self):
        if hasattr(self, "_project"):
            return self._project(self._pe)
        return self._pe

    def forward(self, input, retp=False):
        """
        input: in shape (b,*r,c)
        output: in shape (b,*r,c)
        """
        max_r = ", ".join([f":{_}" for _ in input.shape[1:-1]])
        # TODO XXX support variant length
        # pe = timm.layers.pos_embed.resample_abs_pos_embed(self.pe, ...)
        # pe = self.pe[:, :max_resolut, :]
        pe = eval(f"self.pe[:, {max_r}, :]")
        output = input + pe
        if retp:
            return output, pe
        return output

    def extra_repr(self):
        return f"{self.resolut}, {self.embed_dim}"
        ########


class Project(nn.Module):
    def __init__(self, in_features, out_features, normalized_shape, bias=False):
        super().__init__()
        self.project = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, input):
        input = self.project(input)
        input = self.norm(input)
        return input


TransformerDecoderLayer = nn.TransformerDecoderLayer
TransformerDecoder = nn.TransformerDecoder


class ARRandTransformerDecoder(nn.Module):
    """GeneralZ's new OCL decoder.
    Auto-regressive Transformer decoder with random token permutations.
    """

    def __init__(
            self,
            vfm_dim,
            posit_embed,
            # posit_embed_hw,
            project1,
            project2,
            backbone,
            readout,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(pt.randn(1, 1, vfm_dim) * vfm_dim ** -0.5)
        assert hasattr(posit_embed, "pe")
        self.posit_embed = posit_embed  # 1d
        # self.posit_embed_hw = posit_embed_hw  # 2d
        self.project1 = project1
        self.project2 = project2

        assert isinstance(backbone, nn.TransformerDecoder)
        self.norm0 = backbone.layers[0].norm1  # very beneficial
        backbone.layers[0].norm1 = nn.Identity()  # very beneficial
        self.backbone = backbone
        self.readout = readout

        def attent_hook_forward_pre(module, args, kwargs):
            kwargs["need_weights"] = True  # obtain the attention weights

        def attent_hook_forward(module, args, output):
            self._attent = output[1]

        self.backbone.layers[-1].multihead_attn.register_forward_pre_hook(
            attent_hook_forward_pre, with_kwargs=True
        )
        self.backbone.layers[-1].multihead_attn.register_forward_hook(
            attent_hook_forward
        )

        ### interaction asymmetry

        self._interact = [None for _ in range(len(self.backbone.layers[:-1]))]
        for l, layer in enumerate(self.backbone.layers[:-1]):
            def interact_hook_forward(module, args, output):
                self._interact[l] = output[1]

            layer.multihead_attn.register_forward_pre_hook(
                attent_hook_forward_pre, with_kwargs=True
            )
            layer.multihead_attn.register_forward_hook(interact_hook_forward)

    def forward(self, input, slots, p=0.5):
        """
        input: target to be destructed, shape=(b,m=h*w,c)
        slots: slots, shape=(b,n,c)
        """
        b, m, c = input.shape
        assert m == self.posit_embed.pe.size(1)
        _, n, _ = slots.shape
        device = input.device
        tokens = self.project1(input)  # (b,m,c)

        # TODO XXX disable masking in val for attent2 !!!

        # mim-predict-all-masked-tokens
        # seg1:
        # "ari": 0.20348355174064636, "ari_fg": 0.34435588121414185, "mbo": 0.29168349504470825, "miou": 0.2779198884963989
        # seg2:  # TODO disable masking in val for attent2 !!!
        # 'ari': 0.2038770616054535, 'ari_fg': 0.3444632291793823, 'mbo': 0.29167482256889343, 'miou': 0.27789679169654846

        if self.training:
            idxs = pt.vmap(  # (b,m)
                lambda _: pt.randperm(m, device=device), randomness="different"
            )(tokens)
            idxs_expanded = idxs[:, :, None].expand(-1, -1, c)

            idxs0 = pt.arange(0, m, device=device)[None, :]  # (1,m)
            keep1 = pt.randint(0, m - 1, [b, 1], device=device)  # (b,1)
            keep2 = (
                    pt.ones(b, 1, dtype=pt.long, device=device) * int(256 * 0.1) - 1
            )  # TODO
            # TODO XXX realize a Poisson: when in [0, 1], it is Poisson; when out, then uniformly re-distribute in [0, 1]
            cond = pt.rand(b, 1, device=device) < p
            keep = pt.where(cond, keep1, keep2)
            # XXX 论文论述 XXX
            # keep@0: SlotMixerDecoder
            # 只预测下一个：ARTransformerDecoder
            # 其中9种只预测下一个，AR9TransformerDecoder
            mask = idxs0 < keep  # (b,m)

            # shuffle tokens
            tokens_shuffled = tokens.gather(1, idxs_expanded)  # (b,m,c)
            # mask tokens
            mask_token_expanded = self.mask_token.expand(b, m, -1)
            tokens_masked = tokens_shuffled.where(mask[:, :, None], mask_token_expanded)

            # shuffle pe
            pe_expanded = self.posit_embed.pe[:, :m, :].expand(b, -1, -1)  # (b,m,c)
            # pe_hw_expanded = self.posit_embed_hw.pe.flatten(1, -2)[:, :m, :].expand(
            #     b, -1, -1
            # )  # (b,m,c)
            pe_shuffled = pe_expanded.gather(1, idxs_expanded)  # (b,m,c)
            # pe_hw_shuffled = pe_hw_expanded.gather(1, idxs_expanded)  # (b,m,c)

            query = tokens_masked + pe_shuffled  # + pe_hw_shuffled

        else:
            query = (
                    tokens
                    + self.posit_embed.pe[:, :m, :]
                # + self.posit_embed_hw.pe.flatten(1, -2)[:, :m, :]
            )

        memory = self.project2(slots)
        autoreg = self.backbone(self.norm0(query), memory=memory)
        recon = self.readout(autoreg)  # (b,m,c)
        _, _, d = recon.shape

        if self.training:
            idxs_inverse = idxs.argsort(1)[:, :, None]
            recon = recon.gather(1, idxs_inverse.expand(-1, -1, d))
            attent = self._attent.gather(1, idxs_inverse.expand(-1, -1, n))
        else:
            attent = self._attent

        attent = rearrange(attent, "b m n -> b n m")
        return recon, attent


class BroadcastMLPDecoder(nn.Module):
    def __init__(self, posit_embed, backbone):
        super().__init__()
        self.posit_embed = posit_embed
        self.backbone = backbone

    def forward(self, slots, n_tokens):
        B, K, C = slots.shape

        # Apply slots on each patch
        broad_slots = slots.reshape(-1, 1, C)
        broad_slots = broad_slots.repeat(1, n_tokens, 1)

        embeded_slots = self.posit_embed(broad_slots)

        recon = self.backbone(embeded_slots)  # [B*K, n, 384 + 1]
        recon, alpha = recon[:, :, :-1], recon[:, :, -1:]  # [B*K, n, 384], [B*K, n, 1]

        recon = recon.reshape(B, K, n_tokens, -1)
        alpha = alpha.reshape(B, K, n_tokens, 1)
        alpha = alpha.softmax(dim=1)

        recon = (recon * alpha).sum(dim=1)

        return recon, alpha[:, :, :, 0]


# class DINOSAUR(nn.Module):
#     def __init__(self, encode_posit_embed, encode_project, initializ, aggregat, decode):
#         super().__init__()
#         # self.encode_backbone = encode_backbone
#         self.encode_posit_embed = encode_posit_embed
#         self.encode_project = encode_project
#         self.initializ = initializ
#         self.aggregat = aggregat
#         self.decode = decode
#
#         ################
#         self.reset_parameters(
#             [self.encode_posit_embed, self.encode_project, self.aggregat]
#         )
#
#     @staticmethod
#     def reset_parameters(modules):
#         for module in modules:
#             for m in module.modules():
#                 if isinstance(m, nn.Conv2d):
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, nn.Linear):
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, nn.GRUCell):
#                     if m.bias:
#                         nn.init.zeros_(m.bias_ih)
#                         nn.init.zeros_(m.bias_hh)
#
#     ################
#
#     def forward(self, features):
#         # features = self.encode_backbone(input).detach()  # Freeze backbone, gradients not pass
#         b, t, c = features.shape  # [b, tokens, channels]
#
#         encode = self.encode_project(features)  # [b, tokens, proj_dims]: [b, 576, 512]
#
#         slots_init = self.initializ(encode)
#
#         slots, attn = self.aggregat(encode, slots_init)  # slots: [b, K, h_dim=512] attn: [b, K, tokens]
#         recon, alpha = self.decode(slots, t)  # recon: [b, tokens, c] alpha: [b, K, tokens]
#
#         return slots, attn, alpha, recon


class DINOSAUR(nn.Module):
    def __init__(self, encode_posit_embed, encode_project, initializ, aggregat, decode):
        super().__init__()
        # self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.decode = decode

        ################
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat]
        )

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    ################

    def forward(self, features, n_slots):
        # features = self.encode_backbone(input).detach()  # Freeze backbone, gradients not pass
        b, t, c = features.shape  # [b, tokens, channels]

        encode = self.encode_project(features)  # [b, tokens, proj_dims]: [b, 576, 512]

        slots_init = self.initializ(encode.shape[0], n_slots)

        slots, attn = self.aggregat(encode, slots_init)  # slots: [b, K, h_dim=512] attn: [b, K, tokens]

        if self.decode is not None:
            recon, alpha = self.decode(features, slots)  # recon: [b, tokens, c] alpha: [b, K, tokens]
            return slots, attn, alpha, recon
        else:
            return slots, attn


import torch
import random
from typing import Tuple, List, Optional
import numpy as np
import torch.nn as nn


class OC_LLaVA(nn.Module):
    def __init__(
            self,
            ocl_model,
            llava_model,
            mode="pruning",
            process_layer_idx=8,
            target_num: int = 64,
            top_k: int = 1,
            has_cls: bool = True,
            pad_mode: str = "att_score",
            att_selection_mode: str = "meanq",
    ):
        super().__init__()
        # ------------- Model Initialization -------------
        self.ocl = ocl_model
        self.llava = llava_model

        # Configs for hook
        self.process_layer_idx = process_layer_idx
        self.mode = mode
        self.has_cls = has_cls

        if mode == "pruning" and self.has_cls:
            self.target_num = target_num + 1
        else:
            self.target_num = target_num
        self.top_k = top_k
        if process_layer_idx == -2:
            pad_mode = "random"
        self.pad_mode = pad_mode
        self.att_selection_mode = att_selection_mode

        # Define a hook and register
        self.vision = self.llava.get_vision_tower().vision_tower
        # vision.output_attentions = True
        if hasattr(self.vision, "vision_model"):
            self.vision.vision_model.config.output_attentions = True

        self.target_layer = self.vision.vision_model.encoder.layers
        self.last_layer = self.target_layer[-2]

        # Get features
        self.state = {
            "compressed_ids": None,
            "busy": False,
            "handle_get_features": None,

            "handle_get_attnScores": None,
            "handle_alternate_tokens": None,
        }

        self.config = self.llava.config

    def forward(self, *args, **kwargs):
        # self.state["compressed_ids"] = None
        # self.state["busy"] = False
        #
        # self.state["handle_get_features"] = self.target_layer[self.process_layer_idx].register_forward_hook(
        #     self.hook_gf)
        #
        # if self.pad_mode == "att_score":
        #     self.state["handle_get_attnScores"] = getattr(self.last_layer, "self_attn",
        #                                                   getattr(self.last_layer, "attn", None)).register_forward_hook(
        #         self.hook_att)
        #
        # # Select indices and alternate the final tokens
        # self.state["handle_alternate_tokens"] = self.target_layer[-1].register_forward_hook(
        #     self.hook_al)
        #
        # out = self.llava(*args, **kwargs)
        #
        # for key in ("handle_get_features", "handle_alternate_tokens"):
        #     if self.state.get(key) is not None:
        #         try:
        #             self.state[key].remove()
        #         except Exception:
        #             raise ValueError(f"{key} has no attribute remove.")
        #         self.state[key] = None
        # if self.pad_mode == "att_score":
        #     if self.state["handle_get_attnScores"] is not None:
        #         self.state["handle_get_attnScores"].remove()
        #         self.state["handle_get_attnScores"] = None
        #
        # self.state["compressed_ids"] = None
        # self.state["busy"] = False
        # return out
        pass

    def hook_gf(self, module, input, output):
        if self.state["busy"] or self.state["compressed_ids"] is not None:
            return

        if isinstance(output, torch.Tensor):
            input_features = output.detach()
        elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
            input_features = output[0].detach()
        else:
            raise TypeError("Unsupported object feature type!")

        self.state["busy"] = True
        if self.state["handle_get_features"] is not None:
            self.state["handle_get_features"].remove()
            self.state["handle_get_features"] = None
        try:
            with torch.inference_mode():
                input_features, _ = OC_LLaVA.remove_and_remember_cls(
                    input_features.to(device=input_features.device, dtype=torch.float32))
                attn_qk = self.ocl(input_features.to(device=input_features.device, dtype=torch.float32),
                                   self.target_num // self.top_k)[1]
                compressed_ids = OC_LLaVA.compress_tokens_by_slots(
                    attn_qk,
                    self.target_num,
                    self.top_k,
                    self.has_cls,
                    self.pad_mode
                )
                self.state["compressed_ids"] = compressed_ids
        finally:
            self.state["busy"] = False

    def hook_att(self, module, input, output):
        if self.state["compressed_ids"] is None:
            return

        if len(self.state["compressed_ids"]) == self.target_num:
            return
        elif len(self.state["compressed_ids"]) > self.target_num:
            raise ValueError("[!ERROR!] The number of compressed tokens should not exceed the target number of tokens.")
        num_gap = self.target_num - len(self.state["compressed_ids"])

        if isinstance(output, (tuple, list)) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
            attn_scores = output[1].detach()  # [B, H, N, N]
        else:
            return

        attn_scores = attn_scores.mean(dim=1)  # [B, N, N]

        if self.att_selection_mode == "cls":
            attn_scores = attn_scores[:, 0, :]  # [B, N]
        else:
            attn_scores = attn_scores.mean(dim=1)  # [B, N]

        # attn_scores = attn_scores.view(-1)  # [N]
        # attn_ids = torch.argsort(-attn_scores)
        #
        N = attn_scores.numel()
        # attn_ids_set = set(range(N)) - set(self.state["compressed_ids"])
        #
        # attn_ids = [int(i) for i in attn_ids.tolist() if i in attn_ids_set]
        #
        # # Update compress ids
        # self.state["compressed_ids"] = self.state["compressed_ids"] + attn_ids[:num_gap]
        # self.state["compressed_ids"].sort()

        cur = self.state["compressed_ids"]
        order = torch.argsort(-attn_scores[0])  # [N]，B=1
        pool = [i for i in range(N) if i not in set(cur)]
        att_rank = [int(i) for i in order.tolist() if i in pool]
        add = att_rank[:max(0, num_gap)]
        cur += add

        if len(cur) < self.target_num:
            rest = [i for i in pool if i not in set(add)]
            extra = rest[:(self.target_num - len(cur))]
            cur += extra

        self.state["compressed_ids"] = sorted(cur[:self.target_num])

        if self.state["handle_get_attnScores"] is not None:
            self.state["handle_get_attnScores"].remove()
            self.state["handle_get_attnScores"] = None

    def hook_al(self, module, input, output):
        output_, aux = output  # [1, 577, 1024]

        output_ = torch.index_select(
            output_, dim=1,
            index=torch.tensor(self.state["compressed_ids"], dtype=torch.long, device=output_.device)
        )
        if self.state["handle_alternate_tokens"] is not None:
            self.state["handle_alternate_tokens"].remove()
            self.state["handle_alternate_tokens"] = None
        return output_, aux

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

    @staticmethod
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

                    # if more, trim
        out = list(pick)
        if len(out) > target_num:
            out = out[:target_num]
        out.sort()
        return out

    def generate(self, *args, **kwargs):
        self.state["compressed_ids"] = None
        self.state["busy"] = False

        self.state["handle_get_features"] = self.target_layer[self.process_layer_idx].register_forward_hook(
            self.hook_gf)

        if self.pad_mode == "att_score":
            self.state["handle_get_attnScores"] = getattr(self.last_layer, "self_attn",
                                                          getattr(self.last_layer, "attn", None)).register_forward_hook(
                self.hook_att)

        # Select indices and alternate the final tokens
        self.state["handle_alternate_tokens"] = self.last_layer.register_forward_hook(
            self.hook_al)

        # mm_proj = getattr(self.llava.get_model(), "mm_projector", None)
        # mm_in_hook = None
        # mm_out_hook = None
        # if mm_proj is not None:
        #     def _mm_in_hook(m, i, o):
        #         x = i[0] if isinstance(i, (tuple, list)) else i
        #         if isinstance(x, torch.Tensor):
        #             print(f"[OCL][probe] mm_projector IN: {tuple(x.shape)}", flush=True)
        #             assert x.dim() == 3, "mm_projector input must be (B, M, C)"
        #
        #     def _mm_out_hook(m, i, o):
        #         y = o[0] if isinstance(o, (tuple, list)) else o
        #         if isinstance(y, torch.Tensor):
        #             print(f"[OCL][probe] mm_projector OUT: {tuple(y.shape)}", flush=True)
        #
        #     mm_in_hook = mm_proj.register_forward_hook(_mm_in_hook)
        #     mm_out_hook = mm_proj.register_forward_hook(_mm_out_hook)

        return self.llava.generate(*args, **kwargs)

    def tie_weights(self, *args, **kwargs):
        if hasattr(self.llava, "tie_weights"):
            return self.llava.tie_weights(*args, **kwargs)

    def get_output_embeddings(self):
        if hasattr(self.llava, "get_output_embeddings"):
            return self.llava.get_output_embeddings()

    def set_output_embeddings(self, new_emb):
        if hasattr(self.llava, "set_output_embeddings"):
            return self.llava.set_output_embeddings(new_emb)

