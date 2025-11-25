#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn(
                'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'),
                                                 map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')

                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                                   non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                       non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'),
                                    os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                            **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                              **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True,
                                                             **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    # ================================== OC-LLaVA ================================== #
    def _get_bool_env(k, default="0"):
        try:
            return bool(int(os.getenv(k, default)))
        except Exception:
            return False

    if _get_bool_env("OCL_ENABLE", "0"):
        """LLaVA-NeXT"""
        # Ensure the number of original vision tokens is 2880.
        pinpoints = [[672, 672]]
        if hasattr(model, "config") and hasattr(model.config, "image_grid_pinpoints"):
            model.config.image_grid_pinpoints = pinpoints
        """LLaVA-NeXT"""

        # Ensure vision tower and attentions
        try:
            vt = model.get_vision_tower().vision_tower
            if hasattr(vt, "vision_model"):
                vt.vision_model.config.output_attentions = True
        except Exception as e:
            print(f"[OCL] warn: cannot enable output_attentions: {e}")

        # Load OCL
        try:
            from llava.model.ocl_model import OC_LLaVA, build_ocl_model
        except Exception as e:
            raise ImportError(f"[OCL] cannot import OC_LLaVA/build_ocl_model: {e}")

        # json or yaml
        ocl_cfg = {}
        ocl_cfg_path = os.getenv("OCL_CONFIG", "")
        if ocl_cfg_path:
            try:
                if ocl_cfg_path.lower().endswith((".yml", ".yaml")):
                    import yaml
                    with open(ocl_cfg_path, "r") as f:
                        ocl_cfg = yaml.safe_load(f) or {}
                else:
                    import json
                    with open(ocl_cfg_path, "r") as f:
                        ocl_cfg = json.load(f) or {}
                if not isinstance(ocl_cfg, dict):
                    raise ValueError("OCL_CONFIG must load to a dict.")
            except Exception as e:
                raise RuntimeError(f"[OCL] failed to load OCL_CONFIG: {ocl_cfg_path}, error: {e}")

        mode = str(os.getenv("OCL_MODE", "pruning"))
        process_layer_idx = int(os.getenv("OCL_LAYER_IDX", "8"))
        target_num = int(os.getenv("OCL_TARGET_NUM", "64"))
        top_k = int(os.getenv("OCL_TOPK", "1"))
        has_cls = bool(int(os.getenv("OCL_HAS_CLS", "1")))
        pad_mode = os.getenv("OCL_PAD_MODE", "att_score")  # random / att_score / None
        att_sel = os.getenv("OCL_ATTSEL", "meanq")  # cls / meanq

        if "init_num" not in ocl_cfg or ocl_cfg.get("init_num") is None:
            ocl_cfg["init_num"] = target_num - 1 if has_cls else target_num

        try:
            ocl = build_ocl_model(**ocl_cfg)
        except TypeError as e:
            raise TypeError(f"[OCL] build_ocl_model(**cfg) failed; check keys in {ocl_cfg_path}. Error: {e}")

        ocl_ckpt = os.getenv("OCL_CKPT", "")
        if ocl_ckpt:
            state = torch.load(ocl_ckpt, map_location="cpu")
            state = state.get("state_dict", state)
            missing, unexpected = ocl.load_state_dict(state, strict=False)
            print(f"[OCL] loaded ckpt: {ocl_ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")

        # LLaVA -> OC_LLaVA
        wrapped = OC_LLaVA(
            ocl_model=ocl.to(device=model.device, dtype=torch.float32),
            llava_model=model,
            mode=mode,
            process_layer_idx=process_layer_idx,
            target_num=target_num,
            top_k=top_k,
            has_cls=has_cls,
            pad_mode=pad_mode,
            att_selection_mode=att_sel,
        )

        try:
            dev = device if device != "cuda" else (device_map if isinstance(device_map, str) else "cuda")
            if dev != "auto":
                if isinstance(dev, str) and dev.startswith("cuda"):
                    wrapped.to(torch.device(dev))
                elif isinstance(dev, str):
                    wrapped.to(dev)
        except Exception as e:
            print(f"[OCL] warn: device placement failed, continuing: {e}")

        model = wrapped
        print("[OCL] OC_LLaVA wrapper enabled. (config file loaded)")
    # ================================== OC-LLaVA ================================== #
    return tokenizer, model, image_processor, context_len
