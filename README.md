# Object-Centric Vision Token Pruning for Vision Language Models

## Contributions

<img src="./imgs/Teaser.png" alt="image-20251125160138224" style="zoom:96%;" />

We are **the first** to realize guaranteed vision token pruning (VTP)  that can select **the most representative** tokens. Our method achieves roughly new SotA in terms of VTP for VLM with intuitive object-level interpretability.

- Our OC-VTP achieves over 95% of the performance with only 10% retained visual tokens.
- OC-Pruner only requires training once without further fine-tuning and the training can easily basing on CoCo dataset.
- OC-VTP saves nearly 85% FLOPs on LLaVA-1.5 and 95% FLOPs on LLaVA-NeXT while remaining comparable performance. OC-Pruner only requires 1/1000 FLOPs cost compared to VLMs. 
- OC-VTP also saves inference time, whose efficiency performance is similar to the training-free pruners.

## Installation

1. Install the [LLaVA](https://github.com/haotian-liu/LLaVA) environment.
2. Run followings to get project codes:

```
git clone https://github.com/GarryLarry010131/OC-VTP
```

3. Use `requirements.txt` to build an environment.

## Training

1. Use [`get_training_only_data.py`](./OC-VTP/get_training_only_data.py) first to extract training-ready features.
2. Use [`train_transDecoder_noVal.py`](./OC-VTP/train_transDecoder_noVal.py) to train the OC-Pruner and validate the pruner only through the validation loss.

 [`model.py`](./OC-VTP/model.py) contains the structure of OC-VTP.

 [`utils.py`](./OC-VTP/utils.py) contains the AW-MSE loss and functions for plotting.

## Evaluation

We use LMMs-Eval to evaluate OC-VTP across different benchmarks. Use following codes to run the evaluation:

```bash
export HF_ENDPOINT=https://hf-mirror.com # If you encounter network issue, please uncomment this
export CUDA_VISIBLE_DEVICES=0

export OCL_ENABLE=1
export OCL_MODE=pruning
export OCL_LAYER_IDX=8  # 0 ~ N
export OCL_CONFIG=OC-VTP/OCL_light_decoder_config/oc-vlm_64.json
export OCL_CKPT=...
export OCL_TARGET_NUM=64
export OCL_TOPK=1
export OCL_HAS_CLS=1
export OCL_PAD_MODE=att_score
export OCL_ATTSEL=meanq

ckpt=liuhaotian/llava-v1.5-7b
tasks=("gqa" "mmbench" "mme" "pope" "scienceqa_img" "vizwiz_vqa_val" "mmmu" "seedbench")
for task in "${tasks[@]}"; do
    accelerate launch --num_processes=1 -m lmms_eval --model llava \
        --model_args pretrained=$ckpt \
        --tasks "$task" --batch_size 1 --log_samples \
        --log_samples_suffix reproduce --output_path ./test_logs_oc-vlm/layer_9/test_logs_"$task"/
done
```

After this,  [`figure_plotting.py`](./OC-VTP/figure_plotting.py) can be used to plot **token-selection images**.

## Acknowledgement

This work is built basing on  [LLaVA](https://github.com/haotian-liu/LLaVA) and [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Thanks to their contributions!

## Citation

If you think it is helpful, please consider citing our work!

```
@misc{li2025objectcentricvisiontokenpruning,
      title={Object-Centric Vision Token Pruning for Vision Language Models}, 
      author={Guangyuan Li and Rongzhen Zhao and Jinhong Deng and Yanbo Wang and Joni Pajarinen},
      year={2025},
      eprint={2511.20439},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.20439}, 
}
```

