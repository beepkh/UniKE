<h1 align = "center">
Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration
</h1>

This repo contains the PyTorch implementation of [Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration](https://arxiv.org/abs/2409.19872)

## 1. Environment Setup

```bash
conda create -n unike python=3.10 -y
conda activate unike

pip install -r requirements.txt
```

## 2. Model Download

You should first prepare **MiniGPT-4** as the MLLM (with Vicuna-7B backbone). Download the following model weights and update the corresponding paths in `hparams/minigpt4_caption.yaml` or `hparams/minigpt4_vqa.yaml`.

### 2.1 Base Models

| Model | Config Field | Source |
|---|---|---|
| Vicuna-7B | `name`, `tokenizer_name` | [vicuna-7b](https://huggingface.co/Vision-CAIR/vicuna-7b) |
| BERT (base, uncased) | `qformer_name_or_path` | [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| EVA-ViT-G | `state_dict_file` | [eva_vit_g.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth) |
| BLIP-2 Q-Former | `qformer_checkpoint` | [blip2_pretrained_flant5xxl.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth) |
| MiniGPT-4 Pretrained | `pretrained_ckpt` | [pretrained_minigpt4_7b.pth](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view) |

> We recommend downloading all HuggingFace models to `./hugging_cache/`. For example, download `bert-base-uncased` to `./hugging_cache/bert-base-uncased`.
>
> If you choose a different download location, you also need to modify `cache_dir='./hugging_cache'` in `easyeditor/trainer/blip2_models/blip2.py` to the parent directory of your `bert-base-uncased`.


### 2.2 UniKE Checkpoints

Download the checkpoints and knowledge states from [Parva1012/UniKE_ckpt](https://huggingface.co/Parva1012/UniKE_ckpt) for UniKE and update these paths in the YAML config:

| File | Config Field |
|---|---|
| `ike_*.pth` (directory) | `retrieved_knowledge_path` |
| `l-ike.pth` | `latent_ike_path` |
| `semantic_encoder.pt` | `semantic_encoder_path` |

### 2.3 Data & Images

Download the MMEdit datasets and place the JSON files at `./data/MMEdit/editing-data/caption/caption_eval_edit.json` (for Caption) or `./data/MMEdit/editing-data/vqa/vqa_eval.json` (for VQA). For details, refer to [MMEdit.md](https://github.com/zjunlp/EasyEdit/blob/main/examples/MMEdit.md).

| **Dataset** | Google Drive | BaiduNetDisk | Description |
| :---: | :---: | :---: | :---: |
| E-IC | [[Google Drive]](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link) | [[BaiduNetDisk]](https://pan.baidu.com/s/1g9nMv-5BJmztxYU-BWRdvg?pwd=ik5c) | Dataset for editing *Image Captioning* |
| E-VQA | [[Google Drive]](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link) | [[BaiduNetDisk]](https://pan.baidu.com/s/1g9nMv-5BJmztxYU-BWRdvg?pwd=ik5c) | Dataset for editing *Visual Question Answering* |

- All **images** used in **E-IC** and **E-VQA** are available for download at [Google Drive](https://drive.google.com/file/d/1fQzJBFkok5kFZT6QUuT-HCuYKk2Vb93O/view) or [BaiduNetDisk](https://pan.baidu.com/s/15WJuPOvxRyF6GDZ-B6_NiA?pwd=wqlg).
- After downloading, update `coco_image` and `rephrase_image` in the YAML config to point to your image directory.

### 2.4 Update YAML Config

After downloading all models, update `hparams/minigpt4_vqa.yaml` (or the caption variant):

```yaml
name: /path/to/vicuna-7b
tokenizer_name: /path/to/vicuna-7b
qformer_name_or_path: /path/to/bert-base-uncased
qformer_checkpoint: /path/to/blip2_pretrained_flant5xxl.pth
state_dict_file: /path/to/eva_vit_g.pth
pretrained_ckpt: /path/to/pretrained_minigpt4_7b.pth
coco_image: /path/to/images
rephrase_image: /path/to/images
# /path/to/images should contain two subdirectories: val2014 and val2014_image_rephrase
retrieved_knowledge_path: /path/to/UniKE_ckpt/minigpt4/retrieved_ik_states/mmedit/caption (or vqa)
latent_ike_path: /path/to/UniKE_ckpt/minigpt4/ckpt/l-ike.pth
semantic_encoder_path: /path/to/UniKE_ckpt/minigpt4/ckpt/semantic_encoder.pt
```

## 3. Running

### VQA Task

```bash
python run_edit.py \
    --task vqa \
    --hparams_dir ./hparams/minigpt4_vqa.yaml \
    --data_path /path/to/vqa_eval.json \
    --gpu_id 0
```

### Caption Task

```bash
python run_edit.py \
    --task caption \
    --hparams_dir ./hparams/minigpt4_caption.yaml \
    --data_path /path/to/caption_eval_edit.json \
    --gpu_id 0
```

### Optional Arguments

| Argument | Description |
|---|---|
| `--save_name` | Custom output filename (without `.json`), saved to `./logs/{data_name}/{save_name}.json` |
| `--retrieved_knowledge_path` | Override retrieved knowledge directory (contains per-sample `ike_*.pth`) |
| `--latent_ike_path` | Override latent IKE path (`l-ike.pth`) |
| `--semantic_encoder_path` | Override semantic encoder path (`semantic_encoder.pt`) |


Results will be saved to `./logs/{data_name}/{save_name}.json` (e.g., `./logs/vqa_eval/{save_name}.json` for VQA or `./logs/caption_eval_edit/{save_name}.json` for Caption). 


## Acknowledgment

Our project is developed based on the following repositories:

* [EasyEdit](https://github.com/zjunlp/EasyEdit): An Easy-to-use Knowledge Editing Framework for LLMs.

* [Transformer-Patcher](https://github.com/ZeroYuHuang/Transformer-Patcher): One mistake worth one neuron.

## Citation
If you found this work useful, please consider  citing our paper as follows:
```
@article{pan2024towards,
  title={Towards unified multimodal editing with enhanced knowledge collaboration},
  author={Pan, Kaihang and Fan, Zhaoyu and Li, Juncheng and Yu, Qifan and Fei, Hao and Tang, Siliang and Hong, Richang and Zhang, Hanwang and Sun, Qianru},
  journal={arXiv preprint arXiv:2409.19872},
  year={2024}
}
```
