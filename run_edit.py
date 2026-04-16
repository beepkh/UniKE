import os
import argparse
from utils import *
parser = argparse.ArgumentParser(description='train args')
parser.add_argument('--task', type=str, metavar='TASK', choices=['caption', 'vqa'], default='vqa', help='Task type')
parser.add_argument('--hparams_dir', type=str, default='./hparams/minigpt4_vqa.yaml')
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--data_path', type=str, default=None,
                    help="Override data path (VQA json for vqa task, Caption json for caption task)")
parser.add_argument('--save_name', type=str, default=None,
                    help="Override output filename (without .json). "
                         "Saved to ./logs/{data_filename}/{save_name}.json")
parser.add_argument('--retrieved_knowledge_path', type=str, default=None,
                    help="Override retrieved knowledge directory (per-sample ike_*.pth)")
parser.add_argument('--latent_ike_path', type=str, default=None,
                    help="Override latent IKE path (l-ike.pth)")
parser.add_argument('--semantic_encoder_path', type=str, default=None,
                    help="Override semantic encoder path (semantic_encoder.pt)")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
from typing import *
import torch
from torch.utils.data import Dataset, ConcatDataset
import types
from itertools import chain
from dataclasses import asdict
from statistics import mean
from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset
from easyeditor.dataset.processor.base_dataset import BaseDataset
from easyeditor import UniKEHyperParams
from easyeditor.util.hparams import HyperParams
from utils import *
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


def edit_UNIKE_MiniGPT4_Caption(hparam_path, data_path=None, save_name=None,
                                retrieved_knowledge_path=None, latent_ike_path=None, semantic_encoder_path=None):
    hparams = UniKEHyperParams.from_hparams(hparam_path)
    hparams.task_name = 'caption'
    if retrieved_knowledge_path is not None:
        hparams.retrieved_knowledge_path = retrieved_knowledge_path
    if latent_ike_path is not None:
        hparams.latent_ike_path = latent_ike_path
    if semantic_encoder_path is not None:
        hparams.semantic_encoder_path = semantic_encoder_path
    file_path = data_path or './data/MMEdit/editing-data/caption/caption_eval_edit.json'
    eval_ds = CaptionDataset(file_path, config=hparams)
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='caption'
    )
    if save_name:
        out_path = f'./logs/{get_filename(file_path)}/{save_name}.json'
    else:
        out_path = f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json'
    parse_result(metrics, out_path, config=hparams)

def edit_UNIKE_MiniGPT4_VQA(hparam_path, data_path=None, save_name=None,
                            retrieved_knowledge_path=None, latent_ike_path=None, semantic_encoder_path=None):
    hparams = UniKEHyperParams.from_hparams(hparam_path)
    hparams.task_name = 'vqa'
    if retrieved_knowledge_path is not None:
        hparams.retrieved_knowledge_path = retrieved_knowledge_path
    if latent_ike_path is not None:
        hparams.latent_ike_path = latent_ike_path
    if semantic_encoder_path is not None:
        hparams.semantic_encoder_path = semantic_encoder_path
    file_path = data_path or './data/MMEdit/editing-data/vqa/vqa_eval.json'
    eval_ds = VQADataset(file_path, config=hparams)
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa'
    )
    if save_name:
        out_path = f'./logs/{get_filename(file_path)}/{save_name}.json'
    else:
        out_path = f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json'
    parse_result(metrics, out_path, config=hparams)


if __name__ == "__main__":
    hparams_dir = args.hparams_dir
    LOG.info(f"Task: {args.task}")
    LOG.info(f"Hyperparams: {hparams_dir}")
    LOG.info(f"Running on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.makedirs('./results/cache', exist_ok=True)

    if args.task == 'caption':
        edit_UNIKE_MiniGPT4_Caption(hparams_dir, data_path=args.data_path, save_name=args.save_name,
                                    retrieved_knowledge_path=args.retrieved_knowledge_path,
                                    latent_ike_path=args.latent_ike_path,
                                    semantic_encoder_path=args.semantic_encoder_path)
    elif args.task == 'vqa':
        edit_UNIKE_MiniGPT4_VQA(hparams_dir, data_path=args.data_path, save_name=args.save_name,
                                retrieved_knowledge_path=args.retrieved_knowledge_path,
                                latent_ike_path=args.latent_ike_path,
                                semantic_encoder_path=args.semantic_encoder_path)
