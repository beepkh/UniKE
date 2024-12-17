import os
import argparse
from utils import *
parser = argparse.ArgumentParser(description='train args')
parser.add_argument('--hparams_dir', type=str, default='./hparams/TPATCHER/minigpt4.yaml')
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
from easyeditor import CaptionDataset, VQADataset, CapVQADataset
from easyeditor.dataset.processor.base_dataset import BaseDataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, UniKEHyperParams
from easyeditor.util.hparams import HyperParams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import pickle
from utils import *
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


def edit_TP_MiniGPT4_Caption(hparam_path):
    hparams = UniKEHyperParams.from_hparams(hparam_path)
    hparams.task_name = 'caption'
    file_path = './data/caption/caption_eval_edit.json'
    eval_ds = CaptionDataset(file_path, config=hparams)
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='caption'
    )
    # parse_result(metrics, f'./logs/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams)
    parse_result(metrics, f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams, ablation=args.ablation)

def edit_TP_MiniGPT4_VQA(hparam_path):
    hparams = UniKEHyperParams.from_hparams(hparam_path)
    hparams.task_name = 'vqa'
    file_path = './data/vqa/vqa_eval.json'
    eval_ds = VQADataset(file_path, config=hparams)
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa' 
    )
    parse_result(metrics, f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams, ablation=args.ablation)

def edit_TP_MiniGPT4_Mixed(hparam_path):
    hparams = UniKEHyperParams.from_hparams(hparam_path)
    eval_ds = CapVQADataset(cap_dir='./data/caption/caption_eval_edit.json', 
                            vqa_dir='./data/vqa/vqa_eval.json', config=hparams) 
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True        
    )
    parse_result(metrics, f'./logs/cross/{get_filename(hparam_path)}_{get_date()}.json', config=hparams, ablation=args.ablation)
    

    
if __name__ == "__main__":
    from easyeditor.models.unike.unike_main import get_alpha_pth
    hparams_dir = args.hparams_dir
    LOG.info(f"Task: {args.task}")
    LOG.info(f"Hyperparams: {hparams_dir}")
    LOG.info(f"Running on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    if args.task == 'all':
        edit_TP_MiniGPT4_Caption(hparams_dir)
        edit_TP_MiniGPT4_VQA(hparams_dir)
    elif args.task == 'caption':
        edit_TP_MiniGPT4_Caption(hparams_dir)
    elif args.task == 'vqa':
        edit_TP_MiniGPT4_VQA(hparams_dir)
    elif args.task == 'mixed':
        edit_TP_MiniGPT4_Mixed(hparams_dir)
    