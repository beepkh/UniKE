from dataclasses import asdict
from datetime import datetime
from statistics import mean
import argparse
import os
import argparse
import json
import pickle


def str_to_int_list(s):
    try:
        return list(map(int, s.split(',')))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {s}")


def parse_result(metrics, output_path, config=None, ablation=False):
    import torch
    def metric_remove_tensor(metrics):
        outer_keys = ['pre', 'post']
        inner_keys = ['rewrite_acc', 'rephrase_acc', 'image_rephrase_acc', 'locality_acc', 'multimodal_locality_acc']
        for m in metrics:
            for ok in [k for k in outer_keys if k in m.keys()]:
                for ik in inner_keys:
                    if ik in m[ok] and isinstance(m[ok][ik], torch.Tensor):
                        m[ok][ik] = m[ok][ik].item()
        return metrics
    try:
        dir_path = os.path.dirname(output_path)
        os.makedirs(dir_path, exist_ok=True)
        rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
        rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
        rephrase_image_acc = mean([m['post']['image_rephrase_acc'].item() for m in metrics])
        locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
        locality_image_acc = mean([m['post']['multimodal_locality_acc'].item() for m in metrics])
        

  
        res_dict = {
            'rewrite_acc': rewrite_acc,
            'rephrase_acc': rephrase_acc,
            'rephrase_image_acc': rephrase_image_acc,
            'locality_acc': locality_acc,
            'locality_image_acc': locality_image_acc,
            'num_samples': len(metrics),
        }
        if ablation:
            res_dict['ablation'] = 1
        assert len(metrics) > 0, 'No metrics to parse'
        if config is not None:
            res_dict['config'] = asdict(config)
        res_dict['metrics'] = metric_remove_tensor(metrics)
        print(f"Saving results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(res_dict, f, indent=4)
        return res_dict
    except Exception as e:
        print(e)
        print(f'Failed to parse while output to{output_path}, trying to save metrics as a pickle obj')


def load_object(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_object(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def print_requires_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter '{name}' requires_grad=True")
            
def check_params_device(model):
    for name, param in model.named_parameters():
        if param.device == 'cpu':
            print(f"Parameter '{name}' is on CPU")    

def print_result(metrics):
    rewrite_acc = mean([m['post']['rewrite_acc'] for m in metrics])
    print(f'rewrite_acc: {rewrite_acc}')
    rephrase_acc = mean([m['post']['rephrase_acc'] for m in metrics])
    print(f'rephrase_acc: {rephrase_acc}')
    rephrase_image_acc = mean([m['post']['image_rephrase_acc'] for m in metrics])
    print(f'image_rephrase_acc: {rephrase_image_acc}')
    locality_acc = mean([m['post']['locality_acc'] for m in metrics])
    print(f'locality_acc: {locality_acc}')
    locality_image_acc = mean([m['post']['multimodal_locality_acc'] for m in metrics])
    print(f'multimodal_locality_acc: {locality_image_acc}')


def get_filename(path: str):
    return os.path.basename(path).split('/')[-1].split('.')[0]

def get_date():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
