from dataclasses import dataclass
from typing import *
import sys

from ...util.hparams import HyperParams

import yaml


@dataclass
class UniKEHyperParams(HyperParams):
    # Multimodal
    qformer_name_or_path: str
    state_dict_file: str

    # Image_dir
    coco_image: str
    rephrase_image: str

    # Model
    name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str

    tp_layers: Dict[str, str]
    l_ike_layers: List[str]

    alg_name: str
    edit_lr: float
    seed: int
    opt_class: str

    results_dir: str

    device: str
    max_norm: float

    batch_size: int = 1
    max_length: int = 30
    max_epochs: Optional[int] = 10
    model_parallel: bool = False
    qformer_checkpoint: Optional[str] = None
    freeze_qformer: bool = True
    pretrained_ckpt: Optional[str] = None

    max_add_neuron_num: Optional[int] = 30
    freeze_model: Optional[bool] = True
    freeze_k: Optional[int] = 0
    freeze_a: Optional[int] = 0
    memory_size: Optional[int] = 40000
    memory_loss: Optional[str] = 'non_use'
    amplify_v: Optional[int] = 1
    activate_loss: Optional[str] = 'non_use'
    act_margin_val: Optional[float] = 0.0
    margin_val1: Optional[int] = 3
    margin_val2: Optional[int] = 3
    hyperparams_nn_count: Optional[int] = 30

    continuous: Optional[bool] = True
    # Continuous editing if continuous_sample>1, else single-turn editing
    continuous_sample: Optional[int] = 1

    add_l_ike_layer: Optional[bool] = True
    do_clip_norm: Optional[bool] = True

    ike: Optional[bool] = False
    task_name: Optional[str] = None

    prob_use_result: Optional[float] = 0.3

    retrieved_knowledge_path: Optional[str] = None        # dir containing ike_{sample_id}.pth
    latent_ike_path: Optional[str] = None                # path to l-ike.pth
    semantic_encoder_path: Optional[str] = None          # path to semantic_encoder.pt


    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'UNIKE') or print(f'UniKEHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
