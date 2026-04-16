from typing import Literal
import torch
import os
import torch.nn as nn
import copy
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import (
    BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .patch import ModifyLinearOutput, ModuleDetector, ModifyLinearInput, AdapterLayer, ModifyMLPLayer, PassThroughLayer
from ..dataset.zsre_dataloader import Seq2SeqData
from ..utils import label_smoothed_nll_loss, get_kl_diver_loss, patch_related_args
from .....trainer.utils import cu_del, dict_to


class Editor(nn.Module):

    def __init__(self,
                 model, hidden_size=768, device=None,
                 max_add_neuron_num=1,
                 activate_loss='non_use', memory_loss='non_use',
                 freeze_model=True, freeze_k=False, freeze_a=False,
                 memory_size=50000,
                 amplify_v=False, amplify_con=10.0,
                 drop_num=0, drop_rate=0.5,
                 act_margin_val=0.0, margin_val1=0.0, margin_val2=0.0,
                 hparams=None,
                 **kwargs
                 ):
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.device = device
        self.max_add_neuron_num = max_add_neuron_num
        self.memory_size = memory_size
        self.activate_loss = activate_loss
        self.memory_loss = memory_loss
        self.freeze_model = freeze_model
        self.freeze_a = freeze_a
        self.amplify_con = amplify_con
        if self.freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False

        # initialization, may be re-initialized after every edit
        self.model_named_modules = None
        self.get_named_modules()

        self.editors = []
        self.detectors = []

        self.train_memories = {}
        self.val_memories = {}

        self.amplify_v = amplify_v
        self.freeze_k = freeze_k

        self.drop_num = drop_num
        self.drop_rate = drop_rate

        self.act_margin_val = act_margin_val
        self.margin_val1 = margin_val1
        self.margin_val2 = margin_val2

        self.detected_modules={'llama_model.model.layers.31.mlp': 'up_proj'}
        
        self.tp_layers = hparams.tp_layers
        
        self.l_ike_layers = hparams.l_ike_layers
        
        self.layer_backup = {}
        
        self.inserted_layers = []
        
        self.add_neuron_num = 0

        self.hparams = hparams
        
        if not self.hparams.add_l_ike_layer:
            self.l_ike_layers = []
        
        self.latent_ike = None

        self.backup_edit_layers()
    
    def set_kv(self, retrieves_kv_path: str = None):
        assert os.path.exists(retrieves_kv_path), f"Retrieved knowledge path does not exist: {retrieves_kv_path}"
        self.kv = torch.load(retrieves_kv_path)
    
    def set_add_neuron_num(self, num):
        self.add_neuron_num = self.max_add_neuron_num if num is None else num

    
    def reset_model(self, model, clear_memory):
        self.model = copy.deepcopy(model)
        if self.freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False
        self.model_named_modules = None
        self.get_named_modules()
        self.editors = []
        for ms in [self.train_memories, self.val_memories]:
            for m in ms.values():
                del m[-1 * clear_memory:]

    def clear_memory(self):
        # clear all memories
        self.memories = {}

    def clear_detectors(self):
        for d in self.detectors:
            self.model_named_modules[d['module']]._modules[d['child']] = d['original_module'].to(self.device)
        self.detectors = []
        # self.get_named_modules()

    def clear_editors(self):
        for e in self.editors:
            self.model_named_modules[e['module']]._modules[e['child']] = e['original_module'].to(self.device)
        self.editors = []

    def set_latent_ike(self, path: str):
        assert os.path.exists(path), f"Latent IKE path does not exist: {path}"
        loaded = torch.load(path)
        if not isinstance(loaded, nn.Parameter):
            loaded = nn.Parameter(loaded)
        self.latent_ike = loaded
        
    
    def get_latent_ike(self):
        return self.latent_ike
    
    def set_editors(self, batch=None, init_weights=None, error_count=1, select_index=0, alpha=0.9):
        # before every turn, we call this function to set the editors
        self.get_editors(
            batch,
            init_weights=dict() if init_weights is None else init_weights,
            error_count=error_count, select_index=select_index,
            alpha=alpha
        )
        for e in self.editors:
            self.model_named_modules[e['module']]._modules[e['child']] = e['editor']

    def clean_cache(self):
        for l in self.inserted_layers:
            cu_del(l)
            del l
        self.inserted_layers = []
        torch.cuda.empty_cache()
    
    def set_detectors(self):
        for d in self.detectors:
            self.model_named_modules[d['module']]._modules[d['child']] = d['detector']

    def step(self):
        # we assign the trained linear layer to the edit target
        for e in self.editors:
            self.model_named_modules[e['module']]._modules[e['child']] = e['editor'].assign_layer().to(self.device)
        self.editors = []
        self.get_named_modules()

    def backup_edit_layers(self):
        tp_layers = self.tp_layers
        for name in self.l_ike_layers:
            n = name.rsplit('.', 1)
            self.layer_backup[(n[0], n[-1])] = copy.deepcopy(self.model_named_modules[n[0]].__getattr__(n[-1])).to(self.model.device)

        for name, edit_type in tp_layers.items():
            n = name.rsplit('.', 1)
            # ['model.model.decoder.layers.5', 'fc1']
            self.layer_backup[(n[0], n[-1])] = copy.deepcopy(self.model_named_modules[n[0]].__getattr__(n[-1])).to(self.model.device)
        
        
        
    def restore_edit(self):
        for k, v in self.layer_backup.items():
            self.model_named_modules[k[0]]._modules[k[1]] = v

    def restore_to_original(self):
        """Restore model to true original layers and clear all editing state."""
        self.restore_edit()
        self.editors = []
        for l in self.inserted_layers:
            cu_del(l)
            del l
        self.inserted_layers = []
        torch.cuda.empty_cache()

    def train_params(self, mode: str = 'unike'):
        for layer in self.inserted_layers:
            layer.train_mode(mode.lower())
    
    def lock_hidden_detectors(self):
        for d in self.detectors:
            d['detector'].turn_off_hidden()

    def unlock_hidden_detectors(self):
        for d in self.detectors:
            d['detector'].turn_on_hidden()

    def lock_memory_detectors(self):
        for d in self.detectors:
            d['detector'].turn_off_memory()

    def unlock_memory_detectors(self):
        for d in self.detectors:
            d['detector'].turn_on_memory()

    def insert_hidden_detector(self):
        self.get_detectors(
            detected_modules={'model.model.decoder.layers.5': 'fc1'},
            memory_loc='bart_seq', hidden_loc='bart_seq'
        )
        self.set_detectors()

    def get_hidden(self, index=None):
        res = dict()
        for d in self.detectors:
            k = d['module'] + '.' + d['child']
            v = d['detector'].get_hidden()
            res[k] = v[index] if index is not None else v
        return res

    def feed_one_memory(self, m):
        for k, v in m.items():
            assert k in self.train_memories
            self.train_memories[k] += [v]
            self.val_memories[k] += [v]
            print("This is a update and now we have {} train memories and {} val_memories for module {}".format(
                len(self.train_memories[k]), len(self.val_memories[k]), k
            ))

    def construct_memory(self, data: DataLoader, memory_size, device, update=False, memory_use='train'):
        self.detectors = []
        self.model.eval()
        self.model.to(device)
        self.get_detectors(
            detected_modules={'model.model.decoder.layers.5': 'fc1'},
            memory_loc='bart_seq', hidden_loc='bart_seq'
        )
        self.set_detectors()
        for d in self.detectors:
            if not update:
                d['detector'].turn_on_memory()
            else:
                d['detector'].turn_on_hidden()
        # bar = tqdm(enumerate(data), total=len(data.dataset) // data.batch_size)
        for _, batch in enumerate(data):
            input_ids = batch["src_input_ids"].to(device)
            attention_mask = batch["src_attention_mask"].to(device)
            decoder_input_ids = batch["trg_input_ids"].to(device)
            decoder_attention_mask = batch["trg_attention_mask"].to(device)
            for d in self.detectors:
                d['detector'].feed_memory_mask(decoder_attention_mask[:, :-1])
            self.model(
                input_ids, attention_mask,
                decoder_input_ids[:, :-1], decoder_attention_mask[:, :-1]
            )

        for d in self.detectors:
            name = d['module']+'.'+d['child']
            if not update:
                # this is for construction
                if memory_use == 'train':
                    self.train_memories[name] = d['detector'].get_memory()
                else:
                    self.val_memories[name] = d['detector'].get_memory()
            else:
                assert name in self.train_memories
                # print(len(list(torch.split(d['detector'].get_hidden(), 1))))
                self.train_memories[name] += list(torch.split(d['detector'].get_hidden(), 1))[1:-1]
                self.val_memories[name] += list(torch.split(d['detector'].get_hidden(), 1))[1:-1]
                print("This is a update and now we have {} train memories and {} val_memories".format(
                    len(self.train_memories[name]), len(self.val_memories[name])
                ))
            self.model_named_modules[d['module']]._modules[d['child']] = d['original_module']
        self.detectors = []
        # self.get_named_modules()

    def get_named_modules(self):
        # For now we just edit one linear layer once
        self.model_named_modules = None
        self.model_named_modules = {x[0]: x[1] for x in self.model.named_modules()}

    def get_editors(self, batch, init_weights=None, error_count=None, select_index=None, alpha=0.9):
        tp_layers = self.tp_layers
        for name, edit_type in tp_layers.items():
            e_tmp = dict()
            n = name.rsplit('.', 1)
            # ['model.model.decoder.layers.5', 'fc1']
            e_tmp['module'], e_tmp['child'] = n[0], n[-1]
            e_tmp['name'] = name
            if edit_type == 'input':
                inserted_layer = ModifyLinearInput(
                    self.model_named_modules[n[0]].__getattr__(n[-1]),
                    amplify=self.amplify_v, freeze_a=self.freeze_a,
                    amplify_con=self.amplify_con,
                    add_neuron_num=self.max_add_neuron_num if error_count is None else error_count,
                    hparams=self.hparams,
                    vec_avg=self.latent_ike
                )
                e_tmp['editor'] = inserted_layer
                self.inserted_layers.append(inserted_layer)
                e_tmp['original_module'] = self.model_named_modules[n[0]].__getattr__(n[-1])
                self.editors.append(e_tmp)
                
            elif edit_type == 'output' and (self.hparams.model_name == 'minigpt4' or name not in self.l_ike_layers):
                init_weight = init_weights[name] if name in init_weights.keys() else None
                train_memo, val_memo = None, None
                if name in self.train_memories.keys():
                    train_memo = self.train_memories[name]
                    val_memo = self.val_memories[name]
                inserted_layer = ModifyLinearOutput(
                    self.model_named_modules[n[0]].__getattr__(n[-1]),
                    init_weight=init_weight,  freeze=self.freeze_k,
                    activate_loss=self.activate_loss, memory_loss=self.memory_loss,
                    train_memories=train_memo, val_memories=val_memo,
                    drop_num=self.drop_num, drop_rate=self.drop_rate,
                    act_loc=0 if select_index is None else select_index,
                    add_neuron_num=self.max_add_neuron_num if error_count is None else error_count,
                    hparams=self.hparams,
                    vec_avg=self.latent_ike
                )
            
                e_tmp['editor'] = inserted_layer
                self.inserted_layers.append(inserted_layer)
                e_tmp['original_module'] = self.model_named_modules[n[0]].__getattr__(n[-1])
                self.editors.append(e_tmp)
        
        ### Latent incontext Editing
        if self.hparams.add_l_ike_layer:
            for name in self.l_ike_layers:
                e_tmp = dict()
                n = name.rsplit('.', 1)
                e_tmp['module'], e_tmp['child'] = n[0], n[-1]
                if self.hparams.model_name == 'minigpt4':
                    encoder_path = getattr(self.hparams, 'semantic_encoder_path', None)
                    if not encoder_path:
                        encoder_path = './cache/minigpt4/semantic_encoder.pt'
                    inserted_layer = ModifyMLPLayer(
                        mlp=self.model_named_modules[n[0]].__getattr__(n[-1]),
                        kv=self.kv,
                        device=self.device, hparams=self.hparams,
                        alpha=alpha,
                        encoder_path=encoder_path
                    ).to(self.device)
                else:
                    raise NotImplementedError(f"current code is for minigpt4")
                    
                e_tmp['editor'] = inserted_layer
                self.inserted_layers.append(inserted_layer)
                e_tmp['original_module'] = self.model_named_modules[n[0]].__getattr__(n[-1])
                self.editors.append(e_tmp)

    def get_detectors(self, *args, **kwargs):
        detected_modules = kwargs.get("detected_modules")
        memory_loc = kwargs.get("memory_loc") if "memory_loc" in kwargs else 0
        hidden_loc = kwargs.get("hidden_loc") if "hidden_loc" in kwargs else 0
        mode = kwargs.get("mode") if "mode" in kwargs else "input"
        for module_name, child in detected_modules.items():
            detector = ModuleDetector(
                model=self.model_named_modules[module_name]._modules[child],
                memory_size=self.memory_size, mode=mode,
                memory_loc=memory_loc, hidden_loc=hidden_loc
            )
            self.detectors.append({
                'module': module_name, 'child': child,
                'detector': detector, 'original_module': self.model_named_modules[module_name]._modules[child]
            })

    def get_act_loss(self):
        act_loss = 0
        for e in self.editors:
            if isinstance(e['editor'], ModifyLinearOutput):
                act_loss_ = e['editor'].get_act_loss()
                if act_loss_ is not None:
                    act_loss += act_loss_
        return act_loss

    def get_memo_loss(self):
        memo_loss = 0
        for e in self.editors:
            if isinstance(e['editor'], ModifyLinearOutput):
                memo_loss_ = e['editor'].get_memo_loss()
                if memo_loss_ is not None:
                    memo_loss += memo_loss_
        return memo_loss

    def repeat_tensor(self, t):
        return torch.repeat_interleave(t, self.drop_num + 1, dim=0)

    def feed_kl_input(self, memo_loader, his_edit_data, total_loc_num):
        self.memo_loader = memo_loader
        self.total_loc_num = total_loc_num
        self.his_edit_data = his_edit_data

    def do_not_act_val(self):
        for e in self.editors:
            e['editor'].activate_loss = 'non_use'

    def do_act_val(self):
        for e in self.editors:
            e['editor'].activate_loss = self.activate_loss

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        target_for_loss = decoder_input_ids[:, 1:]
        res = dict()

        logits = self.model(
            input_ids, attention_mask,
            decoder_input_ids[:, :-1], decoder_attention_mask[:, :-1]
        )
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=logits.log_softmax(-1), target=target_for_loss,
            epsilon=self.model.hparams.eps, ignore_index=self.model.tokenizer.pad_token_id,
        )
        ntokens = decoder_attention_mask[:, 1:].sum()
        loss, nll_loss = loss / ntokens, nll_loss / ntokens
        res['logtis'] = logits
        res['loss'] = loss
        if self.activate_loss != 'non_use':
            res['act_loss'] = self.get_act_loss()
        if self.memory_loss != 'non_use':
            if self.memory_loss.startswith('kl'):
                print(f"kl loss is not supported yed due to memory limits")
            else:
                res['memo_loss'] = self.get_memo_loss()
        return res
