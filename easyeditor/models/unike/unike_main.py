from typing import Any, Dict, List, Optional, Tuple, Union
import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from easyeditor.trainer.blip2_models import MiniGPTOutput

from .unike_hparams import UniKEHyperParams
from .src.models.seq2seq_modules import Editor
from ...trainer.utils import dict_to, cu_del
import gc


CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_unike_to_model_mm(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: UniKEHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    model, weights_copy = execute_unike(model, tok, requests, hparams, **kwargs)


    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy

def _logits(x):
    return x if not hasattr(x, "logits") else x.logits


def construct_mm_samples(
    model,
    model_name,
    hparams: UniKEHyperParams,
    tok: AutoTokenizer,
    record: Dict,
    device
) -> Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    from ...evaluate import prepare_multimodal_edit, compute_multimodal_edit_results_demo
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    device = model.device
    # edit_inner = dict_to(edit_inner, device)
    ret['rewrite_sample'] = edit_inner
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_sample'] = edit_outer
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_sample'] = edit_image_outer

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        ret['locality_output_sample'] = locality
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        ret['multimodal_locality_output_sample'] = m_locality

    torch.cuda.empty_cache()
    gc.collect()
    
    return ret


def print_trainable_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Trainable params: ", name)


def _small_init_editing_params(editor, seed: int = 42, std: float = 1e-3):
    """Small-initialize the editing parameters to avoid disrupting the original model."""
    from .src.models.patch import ModifyLinearOutput, ModifyLinearInput, ModifyMLPLayer

    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.random.get_rng_state() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    def _init_single_module(mod):
        if isinstance(mod, ModifyLinearOutput):
            nn.init.normal_(mod.extra_output.weight, std=std)
            nn.init.zeros_(mod.extra_output.bias)
            nn.init.normal_(mod.encoder_linear.weight, std=std)
            nn.init.zeros_(mod.encoder_linear.bias)
        elif isinstance(mod, ModifyLinearInput):
            mod.extra_input.data.normal_(std=std)
            nn.init.normal_(mod.encoder_linear.weight, std=std)
            nn.init.zeros_(mod.encoder_linear.bias)

    for e_item in editor.editors:
        mod = e_item["editor"]
        if isinstance(mod, ModifyMLPLayer):
            for sub in mod.mlp.modules():
                _init_single_module(sub)
        else:
            _init_single_module(mod)

    torch.random.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.random.set_rng_state(cuda_rng_state)


def execute_unike(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: UniKEHyperParams,
    **kwargs
    ) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    # Monkey-patch AdapterLayer.forward to support per-layer kv [m, L, d]
    from .src.models.patch import AdapterLayer
    if not getattr(AdapterLayer, '_patched_per_layer_kv', False):
        _orig_forward = AdapterLayer.forward
        def _adapter_forward_per_layer(self, x):
            kv = self.kv
            layer_idx = getattr(self, 'layer_idx', None)
            if layer_idx is not None and kv.dim() >= 3:
                if kv.dim() == 3:    # [m, L, d] → [m, d]
                    kv = kv[:, layer_idx, :]
                elif kv.dim() == 4:  # [B, m, L, d] → [B, m, d]
                    kv = kv[:, :, layer_idx, :]
            # Now kv is [m, d] or [B, m, d], use original logic
            if kv.dim() == 3:
                attn_weights = torch.bmm(x, kv.transpose(1, 2))
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_output = torch.bmm(attn_weights, kv)
            else:
                attn_weights = torch.matmul(x, kv.T)
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, kv)
            alpha = F.cosine_similarity(
                self.semantic_encoder(attn_output), x, dim=-1
            ).unsqueeze(-1)
            self.alpha = alpha
            x_norm = x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            attn_norm = attn_output.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            attn_output_scaled = attn_output * (x_norm / attn_norm)
            return (1.0 - alpha) * x + alpha * attn_output_scaled
        AdapterLayer.forward = _adapter_forward_per_layer
        AdapterLayer._patched_per_layer_kv = True

    # get editor from kwargs
    editor: Editor = kwargs.get("editor", None)
    if editor is None:
        raise ValueError("Editor is not provided")
    collate_fn = kwargs.get("collate_fn", None)
    if collate_fn is None:
        raise ValueError("Collate function is not provided")
    pre = kwargs.get("pre", None)
    if pre is None:
        raise ValueError("Pre is not provided")
    task = kwargs.get("task", None)
    from ...trainer.losses import masked_log_probs
    # assert the model in editor is the same as model
    assert id(model) == id(editor.model), "Model in editor is not the same as the model passed in"
    samples = construct_mm_samples(model, hparams.model_name, hparams, tok, requests[0], model.device)
    batch_samples = dict_to(samples, model.device) # send to device
    error_count, select_index, init_weights = 0, [], None
    
    error_count = hparams.hyperparams_nn_count
    editor.set_add_neuron_num(error_count)
    # hidden_size = model.llama_model.config.hidden_size
            
    max_epochs = hparams.max_epochs
    # configure optimizer
    opt_class = getattr(torch.optim, hparams.opt_class)
    
    ### External Knowledge Resorting
    sample_id = kwargs.get("sample_id", -1)
    print(f"[Edit] Start editing sample {sample_id}")
    print(f"[Edit] Loading external knowledge for sample {sample_id}...")
    assert hparams.retrieved_knowledge_path, "retrieved_knowledge_path must be specified"
    cur_retrieved_knowledge_tensor_path = os.path.join(hparams.retrieved_knowledge_path, f"ike_{sample_id}.pth")
    editor.set_kv(cur_retrieved_knowledge_tensor_path)

    assert hparams.latent_ike_path, "latent_ike_path must be specified"
    print(f"[Edit] Loading intrinsic knowledge for sample {sample_id}...")
    editor.set_latent_ike(hparams.latent_ike_path)
    editor.latent_ike.requires_grad_(False)
    editor.set_editors(init_weights=init_weights, error_count=error_count, select_index=select_index)

    from .src.models.patch import ModifyLinearOutput, ModifyLinearInput, ModifyMLPLayer
    tp_editors = {}  # name -> editor module
    mlp_editors = {} # mlp_name -> ModifyMLPLayer module
    for e_item in editor.editors:
        mod = e_item["editor"]
        name = e_item.get("name", "")
        if isinstance(mod, (ModifyLinearOutput, ModifyLinearInput)):
            tp_editors[name] = mod
        elif isinstance(mod, ModifyMLPLayer):
            mlp_name = e_item['module'] + '.' + e_item['child']
            mlp_editors[mlp_name] = mod

    for tp_name, tp_mod in tp_editors.items():
        parts = tp_name.rsplit('.', 1)
        mlp_name, child_name = parts[0], parts[1]
        if mlp_name in mlp_editors:
            mlp_mod = mlp_editors[mlp_name]
            # Replace the plain linear inside ModifyMLPLayer.mlp with the tp_layer editor
            mlp_mod.mlp._modules[child_name] = tp_mod


    ### Intrinsic Knowledge Editing
    editor.train_params("unike")

    layer_idx = 0
    for mlp_name, mlp_mod in mlp_editors.items():
        mlp_mod.adapt.layer_idx = layer_idx
        layer_idx += 1

    for e_item in editor.editors:
        mod = e_item["editor"]
        if isinstance(mod, ModifyMLPLayer):
            mod.adapt.kv.requires_grad_(False)
            for p in mod.adapt.semantic_encoder.parameters():
                p.requires_grad_(False)
            for sub in mod.mlp.modules():
                if isinstance(sub, (ModifyLinearOutput, ModifyLinearInput)):
                    for p in sub.linear.parameters():
                        p.requires_grad_(False)

    # Small-init the editing parameters
    _small_init_editing_params(editor, seed=hparams.seed)

    params = [p for n,p in model.named_parameters() if p.requires_grad]
    if params and max_epochs > 0:
        print(f"[Edit] Start updating intrinsic knowledge with {max_epochs} steps (lr={hparams.edit_lr})...")
        opt = opt_class(params, lr=hparams.edit_lr)
        batch = batch_samples["rewrite_sample"]
        for e in range(max_epochs):
            out = model(batch)
            loss = out.loss
            loss.backward()
            if (e % 5 == 0 and e>0) or e == max_epochs - 1:
                print(f"Step {e}, loss: {loss.item():.4f}")
            if hparams.do_clip_norm:
                torch.nn.utils.clip_grad_norm_(params, max_norm=hparams.max_norm)
            opt.step()
            opt.zero_grad()
        print(f"[Edit] Intrinsic knowledge optimization ({max_epochs} steps) completed for sample {sample_id}")
    elif max_epochs > 0:
        print("[INFO] No editable parameters found, skipping online editing optimization.")

    return model, {}

  
if __name__ == "__main__":
    pass