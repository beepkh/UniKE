import time
import sys
import higher
import torch
import torch.nn as nn
from .editable_model import EditableModel
from higher.patch import monkeypatch as make_functional
from ..losses import kl_loc_loss
from ..utils import _inner_params, _logits
from easyeditor.trainer.blip2_models.mini_gpt4 import MiniGPTOutput
sys.path.append('../../..')
def count_error_nums(model, data_point, device=None):
    device = next(model.parameters()).device
    with torch.no_grad():
        model.eval()
        model.to(device)
        batch = data_point

        # model.unlock_hidden_detectors()
        # since the original batch may be repeated
        output: MiniGPTOutput = model(batch)
        pred = torch.argmax(output.logits, dim=-1)
        # we count how many tokens are wrongly generated under the teacher-forcing setting and add one patch for each of them.
        trg = batch["labels"][[0],:].to(device)
        concat_len = min(trg.shape[1], pred.shape[1])
        # cut the sequence to the same length
        trg = trg[:, :concat_len]
        pred = pred[:, :concat_len]
        select_index = [i for i, s in enumerate(list((pred != trg).cpu().squeeze())) if s]

    return len(select_index), select_index



class TP(EditableModel):

    def __init__(self, model, config, model_constructor, edit_loss_fn=None):
        super().__init__(model, config, model_constructor)

        if edit_loss_fn is not None:
            self.edit_loss_fn = edit_loss_fn

        self.locality_loss_fn = kl_loc_loss
        self.loc_ids = None
        self.loc_masks = None
        self.loc_sampler = None
        # from ...models.tpatcher.src import Editor
        # self.editor = Editor(
        #     model=model,
        #     max_add_neuron_num=self.config.max_add_neuron_num,
        #     freeze_model=self.config.freeze_model, freeze_k=self.config.freeze_k, freeze_a=self.config.freeze_a,
        #     memory_size=self.config.memory_size, memory_loss=self.config.memory_loss,
        #     amplify_v=self.config.amplify_v, activate_loss=self.config.activate_loss,
        #     act_margin_val=self.config.act_margin_val, margin_val1=self.config.margin_val1,
        #     margin_val2=self.config.margin_val2, device=model.device
        # )
        # self.editor.backup_edit_layers()

    def _edit_loss(self, model, p0, p_edited, edit_batch):
        output = _logits(model(**edit_batch, params=p_edited))
        loss_dict = self.edit_loss_fn(output, edit_batch["labels"])
        l_edit, acc = loss_dict["nll"], loss_dict["acc"]
        if self.config.ft.locality.enabled:
            if self.config.ft.locality.oracle:
                loc_batch = next(self.loc_sampler)["loc"]
            else:
                raise NotImplementedError

            with torch.no_grad():
                original_base_logits = _logits(model(**loc_batch, params=p0))
            edited_base_logits = _logits(model(**loc_batch, params=p_edited))
            kl_mask = loc_batch.get(
                "decoder_attention_mask", loc_batch["attention_mask"]
            )
            l_loc = self.locality_loss_fn(
                original_base_logits, edited_base_logits, mask=kl_mask
            )
            loss = l_loc + self.config.ft.locality.cedit * l_edit
        else:
            l_loc = torch.tensor(float("nan"))
            loss = l_edit
        return loss, l_edit, l_loc, acc

    def accuracy(self, output, labels):
        if output.shape[-1] != 1:
            shifted_output = output.argmax(-1)[:, :-1]
            shifted_labels = labels[:, 1:]
            to_predict = (shifted_labels != -100).sum()
            correct = (shifted_output == shifted_labels).sum()
            acc = correct.float() / to_predict.float()
        else:
            acc = ((output > 0) == labels.bool()).sum().float()
        return acc

    def _edit_status(self, step, loss, l_edit, l_loc, acc, res_p):
        return (
            f"step: {step}".ljust(14)
            + f"loss: {loss.item():.5f}".ljust(18)
            + f"l_edit: {l_edit.item():.5f}".ljust(18)
            + f"l_loc: {l_loc.item():.5f}".ljust(18)
            + f"acc: {acc.item():.2f}".ljust(14)
            + f"norm: {res_p.view(-1).norm().item():.5f}"
        )

    def edit(self, batch, condition=None, detach_history=False):
        return None
        edit_model = self.model
        # 1. find which part of the model to edit
        # edit_is_not_suc, aer, re_num = edit_or_not_seq2seq(self.model, data_point=batch, device=args.device)
        # print("successfully create edit model")
        init_weights = None
        error_count, select_index = 0, []
        error_count, select_index = count_error_nums(edit_model, batch)
        # print("successfully count error nums")
        # from easyeditor.models.tpatcher.src import Editor
        add_neuron_num = error_count

        need_edit = True
        if need_edit:
            self.editor.set_editors(init_weights=init_weights, error_count=error_count, select_index=select_index)
            # print("successfully create editor")
            # for _ in range(add_neuron_num):
                # editor.set_editors()
        
        
        # 2. implement the train procedure
        max_epochs = self.config.max_epochs
        # configure optimizer
        OptClass = getattr(torch.optim, self.config.opt)
        opt = OptClass(edit_model.parameters(), lr=self.config.edit_lr)
        # optimizer = torch.optim.Adam(edit_model.parameters(), lr=1e-3)
        # print("successfully create optimizer and ready to start training")
        for epoch in range(max_epochs):
            # forward
            outputs = _logits(edit_model(batch))
            # outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
            torch.nn.utils.clip_grad_norm_(edit_model.parameters(), max_norm=self.config.max_norm)
            # print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item()}")
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        # if not editor.has_stepped:
        #     editor.editor.step()
        self.editor.restore_edit()
        torch.cuda.empty_cache()
        # return the duplicated model
        return (
            TP(edit_model, self.config, self.model_constructor, self.edit_loss_fn),
            {},
        )

