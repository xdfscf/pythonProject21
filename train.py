import math
import os
import random

import pandas as pd
import torch
from torchvision.transforms import transforms
from transformers import BertTokenizerFast
from dataloader import Music_Dataset
from torch.utils.data import DataLoader
from vilt_configure import VILTConfigure
from myvilt import ViltForMaskedLM, ViltModel, ViltForPrediction
import copy
import inspect
from dataclasses import dataclass
from contextlib import nullcontext
# learning rate decay scheduler (cosine with warmup)
def get_lr(it, hyper):
    # 1) linear warmup for warmup_iters steps
    if it < hyper.warmup_iters:
        return hyper.learning_rate * it / hyper.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > hyper.lr_decay_iters:
        return hyper.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - hyper.warmup_iters) / (hyper.lr_decay_iters - hyper.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return hyper.min_lr + coeff * (hyper.learning_rate - hyper.min_lr)

def configure_optimizers(model, hyper):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': hyper.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and hyper.device == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=hyper.learning_rate, betas=(hyper.beta1, hyper.beta2), **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

def replace_mlm_tokens(token_ids, candidate_pred_positions, mlm_pred, config):

    masked_train_tokens=[]
    for i in range(len(token_ids)):
        token_ids_with_mask = copy.deepcopy(token_ids[i])
        random.shuffle(candidate_pred_positions[i])
        masked_pos = 0
        max_pred = max(1, round(len(candidate_pred_positions[i]) * mlm_pred))
        for mlm_pred_position in candidate_pred_positions[i]:
            if masked_pos >= max_pred:
                break
            masked_token = None
            if random.random() < 0.8:
                masked_token = 103
            else:

                if random.random() < 0.5:
                    masked_token = token_ids[i][mlm_pred_position]
                else:
                    masked_token = random.randint(104, config.bert_vocab_size)
            token_ids_with_mask[mlm_pred_position] = masked_token
            masked_pos += 1
        masked_train_tokens.append(token_ids_with_mask.unsqueeze(0))
    return masked_train_tokens

def data_tokenize(train_data, config):
    token_ids = train_data["input_ids"]
    values_to_exclude = torch.tensor([101, 102, 0])
    candidate_pred_positions = [torch.nonzero(~torch.isin(torch.tensor(input_ids), values_to_exclude)).squeeze().tolist() for input_ids in token_ids]
    mlm_pred = 0.15
    masked_train_tokens = replace_mlm_tokens(token_ids, candidate_pred_positions, mlm_pred, config)

    return train_data, masked_train_tokens

@dataclass
class HyperParameters:
    # set target device
    device = 'cuda'

    # optimizer
    learning_rate = 6e-4
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95

    # train and evaluation
    batch_size = 3
    eval_iters = 200
    eval_interval = 100
    max_iters = 600000

    # lr scheduler
    decay_lr = True
    warmup_iters = 2000
    lr_decay_iters = 600000
    min_lr = 6e-5

    always_save_checkpoint = True
    eval_only = False
    eval_on = False

    # number of epochs that update the gradient, used to simulate a larger batch,
    # by calculating the average of loss
    gradient_accumulation_steps = 5 * 8

    # grad scalar
    grad_clip = 1.0
    log_interval = 1

    float_cast = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

@torch.no_grad()
def estimate_loss(model, hyper, eval_file_path, img_path, dataloader):
    out = {}
    model.eval()
    test_dataloader = dataloader(eval_file_path, img_path)

    losses = torch.zeros(hyper.eval_iters)
    for k in range(hyper.eval_iters):
        X, Y = next(iter(test_dataloader))
        with hyper.float_cast:
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    loss = losses.mean()

    model.train()
    return out


def read_labels_from_csv(file_path):
    df=pd.read_csv(file_path)
    num_rows = len(df)
    labels=set()
    for i in range(num_rows):
        labels.add(df.iloc[i, 2])
    return labels


train_data_path = "./train.csv"
test_data_path="./test.csv"
img_path="./mel_spec"
out_dir="./out"

def load_model(vilt_configure):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cude')

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line

    model = ViltModel(vilt_configure)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model

def pretraining_with_mlm(base_model):
    hyper = HyperParameters()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    vilt_configure = VILTConfigure()
    vilt_configure.image_size = (128, 646)
    vilt_configure.img_channels = 1
    model_name=ViltForMaskedLM
    '''
    vilt_model = load_model(vilt_configure)
    '''
    vilt_for_mask = model_name(config=vilt_configure)
    '''
    vilt_for_mask.vilt=vilt_model
    '''
    vilt_for_mask.vilt = base_model
    vilt_for_mask.to(hyper.device)


    _dataloader=Music_Dataset
    training_data =   _dataloader(train_data_path , img_path)
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(True))

    optimizer = configure_optimizers(vilt_for_mask, hyper)

    iter_num=0
    best_val_loss=0
    while True:
        lr = get_lr(iter_num, hyper)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % hyper.eval_interval == 0 and hyper.eval_on:
            losses = estimate_loss(vilt_for_mask, hyper, test_data_path, img_path,  _dataloader)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': vilt_for_mask.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and hyper.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(hyper.gradient_accumulation_steps):
            with hyper.float_cast:
                train_data, train_labels = next(iter(train_dataloader))

                train_img = train_data['image']
                train_text = train_data['text']

                train_data = tokenizer(
                    train_text,
                    return_tensors='pt',
                    padding="max_length",
                    truncation=True,
                    max_length=vilt_configure.bert_max_position_embeddings
                )


                train_data, masked_train_tokens = data_tokenize(train_data, vilt_configure)

                paras = train_data
                paras['pixel_values'] = train_img
                paras['labels'] = torch.cat(masked_train_tokens, dim=0)

                paras = {key: torch.tensor(value).to(hyper.device) for key, value in paras.items()}

                loss, logits, hidden_states, attentions = vilt_for_mask(**paras)

                loss = loss / hyper.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward(retain_graph=True)
        # clip the gradient
        if hyper.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vilt_for_mask.parameters(), hyper.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        print(f"iter {iter_num}: loss {loss:.4f}%")
        iter_num += 1

        # termination conditions
        if iter_num > hyper.max_iters:
            break

def pretraining_with_label_prediction(base_model):
    hyper = HyperParameters()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    labels = read_labels_from_csv(train_data_path)
    label_to_index = {label: index for index, label in enumerate(labels)}
    num_labels = len(labels)

    vilt_configure = VILTConfigure()
    vilt_configure.image_size = (128, 646)
    vilt_configure.img_channels = 1
    vilt_configure.label_size=num_labels

    model_name = ViltForPrediction
    '''
    vilt_model = load_model(vilt_configure)
    '''
    vilt_for_prediction = model_name(config=vilt_configure)
    '''
    vilt_for_mask.vilt=vilt_model
    '''
    vilt_for_prediction.vilt=base_model
    vilt_for_prediction.to(hyper.device)

    _dataloader = Music_Dataset
    training_data = _dataloader(train_data_path, img_path)
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(True))

    optimizer = configure_optimizers(vilt_for_prediction, hyper)

    iter_num = 0
    best_val_loss = 0
    while True:
        lr = get_lr(iter_num, hyper)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % hyper.eval_interval == 0 and hyper.eval_on:
            losses = estimate_loss(vilt_for_prediction, hyper, test_data_path, img_path, _dataloader)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': vilt_for_prediction.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and hyper.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(hyper.gradient_accumulation_steps):
            with hyper.float_cast:
                train_data, train_labels = next(iter(train_dataloader))

                train_labels = [label_to_index[i] for i in train_labels]

                train_img = train_data['image']
                train_text = train_data['text']

                train_data = tokenizer(
                    train_text,
                    return_tensors='pt',
                    padding="max_length",
                    truncation=True,
                    max_length=vilt_configure.bert_max_position_embeddings
                )

                paras = train_data
                paras['pixel_values'] = train_img
                paras['labels'] = train_labels

                paras = {key: torch.tensor(value).to(hyper.device) for key, value in paras.items()}

                loss, logits, hidden_states, attentions = vilt_for_prediction(**paras)

                loss = loss / hyper.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward(retain_graph=True)
        # clip the gradient
        if hyper.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vilt_for_prediction.parameters(), hyper.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        print(f"iter {iter_num}: loss {loss:.4f}%")
        iter_num += 1

        # termination conditions
        if iter_num > hyper.max_iters:
            break


if __name__ == "__main__":
    vilt_configure = VILTConfigure()
    vilt_configure.image_size = (128, 646)
    vilt_configure.img_channels = 1
    vilt=ViltModel(config=vilt_configure)
    pretraining_with_mlm(vilt)
    pretraining_with_label_prediction(vilt)