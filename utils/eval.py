import os 
import tqdm
import time
import torch
import random
import datasets
import argparse
import numpy as np
import torch.nn as nn

__all__ = ['get_c4', 'eval_model']

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = datasets.load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    random.seed(seed)
    trainloader = []

    for _ in tqdm.tqdm(range(nsamples)):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    return trainloader

def eval_model(model, dataloader, args):
    num_batches = len(dataloader) // args.batch_size
    nlls = []
    eval_tokens = 0
    for i in range(num_batches):
        batch = dataloader[i*args.batch_size:(i+1)*args.batch_size]
        inputs = torch.cat([b for b in batch], dim=0).cuda() # (bs, seqlen, dim)
        with torch.no_grad():
            lm_logits = model(input_ids=inputs).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            nlls.append(loss.float() * shift_labels.numel())
            eval_tokens += shift_labels.numel()
    ppl = torch.exp(torch.tensor(nlls).sum() / eval_tokens).item()
    if np.isnan(ppl):
        ppl = 1e9
    return ppl

