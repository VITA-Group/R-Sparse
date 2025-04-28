import os
import json
import tqdm
import torch
import random
import datasets
import numpy as np 

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from models.modeling_llama import LlamaForCausalLM_R_Sparse, R_Sparse_Linear

__all__ = ['setup_config', 'setup_model']

def setup_model(args):
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=args.cache_dir)
    if args.method == 'full':
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                config=config, cache_dir=args.cache_dir, device_map='auto', trust_remote_code=True)
    elif args.method == 'relufiction':
        config.hidden_act = 'relu'
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                config=config, cache_dir=args.cache_dir, device_map='auto', trust_remote_code=True)
    elif args.method == 'r_sparse':
        config = setup_config(config, args)
        model = LlamaForCausalLM_R_Sparse.from_pretrained(args.model_name, 
                config=config, cache_dir=args.cache_dir, device_map='auto', trust_remote_code=True)
        model = set_threshold_r_sparse(model, config, args, R_Sparse_Linear, tokenizer)
    else:
        raise NotImplementedError
    return config, tokenizer, model



def set_threshold_r_sparse(model, config, args, module_type, tokenizer):
    if args.sparse_config_file is not None:
        sparse_config_file = np.loadtxt(args.sparse_config_file)
        index = 0
        for module_name, module in model.named_modules():
            if isinstance(module, module_type):
                if 'self_attn' in module_name:
                    in_channel = config.hidden_size
                    out_channel = config.hidden_size
                else:
                    if 'down_proj' in module_name:
                        in_channel = config.intermediate_size
                        out_channel = config.hidden_size
                    else:
                        in_channel = config.hidden_size
                        out_channel = config.intermediate_size

                alpha = sparse_config_file[index]
                s = sparse_config_file[index + 1]
                index += 2
                module.flag_getting_threshold = True
                module.target_sparsity = 1 - (1 - s) * alpha
                module.sparse_ratio = alpha

                channels = max(int(in_channel * (1 - s) * alpha), 1)
                overall_budget = in_channel * out_channel * (1 - s)
                sparse_budget = channels * out_channel
                low_rank_budget = overall_budget - sparse_budget
                module.rank = max(int(low_rank_budget / (in_channel + out_channel)), 1)
    else:
        for module_name, module in model.named_modules():
            if isinstance(module, module_type):
                if 'self_attn' in module_name:
                    in_channel = config.hidden_size
                    out_channel = config.hidden_size
                else:
                    if 'down_proj' in module_name:
                        in_channel = config.intermediate_size
                        out_channel = config.hidden_size
                    else:
                        in_channel = config.hidden_size
                        out_channel = config.intermediate_size

                module.flag_getting_threshold = True
                module.target_sparsity = 1 - (1 - args.target_sparsity) * args.sparse_ratio
                module.sparse_ratio = args.sparse_ratio

                channels = int(in_channel * (1 - args.target_sparsity) * args.sparse_ratio)
                overall_budget = in_channel * out_channel * (1 - args.target_sparsity)
                sparse_budget = channels * out_channel
                low_rank_budget = overall_budget - sparse_budget
                module.rank = int(low_rank_budget / (in_channel + out_channel))
    model._load_low_rank_module(config)

    # getting dataset
    print('Estimating threshold...')
    model = model.cuda()
    dataloader = get_wikitext2(nsamples=1, seed=42, seqlen=512, tokenizer=tokenizer)
    with torch.no_grad():
        inputs = torch.cat(dataloader, dim=0).cuda()
    lm_logits = model(input_ids=inputs).logits

    for module_name, module in model.named_modules():
        if isinstance(module, module_type):
            module.prefill_ratio = args.prefill_ratio
            if module.sparse_ratio == 1:
                module.mode = 'sparse'
            elif module.sparse_ratio == 0:
                module.mode = 'low_rank'
            else:
                module.mode = 'r_sparse'
    return model

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    return trainloader

def setup_config(config, args):

    with open(args.config_file) as f:
        config_data = json.load(f)

    config.q_threshold = config_data['q_threshold']
    config.q_svd_path = config_data['q_svd_path']
    config.q_low_rank = config_data['q_low_rank']

    config.k_threshold = config_data['k_threshold']
    config.k_svd_path = config_data['k_svd_path']
    config.k_low_rank = config_data['k_low_rank']

    config.v_threshold = config_data['v_threshold']
    config.v_svd_path = config_data['v_svd_path']
    config.v_low_rank = config_data['v_low_rank']

    config.o_threshold = config_data['o_threshold']
    config.o_svd_path = config_data['o_svd_path']
    config.o_low_rank = config_data['o_low_rank']

    config.gate_threshold = config_data['gate_threshold']
    config.gate_svd_path = config_data['gate_svd_path']
    config.gate_low_rank = config_data['gate_low_rank']

    config.up_threshold = config_data['up_threshold']
    config.up_svd_path = config_data['up_svd_path']
    config.up_low_rank = config_data['up_low_rank']

    config.down_threshold = config_data['down_threshold']
    config.down_svd_path = config_data['down_svd_path']
    config.down_low_rank = config_data['down_low_rank']

    return config


