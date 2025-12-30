#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle
from accelerate import dispatch_model
import json
import torch.nn.functional as F
import math

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 
from component.cal_r import seg_W_SVD, rank_loop, direct_svd
from component.cal_r_low_rank import seg_W_SVD_v2
from component.seg_svd_llama import Seg_SVD_LlamaAttention, Seg_SVD_LlamaMLP
from component.local_llama import Local_LlamaAttention, Local_LlamaMLP

import sys
tqdm.write = lambda msg: sys.stderr.write(msg + '\n')

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

@torch.no_grad()
def register_accumulated_cosine_hooks(model, model_name):
    """
    在前向传播过程中累计每层余弦相似度的平均值
    返回包含 running average 的结构
    """
    cosine_stats = {}  # {i: {'sum': ..., 'count': ...}}
    handles = []

    def make_hook(i):
        def hook(module, input, output):
            x_in = input[0]
            x_out = output[0]

            x_out_flat = x_out.view(-1, x_out.shape[-1])
            x_in_flat = x_in.view(-1, x_in.shape[-1]).to(x_out_flat.device)
            
            cos = F.cosine_similarity(x_in_flat, x_out_flat, dim=-1)
            avg_cos = cos.mean().item()

            if i not in cosine_stats:
                cosine_stats[i] = {'sum': 0.0, 'count': 0}
            cosine_stats[i]['sum'] += avg_cos
            cosine_stats[i]['count'] += 1
        return hook
    if 'opt' in model_name:
        for i, layer in enumerate(model.model.decoder.layers):
            h = layer.register_forward_hook(make_hook(i))
            handles.append(h)
    else:
        for i, layer in enumerate(model.model.layers):
            h = layer.register_forward_hook(make_hook(i))
            handles.append(h)

    return cosine_stats, handles

@torch.no_grad()
def calib_layer_cos_similarity(model, calib_loader, args):
    model_id = model.config._name_or_path.split('/')[-1]
    cache_file = f"cache/{model_id.replace('/','_')}_layer_cos_similarity_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
    if os.path.exists(cache_file):
        avg_cos_sim = torch.load(cache_file, map_location="cpu")
        s_tensor = torch.tensor([1-avg_cos_sim[i] for i in avg_cos_sim])
        weights = torch.softmax(-s_tensor / args.t, dim=0)
        layer_ratio = len(avg_cos_sim)*(1 - args.ratio)*weights
        return 1 - layer_ratio

    model.eval()
    cosine_stats, handles = register_accumulated_cosine_hooks(model, args.model)

    for batch in calib_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            model(**batch)

    for h in handles:
        h.remove()
    
    avg_cos_sim = {
        i: stat['sum'] / stat['count']
        for i, stat in cosine_stats.items()
        }
    torch.save(avg_cos_sim, cache_file)
    s_tensor = torch.tensor([1-avg_cos_sim[i] for i in avg_cos_sim])
    weights = torch.softmax(-s_tensor / args.t, dim=0)
    layer_ratio = len(avg_cos_sim)*(1 - args.ratio)*weights
    return 1 - layer_ratio

@torch.no_grad()
def calib_layer_sensitivity_ppl(model, profiling_mat, calib_loader, args):
    model_id = model.config._name_or_path.split('/')[-1]
    cache_file = f"cache/{model_id.replace('/','_')}_layer_sensitivity_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
    if os.path.exists(cache_file):
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict

    model.eval()

    sensitivity_dict = {}
    param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    ppl_base = evaluate_perplexity(model, input_ids, args.whitening_nsamples)
    layers = model.model.layers
    pbar = tqdm(total=len(param_ratio_candidates) * len(layers))
    for i in range(len(layers)):
        layer = layers[i]
        sensitivity_dict[i] = {}
        sensitivity_dict[i][1] = ppl_base
        subset = find_layers(layer)
        original_weights = {}
        scaling_diag_matrix_dict = {}
        for name in subset:
            original_weights[name] = subset[name].weight.data.clone()
            scaling_diag_matrix_dict[name] = profiling_mat[i][name].to(subset[name].weight.device)
        for param_ratio in param_ratio_candidates:
            for name in subset:
                W = subset[name].weight.data.float()
                dtype = subset[name].weight.data.dtype
                scaling_diag_matrix = scaling_diag_matrix_dict[name]
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    print("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()
                W_scale = torch.matmul(W, scaling_diag_matrix)
                U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * param_ratio / (W.shape[0] + W.shape[1]))
                truc_s = S[:num_s_after_trunc]
                truc_u = U[:, :num_s_after_trunc]
                truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)
                #### Replace Attn, MLP ####
                sqrtSigma = torch.sqrt(truc_sigma)
                svd_u = torch.matmul(truc_u, sqrtSigma)
                svd_v = torch.matmul(sqrtSigma, truc_v)
                subset[name].weight.data = (svd_u@svd_v).to(dtype)
            ppl = evaluate_perplexity(model, input_ids, args.whitening_nsamples)
            sensitivity_dict[i][param_ratio] = ppl
            pbar.update(1)
                # print(f"{info['full_name']} {param_ratio} {ppl}")
            for name in subset:
                subset[name].weight.data = original_weights[name]
        # setattr(info["father"], info["name"], raw_linear)
        
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict

@torch.no_grad()
def calib_sensitivity_product(model, profiling_mat, calib_loader, args):
    model_id = model.config._name_or_path.split('/')[-1]
    cache_file = f"cache/{model_id.replace('/','_')}_product_sensitivity_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
    if os.path.exists(cache_file):
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    
    model.eval()
    if profiling_mat is None:
        profiling_mat = torch.load(args.h_mat_path)
    sensitivity_dict = {}
    param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if 'opt' in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    pbar = tqdm(total=len(find_layers(layers[0])) * len(param_ratio_candidates) * len(layers))
    for i in range(len(layers)):
        layer = layers[i]
        sensitivity_dict[i] = {}
        subset = find_layers(layer)
        for name in subset:
            sensitivity_dict[i][name] = {}
            W = subset[name].weight.data.float()
            dtype = subset[name].weight.data.dtype
            # scaling_diag_matrix = profiling_mat[i][name].to(W.device)
            raw_scaling_diag_matrix = profiling_mat[i][name].to(W.device)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                # eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                # raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                damp = 0.05 * torch.mean(torch.diag(raw_scaling_diag_matrix))
                n = raw_scaling_diag_matrix.size(0)
                idx = torch.arange(n, device=raw_scaling_diag_matrix.device)
                raw_scaling_diag_matrix[idx, idx] += damp
                # raw_scaling_diag_matrix += 1e-2 * torch.diag_embed(raw_scaling_diag_matrix).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = raw_scaling_diag_matrix = None
                del eigenvalues
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            # U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            S = torch.linalg.svdvals(W_scale)
            for param_ratio in param_ratio_candidates:
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * param_ratio / (W.shape[0] + W.shape[1]))
                product = torch.sum(S**2).item()
                delta_product = torch.sum(S[num_s_after_trunc:]**2).item()
                sensitivity_dict[i][name][param_ratio] = {'product': product, 'delta_product': delta_product}
                pbar.update(1)
    del profiling_mat
    torch.cuda.empty_cache()
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict

@torch.no_grad()
def calib_sensitivity_ppl(model, profiling_mat, calib_loader, args):
    model_id = model.config._name_or_path.split('/')[-1]
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
    if os.path.exists(cache_file):
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    
    # cache_file = f'cache/Llama-7b_sensitivity_abs_mean_0.5_32_wikitext2_seed233.pt'
    # if os.path.exists(cache_file):
    #     sensitivity_dict = torch.load(cache_file, map_location="cpu")
    #     flat_sens_dict = {}
    #     for k, v in sensitivity_dict.items():
    #         k_split = k.split('.')
    #         layer = int(k_split[2])
    #         if layer not in flat_sens_dict:
    #             flat_sens_dict[layer] = {}
    #         flat_sens_dict[layer][f'{k_split[-2]}.{k_split[-1]}'] = v
    #     return flat_sens_dict
    model.eval()

    sensitivity_dict = {}
    param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    ppl_base = evaluate_perplexity(model, input_ids, args.whitening_nsamples)
    layers = model.model.layers
    pbar = tqdm(total=len(find_layers(layers[0])) * len(param_ratio_candidates) * len(layers))
    for i in range(len(layers)):
        layer = layers[i]
        sensitivity_dict[i] = {}
        subset = find_layers(layer)
        for name in subset:
            sensitivity_dict[i][name] = {}
            sensitivity_dict[i][name][1] = ppl_base
            W = subset[name].weight.data.float()
            dtype = subset[name].weight.data.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(W.device)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            for param_ratio in param_ratio_candidates:
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * param_ratio / (W.shape[0] + W.shape[1]))
                truc_s = S[:num_s_after_trunc]
                truc_u = U[:, :num_s_after_trunc]
                truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)
                #### Replace Attn, MLP ####
                sqrtSigma = torch.sqrt(truc_sigma)
                svd_u = torch.matmul(truc_u, sqrtSigma)
                svd_v = torch.matmul(sqrtSigma, truc_v)
                subset[name].weight.data = (svd_u@svd_v).to(dtype)
                ppl = evaluate_perplexity(model, input_ids, args.whitening_nsamples)
                sensitivity_dict[i][name][param_ratio] = ppl
                pbar.update(1)
                # print(f"{info['full_name']} {param_ratio} {ppl}")
            subset[name].weight.data = W.to(dtype)
        # setattr(info["father"], info["name"], raw_linear)
        
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict

@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    # model = model.cpu()
    h_mat = {}
    for i in range(len(layers)):
        layer_h = {}
        subset = find_layers(layers[i])
        for name in subset:
            layer_h[name] = subset[name].raw_scaling_diag_matrix.cpu()
            del subset[name].raw_scaling_diag_matrix
        h_mat[i] = layer_h
    torch.cuda.empty_cache()
    return h_mat
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat

@torch.no_grad()
def get_h(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    # model = model.cpu()
    h_mat = {}
    for i in range(len(layers)):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            layer_profile[name] = subset[name].raw_scaling_diag_matrix.cpu()
            del subset[name].raw_scaling_diag_matrix
        h_mat[i] = layer_profile
        torch.cuda.empty_cache()
    return h_mat

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        for name, module in model.named_modules():
            if 'rotary_emb' in name:
                module.inv_freq = module.inv_freq.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                if kwargs['attention_mask'] is not None:
                    cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                if kwargs['attention_mask'] is not None:
                    cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            if 'Llama' in model_name:
                if "position_ids" not in batch:
                    input_ids = batch["input_ids"]
                    position_ids = torch.arange(
                        0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
                    ).unsqueeze(0).expand_as(input_ids)
                    batch["position_ids"] = position_ids
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}
    h_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer_h = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                if attention_masks is not None:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
                else:
                    # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=None, position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=None, position_ids=position_ids.to(dev))[0]
            else:
                if attention_masks is not None: 
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=None)[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            layer_h[name] = subset[name].scaling_diag_matrix.cpu()
            subset[name].scaling_diag_matrix = None
            del subset[name].scaling_diag_matrix
            # subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        # for name in subset:
        #     layer_h[name] = subset[name].scaling_diag_matrix.to(dev)
        #     raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
        #     try:
        #         scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
        #     except Exception as e:
        #         print("Warning: eigen scaling_diag_matrix is not positive!")
        #         eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
        #         raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
        #         scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
        #         eigenvalues = None
        #         del eigenvalues
        #     layer_profile[name] = scaling_diag_matrix.cpu()
        #     scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
        #     del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
        #     torch.cuda.empty_cache()
        h_mat[i] = layer_h
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return h_mat
    # return profiling_mat
     
@torch.no_grad()
def tranfer_llama(model):
    model.eval()

    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        svd_attn = Local_LlamaAttention(config=model.config)
        svd_mlp = Local_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act)
        for name in subset:
            W = subset[name].weight.data.clone()
            subset[name].weight = None
            del subset[name].weight
            if "q_proj" in name:
                svd_attn.q_proj.weight.data = W
            elif "k_proj" in name:
                svd_attn.k_proj.weight.data = W
            elif "v_proj" in name:
                svd_attn.v_proj.weight.data = W
            elif "o_proj" in name:
                svd_attn.o_proj.weight.data = W 
                layer.self_attn =  svd_attn
            elif "gate_proj" in name:
                svd_mlp.gate_proj.weight.data = W
            elif "down_proj" in name:
                svd_mlp.down_proj.weight.data = W
            elif "up_proj" in name:
                svd_mlp.up_proj.weight.data = W
                layer.mlp = svd_mlp
        torch.cuda.empty_cache()
    
            

@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        #### Replace Attn, MLP ####
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def calculate_capped_module_sparsities(
    layer_r: float,
    layer_data: dict,
    module_params: dict,
    sparsity_cap: float = 0.8
):
    del_ratio = 1 - layer_r
    layer_sensitivity = {}
    for name, module_data in layer_data.items():
        param_ratios = list(module_data.keys())

        # Find the param_ratio in our candidates that is closest to the target_ratio
        closest_ratio = min(param_ratios, key=lambda p: abs(p - del_ratio))

        # Get the corresponding product and delta_product
        product_data = module_data[closest_ratio]
        product = product_data.get('product', 0)
        delta_product = product_data.get('delta_product', 0)

        # Calculate the sensitivity and handle division by zero
        if product > 0:
            sensitivity = delta_product / product
        else:
            sensitivity = 0
        
        layer_sensitivity[name] = sensitivity
    
    sorted_modules = sorted(
        layer_sensitivity.items(),
        key=lambda item: item[1],  # Sort based on the value (sensitivity score)
        reverse=True              # True means descending order (highest first)
    )
    importance_scores = {}
    for rank, (name, sensitivity) in enumerate(sorted_modules):
        # 3. Calculate importance using the specified formula.
        #    Using 20.0 to ensure float division.
        division = 1 + 10*math.tan(math.pi*del_ratio/2)
        importance = math.exp(rank / division)
        
        importance_scores[name] = importance
    # module_params = {name: module.weight.numel() for name, module in layer_modules.items()}
    layer_total_params = sum(module_params.values())
    layer_prune_params = layer_total_params * del_ratio

    pruning_weights = {}
    for name, importance in importance_scores.items():
        if name in module_params:
            # Weight is proportional to size and "unimportance"
            pruning_weights[name] = module_params[name] * importance
            
    total_weight = sum(pruning_weights.values())

    # Step 2 & 3: Allocate pruning budget and calculate module sparsity
    module_sparsities = {}
    # print("\n--- Module Sparsity Allocation Details ---")
    for name, weight in pruning_weights.items():
        # Allocate parameters to prune for this module
        params_to_prune = layer_prune_params * (weight / total_weight)
        
        # Calculate individual module sparsity
        sparsity = params_to_prune / module_params[name]
        # sparsity = sparsity if sparsity < sparsity_cap else sparsity_cap
        module_sparsities[name] = 1 - sparsity
        print(f"Module: {name:<10} | Importance: {importance_scores[name]:.4f} | Sparsity: {sparsity:.4f}")

    # Sanity Check
    calculated_total_pruned = sum(module_params[name] * (1-sp) for name, sp in module_sparsities.items())
    # print(f"\nTarget params to prune in layer: {layer_prune_params:,.0f}")
    # print(f"Calculated params to prune in layer: {calculated_total_pruned:,.0f}")
    
    return module_sparsities

@torch.no_grad()
def whitening_raw_format(model_name, model, profiling_mat, layer_ratio, sensitivity_dict, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    

    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        
        
        if isinstance(layer_ratio, torch.Tensor):
            ratio = layer_ratio[i]
            if sensitivity_dict is not None:
                params_map = {}
                for name in subset:
                    params_map[name] = subset[name].weight.data.numel()
                module_ratio = calculate_capped_module_sparsities(ratio, sensitivity_dict[i], params_map)
        for name in subset:
            W = subset[name].weight.data.float()
            dtype = subset[name].weight.data.dtype
            # scaling_diag_matrix = profiling_mat[i][name].to(W.device)
            raw_scaling_diag_matrix = profiling_mat[i][name].to(W.device)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                # eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                # raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                damp = 0.05 * torch.mean(torch.diag(raw_scaling_diag_matrix))
                n = raw_scaling_diag_matrix.size(0)
                idx = torch.arange(n, device=raw_scaling_diag_matrix.device)
                raw_scaling_diag_matrix[idx, idx] += damp
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = raw_scaling_diag_matrix = None
                del eigenvalues
                
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)

            if isinstance(layer_ratio, torch.Tensor):
                if sensitivity_dict is not None:
                    ratio = module_ratio[name]
            else:
                ratio = layer_ratio
            if ratio < 1:
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
                truc_s = S[:num_s_after_trunc]
                truc_u = U[:, :num_s_after_trunc]
                truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)
                #### Replace Attn, MLP ####
                sqrtSigma = torch.sqrt(truc_sigma)
                svd_u = torch.matmul(truc_u, sqrtSigma)
                svd_v = torch.matmul(sqrtSigma, truc_v)
                subset[name].weight.data = (svd_u@svd_v).to(dtype)
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma =truc_sigma=svd_u= svd_v=None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma, truc_sigma, svd_u, svd_v
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def trend_collect(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    layers = model.model.layers
    # for i in tqdm(range(len(layers))):
    layer = layers[0]
    subset = find_layers(layer)
    res = {}
    for name in subset:
        W = subset[name].weight.data.float().to(dev)
        dtype = W.dtype
        H = profiling_mat[0][name].to(dev)
        res[name] = rank_loop(W, H, ratio)
    return res

@torch.no_grad()
def seg_whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    error_metrics = {}
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        svd_u_dict = {}
        svd_v_dict = {}
        col_w_dict = {}
        rank_dict = {}
        col_indices_dict = {}
        svd_indices_dict = {}
        error_metrics[f"layer_{i}"] = {}
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            H = profiling_mat[i][name].to(dev)
            truc_wu, truc_wv, col_w, rank, drop_indices, keep_indices, sigma_error_reduction, svd_error_reduction = seg_W_SVD(W, H, ratio)
            module_name = name.split('.')[-1]
            (
                svd_u_dict[module_name], 
                svd_v_dict[module_name],
                col_w_dict[module_name], 
                rank_dict[module_name], 
                col_indices_dict[module_name], 
                svd_indices_dict[module_name]
            ) = (
                truc_wu, 
                truc_wv,
                col_w, 
                rank, 
                drop_indices, 
                keep_indices
            )
            error_metrics[f"layer_{i}"][name] = {
                "sigma_error_reduction": sigma_error_reduction.item(),
                "svd_error_reduction": svd_error_reduction.item()
            }
        if "Llama" in model_name or "vicuna" in model_name:
            svd_attn = Seg_SVD_LlamaAttention(config=model.config, rank=rank_dict, col_indices=col_indices_dict, svd_indices=svd_indices_dict)
            svd_mlp = Seg_SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, 
                                       rank=rank_dict, col_indices=col_indices_dict, svd_indices=svd_indices_dict)
        for name in subset:
            module_name = name.split('.')[-1]
            svd_u = svd_u_dict[module_name].cpu()
            svd_v = svd_v_dict[module_name].cpu()
            col_w = col_w_dict[module_name].cpu()
            if "q_proj" in name:
                svd_attn.q_u_proj.weight.data = svd_u
                svd_attn.q_v_proj.weight.data = svd_v
                svd_attn.q_col.weight.data = col_w
            elif "k_proj" in name:
                svd_attn.k_u_proj.weight.data = svd_u
                svd_attn.k_v_proj.weight.data = svd_v
                svd_attn.k_col.weight.data = col_w
            elif "v_proj" in name:
                svd_attn.v_u_proj.weight.data = svd_u
                svd_attn.v_v_proj.weight.data = svd_v
                svd_attn.v_col.weight.data = col_w
            elif "o_proj" in name:
                svd_attn.o_u_proj.weight.data = svd_u
                svd_attn.o_v_proj.weight.data = svd_v
                svd_attn.o_col.weight.data = col_w
                layer.self_attn =  svd_attn
            elif "gate_proj" in name:
                svd_mlp.gate_u_proj.weight.data = svd_u
                svd_mlp.gate_v_proj.weight.data = svd_v
                svd_mlp.gate_col.weight.data = col_w
            elif "down_proj" in name:
                svd_mlp.down_u_proj.weight.data = svd_u
                svd_mlp.down_v_proj.weight.data = svd_v
                svd_mlp.down_col.weight.data = col_w
            elif "up_proj" in name:
                svd_mlp.up_u_proj.weight.data = svd_u
                svd_mlp.up_v_proj.weight.data = svd_v
                svd_mlp.up_col.weight.data = col_w
                layer.mlp = svd_mlp
        torch.cuda.empty_cache()
    save_dir = 'error_metrics'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = model_name.split('/')[-1] + f'_{ratio}_' + 'error_metrics.json'
    with open(os.path.join(save_dir, save_name), "w") as f:
        json.dump(error_metrics, f, indent=4)


@torch.no_grad()
def seg_whitening_raw_format(model_name, model, profiling_mat, ratio, sensitivity_dict, args):
    model.eval()
    ratio_target = args.ratio
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    error_metrics = {}
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        error_metrics[f"layer_{i}"] = {}

        if isinstance(layer_ratio, torch.Tensor):
            ratio = layer_ratio[i]
            if sensitivity_dict is not None:
                params_map = {}
                for name in subset:
                    params_map[name] = subset[name].weight.data.numel()
                module_ratio = calculate_capped_module_sparsities(ratio, sensitivity_dict[i], params_map)

        for name in subset:
            W = subset[name].weight.data.float().clone()
            dtype = subset[name].weight.dtype
            H = profiling_mat[i][name].to(W.device)
            if isinstance(layer_ratio, torch.Tensor):
                if sensitivity_dict is not None:
                    ratio = module_ratio[name]
            else:
                ratio = layer_ratio
            if ratio < 1:
                if '30b' in model_name and i > 3 and i < 45:
                    if 'q_proj' in name or 'k_proj' in name or 'gate_proj' in name or 'up_gate' in name:
                        svd_w = direct_svd(W, H, ratio)
                        subset[name].weight.data = svd_w.to(dtype)
                else:
                    truc_wu, truc_wv, col_w, rank, col_indices, svd_indices, sigma_error_reduction, svd_error_reduction = seg_W_SVD(W, H, ratio)
                    svd_w = truc_wu@truc_wv
                    W[:,svd_indices] = svd_w
                    W[:,col_indices] = col_w
                    subset[name].weight.data = W.to(dtype)
                    error_metrics[f"layer_{i}"][name] = {
                        "sigma_error_reduction": sigma_error_reduction.item(),
                        "svd_error_reduction": svd_error_reduction.item()
                    }
            # w_new, svd_error_reduction = seg_W_SVD_low_rank(W, H, ratio)
            # subset[name].weight.data = w_new.to(dtype)
            # error_metrics[f"layer_{i}"][name] = {
            #     "svd_error_reduction": svd_error_reduction.item()
            # }
    save_dir = 'error_metrics'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = model_name.split('/')[-1] + f'_{ratio_target}_' + 'error_metrics.json'
    with open(os.path.join(save_dir, save_name), "w") as f:
        json.dump(error_metrics, f, indent=4)

@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio, name=name, direct_update=direct_update)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
        layer = layer.to(dev)
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # trucation SVD
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"truncted error: {self.error}")
        x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        self.updated_uT = torch.linalg.lstsq(x,outs).solution
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='/cluster/home/xulin/programs/models/Llama-13b', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default='Llama_7b_svdllm_0.8.pt', help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true',default=False, help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    
    # parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--h_mat_path', type=str, default=None, help='Local path to load the h matrices`')
    parser.add_argument('--seed',type=int, default=3, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=-1, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    parser.add_argument('--eval_ppl', action='store_true', default=True, help='whether to eval the model before saving')
    parser.add_argument('--eval_zero_shot', action='store_true', default=True, help='whether to eval the model before saving')
    parser.add_argument('--t', type=float, default=0.2, help='temperature for layer compression ratio')
    parser.add_argument('--matrices_optimized', action='store_true', default=True, help='whether to optimize matrices compression ratio')
    parser.add_argument('--trunc_rank_method', type=str, default='cos', choices=['average','cos'], help='way to cal compression ratio of each module')
    parser.add_argument('--save_path', type=str, default='profiles', help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--data_preparation', action='store_true', default=False)
    parser.add_argument('--cuda_devices', type=str, default='0,1', help='the cuda devices to run the model')
    
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    # device_map = 'cpu' if args.run_low_resource else 'auto'
    args.ratio = 1- args.ratio
    # step = -2: 将SVD-LLM的结果用标准形式保存 -1： 将ours的结果用标准形式保存 0：将ours的结果用
    if args.step == -5:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='auto')
        model = model.eval()
        h_mat = torch.load(args.h_mat_path)
        res = trend_collect(args.model, model, h_mat, args.ratio, args.DEV)
        torch.save(res, 'loss_w_c.pt')
    elif args.step == -4:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='cpu')
        model = model.eval()
        profiling_mat = None
        cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len, seed=args.seed)
        ave_com_sim = calib_layer_cos_similarity(model, cali_white_data, args)
        print(ave_com_sim)
        exit()
        s_tensor = torch.tensor([1-ave_com_sim[i] for i in ave_com_sim])
        t = 0.1
        weights = torch.softmax(-s_tensor / t, dim=0)
        layer_ratio = len(ave_com_sim)*(1 - args.ratio)*weights
    if args.step == -3:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='auto')
        model = model.eval()
        cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len, seed=args.seed)
        if args.profiling_mat_path is None:
            if args.run_low_resource:
                profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            else:
                profiling_mat = profle_svdllm(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.split('/')[-1] + '_svdllm_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        calib_sensitivity_product(model, profiling_mat, cali_white_data, args)

    elif args.step == -2:
        if args.run_low_resource:
            model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='cpu')
        else:
            model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='auto')
        # model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map=device_map)
        model = model.eval()
        if args.h_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len, seed=args.seed)
            if args.run_low_resource:
                profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            else:
                profiling_mat = profle_svdllm(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.split('/')[-1] + '_svdllm_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
                if args.data_preparation:
                    exit()
        else:
            profiling_mat = torch.load(args.h_mat_path)
        if args.trunc_rank_method == 'average':
            layer_ratio = args.ratio
            sensitivity_dict = None
        else:
            if args.h_mat_path is not None:
                cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len, seed=args.seed)

            layer_ratio = calib_layer_cos_similarity(model, cali_white_data, args)
            print('layer_ratio', layer_ratio)
            if args.matrices_optimized:
                sensitivity_dict = calib_sensitivity_product(model, profiling_mat, cali_white_data, args)
            else:
                sensitivity_dict = None
            # layer_ratio = compute_compression_ratios(sensitivity_dict, args.ratio)
            # if args.trunc_rank_method == 'binary':
            #     layer_ratio = binary_search_truncation_rank(model, sensitivity_dict, args)
            # elif args.trunc_rank_method == 'dp':
            #     if args.module_respective:
            #         layer_ratio = dp_module_truncation_rank(model, sensitivity_dict, args)
            #     else:
            #         layer_ratio = dp_truncation_rank(model, sensitivity_dict, args)
            
        whitening_raw_format(args.model, model, profiling_mat, layer_ratio, sensitivity_dict, args.DEV)
        if args.eval_ppl:
            ppl_eval(model, tokenizer, datasets=['wikitext2', 'c4'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        if args.eval_zero_shot:
            accelerate=False if torch.cuda.device_count() == 1 else True
            task_list = ["hellaswag","winogrande", "arc_easy", "openbookqa", "piqa", "mathqa"]
            num_shot = 0
            results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
            print("zero_shot evaluation results")
            print(results) 
        if args.save_path is not None:
            args.save_path = args.save_path + '/' + args.model.split('/')[-1] +'_svdllm_' + args.trunc_rank_method + f'_matrices_optimized{args.matrices_optimized}_' + str(args.ratio)
            model.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)
    if args.step == -1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='auto')
        model = model.eval()
        if args.h_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            h_mat = get_h(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(h_mat, args.save_path + "/" + args.model.split('/')[-1] + '_ours_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            h_mat = torch.load(args.h_mat_path)
        # model.to('cpu')
        if args.trunc_rank_method == 'average':
            layer_ratio = args.ratio
            sensitivity_dict = None
        else:
            if args.h_mat_path is not None:
                cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len, seed=args.seed)

            layer_ratio = calib_layer_cos_similarity(model, cali_white_data, args)
            print('layer_ratio', layer_ratio)
            if args.matrices_optimized:
                profiling_mat = None
                sensitivity_dict = calib_sensitivity_product(model, profiling_mat, cali_white_data, args)
            else:
                sensitivity_dict = None
        seg_whitening_raw_format(args.model, model, h_mat, layer_ratio, sensitivity_dict, args)
        # pickle.dumps(model)
        if args.eval_ppl:
            ppl_eval(model, tokenizer, datasets=['wikitext2', 'c4'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        if args.eval_zero_shot:
            accelerate=False if torch.cuda.device_count() == 1 else True
            task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa", "mathqa", "piqa"]
            num_shot = 0
            results = zeroshot_eval(model, tokenizer, task_list)
            print("zero_shot evaluation results")
            print(results)
        if args.save_path is not None:
            args.save_path = args.save_path + '/' + args.model.split('/')[-1] +'_ours_' + args.trunc_rank_method +f'_matrices_optimized{args.matrices_optimized}_' + str(args.ratio)
            model.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)
    elif args.step == 0:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map=None)
        model = model.eval()
        if args.h_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            h_mat = get_h(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(h_mat, args.save_path + "/" + args.model.split('/')[-1] + '_ours_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            h_mat = torch.load(args.h_mat_path)
        # model.to('cpu')
        seg_whitening(args.model, model, h_mat, args.ratio, args.DEV)
        # pickle.dumps(model)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.split('/')[-1] +'_ours_' + str(args.ratio) + '.pt')
    elif args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='auto')
        model = model.eval()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            if args.run_low_resource:
                profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            else:
                profiling_mat = profle_svdllm(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_svdllm_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        model = model.cpu()
        whitening(args.model, model, profiling_mat, args.ratio, args.DEV)
        # pickle.dumps(model)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_svdllm_' + str(args.ratio) + '.pt')   # fp32
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()  # need to set to float
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.step == 5:
            model, tokenizer = get_model_from_local(args.model_path)
            model = model.to(torch.float16)
            model = model.to('cuda')
            # model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_huggingface(args.model_path, device_map='auto')
            # model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.eval()
        # model = model.float()
        # model = model.to(args.DEV)
        # model = dispatch_model(model, device_map="auto")
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['c4'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 6:
            if torch.cuda.device_count() == 1:
                accelerate=False
            else:
                accelerate=True
            # task_list = ["hellaswag","winogrande", "arc_easy", "openbookqa", "piqa", "mathqa"]
            task_list = ["mathqa"]
            num_shot = 0
            results = zeroshot_eval(model, tokenizer, task_list)
            print("zero_shot evaluation results")
            print(results)
