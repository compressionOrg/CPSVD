import torch
import numpy as np
from tqdm import tqdm
import time
import itertools
from utils.data_utils import get_test_data
import os
import sys
import fnmatch
import torch.nn as nn

from lm_eval.base import BaseLM
from lm_eval import evaluator

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

class EvalLM(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        device="cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self._device = self.model.device 

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.seqlen = 2048

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :self.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False)



def _get_model_device(model):
    """获取模型的设备，兼容不同模型类型"""
    if hasattr(model, 'device'):
        return model.device
    # 对于使用 device_map='auto' 的模型，尝试从第一个参数获取设备
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate_perplexity(model, dataset, limit):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()
    device = _get_model_device(model)

    nlls = []
    valid_samples = 0

    for i in range(nsamples):
        if i == limit:
            break
        input_ids = dataset[i : i + 1, :-1].to(device)
        labels = dataset[i : i + 1, 1:].contiguous()
        logits = model(input_ids=input_ids)[0]
        
        # 检查 logits 是否包含 NaN 或 Inf
        if not torch.isfinite(logits).all():
            print(f"Warning: Non-finite logits at sample {i}, skipping...")
            continue
            
        shift_logits = logits[:, :, :]
        shift_labels = labels.to(device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        # 检查 loss 是否有效
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss at sample {i}, skipping...")
            continue
            
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
        valid_samples += 1
    
    if len(nlls) == 0:
        print("Warning: No valid samples for perplexity calculation!")
        return float('inf')
    
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    return ppl.item()


@torch.no_grad()
def ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device="cuda"):
    # model.to(device)
    model.eval()
    ppls = {}
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size = batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            output = model(batch, use_cache=False)
            lm_logits = output.logits
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
    print("PPL after pruning: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

# only call this function when for 65b or more model    
@torch.no_grad()
def ppl_eval_large(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], seq_len=2048, batch_size=32, device="cuda"):
    import  torch.nn as nn
    class LlamaRMSNorm(nn.Module):
        def __init__(self, hidden_size=model.config.hidden_size, eps=model.config.rms_norm_eps):
            """
            LlamaRMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
    norm = LlamaRMSNorm().half().cuda()
    lm_head = model.lm_head.cuda()
    model.eval()
    ppls = {}
    layers = model.model.layers
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            model.model.embed_tokens = model.model.embed_tokens.cuda()
            model.model.norm = model.model.norm.cuda()
            layers[0] = layers[0].cuda()

            dtype = next(iter(model.parameters())).dtype
            inps = torch.zeros(
                (batch.shape[0], model.seqlen, model.config.hidden_size), dtype=dtype, device="cuda"
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
                        cache['position_ids'] = kwargs['position_ids']
                    else:
                        cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                        cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
                    raise ValueError
            layers[0] = Catcher(layers[0])
            for j in range(batch.shape[0]):
                try:
                    model(batch[j].unsqueeze(0).cuda())
                except ValueError:
                    pass
            layers[0] = layers[0].module
            layers[0] = layers[0].cpu()
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            model.model.norm = model.model.norm.cpu()
            torch.cuda.empty_cache()
            attention_masks = cache['attention_mask']
            position_ids = cache['position_ids']
            for i in range(len(layers)):
                layer = layers[i].cuda()
                outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
                layers[i] = layer.cpu()
                inps = outs
                torch.cuda.empty_cache()
            hidden_states = norm(outs)
            lm_logits = lm_head(hidden_states)
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous().cuda()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
            else:
                print("warning: nan or inf in lm_logits")
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
    print("PPL after pruning: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            throughput += end_time - start_time
            print("time: {}".format(end_time - start_time))
        if batch_idx >=5:
            break
        print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
        print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
        print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
        print("Throughput: {} tokens/sec".format(token_num / throughput))
        

@torch.no_grad()
def zeroshot_eval(
    model,
    tokenizer,
    tasks,
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    device="cuda"):
    """
    model: model name
    limit: number of test samples for debug, set to -1 is no limit
    tasks: str tasks are split by ,
    num_fewshot: Number of examples in few-shot context
    eval_ppl: str datasets are split by , such as 'wikitext2,ptb,c4'
    """
    lm = EvalLM(model, tokenizer, batch_size=batch_size, device=device)
    
    results = {}
            
    if tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            no_cache=True,
        )
        t_results = t_results["results"]
        print("results: {}".format(t_results))
        acc_list = [t_results[key]["acc"] for key in t_results.keys() if "acc" in t_results[key]]
        mean_acc = sum(acc_list) / len(acc_list)
        t_results["mean"] = mean_acc
        results.update(t_results)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS (formatted for easy copying)")
        print("="*50)
        
        for task_name in sorted(t_results.keys()):
            if task_name != "mean" and "acc" in t_results[task_name]:
                acc_value = t_results[task_name]["acc"] * 100  
                print(f"{task_name}: {acc_value:.2f}")
        print(f"mean: {mean_acc * 100:.2f}")  
    return results


# @torch.no_grad()
# def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
    #     num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    # from lm_eval import tasks, evaluator 
    # def pattern_match(patterns, source_list):
    #     task_names = set()
    #     for pattern in patterns:
    #         for matching in fnmatch.filter(source_list, pattern):
    #             task_names.add(matching)
    #     return list(task_names)
    # task_names = pattern_match(task_list, tasks.ALL_TASKS)
    # model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    # limit = None 
    # if "70b" in model_name or "65b" in model_name:
    #     limit = 2000
    # if use_accelerate:
    #     model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    # results = evaluator.simple_evaluate(
    #     model="hf-causal-experimental",
    #     model_args=model_args,
    #     tasks=task_names,
    #     num_fewshot=num_fewshot,
    #     batch_size=None,
    #     device=None,
    #     no_cache=True,
    #     limit=limit,
    #     description_dict={},
    #     decontamination_ngrams_path=None,
    #     check_integrity=False,
    #     pretrained_model=model,
    #     tokenizer=tokenizer, 
    #     add_special_tokens=add_special_tokens
    # )

    # return results 