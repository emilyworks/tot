import os
import json
import argparse
import time
import vllm 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig
import torch
import torch.quantization
# import evaluate
from peft import AutoPeftModelForCausalLM

from src.tot.data.benchmark.bench import *
from src.tot.prompts.bench import value_prompt, propose_prompt

from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import multiprocessing

import ast
import re

import pandas as pd

import torchao
from torchao.quantization.quant_api import (
    quantize_,
    int8_dynamic_activation_int8_weight,
    int4_weight_only,
    int8_weight_only
)

all_gt = []
all_pred = []

# total_runtime = []
# average_sample_runtime = []
# setup_time = []
average_solving_time_per_sample = []
average_proposal_time_per_sample = []
average_eval_time_per_sample = []

temp_tuning = {}

def load_llama(quant=None):
    '''Load in one of the llama models'''
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    if args.quantize and args.quantize=='ptq_int4':
        quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="auto", quantization_config=quantization_config)
        torchao.quantization.utils.recommended_inductor_config_setter()
        # model = AutoModelForCausalLM.from_pretrained("src/tot/quant/hf_quant_int4", device_map="cuda", weights_only=False)
        model = torch.compile(model, mode="max-autotune")
    elif args.quantize and args.quantize=='ptq_int8':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        model.to('cuda')
        quantize_(model, int8_weight_only())
        # model = AutoModelForCausalLM.from_pretrained("src/tot/quant/ptq_int8", device_map="cuda", weights_only=False)
        model = torch.compile(model, mode="max-autotune")
    elif args.quantize and args.quantize == 'qat':
        model = AutoModelForCausalLM.from_pretrained("src/tot/quant/qat_int8_20", device_map="cuda", weights_only=False)
        model = torch.compile(model, mode="max-autotune")
    elif args.lora:
        model = AutoPeftModelForCausalLM.from_pretrained("src/tot/lora/peft_15")
    # elif args.vllm:
        # sampling_params = SamplingParams(n=1, max_tokens=100)
        # model = LLM(model="meta-llama/Llama-3.2-3B-Instruct", trust_remote_code=True, gpu_memory_utilization=0.9, max_model_len=2048)  # Name or path of your model
    else:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    return model, tokenizer

def a_star_penalty(num, depth, k=0.1):
    return num * np.exp(-k*depth)

def value_proposals(problem, current_state, proposals, tokenizer, model, device, cache=None, a_star=False, depth=None):
    '''
    Takes in string values of problem, current state, and proposals. 
    Returns a numerical valuation of each combination of the three factors above.
    '''
    valuations = []
    prompts = []

    # only eval if not prev evaluated
    noncached_proposals = [p for p in proposals if p not in cache]
    cache_hits = len(proposals) - len(noncached_proposals)
    
    for p in noncached_proposals:
        prompts.append(value_prompt.format(problem=problem, current_state=current_state, proposal=p))
    
    values = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    value_inputs = values['input_ids'].to(device)
    value_masks = values['attention_mask'].to(device)

    outputs = model.generate(value_inputs, attention_mask=value_masks, max_new_tokens=5)
    readable_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if not a_star:
        for o in readable_outputs:
            o = o.split("Evaluation:")[-1]
            if 'sure' in o and 'current state is the solution' in o:
                valuations.append(100.0)
            elif 'sure' in o and 'current state is the solution' not in o:
                valuations.append(1.0)
            elif 'likely' in o:
                valuations.append(0.5)
            else:
                valuations.append(0.0)
    else:
        for o in readable_outputs:
            o = o.split("Evaluation:")[-1]
            if 'sure' in o and 'current state is the solution' in o:
                valuations.append(100.0)
            elif 'sure' in o and 'current state is the solution' not in o:
                valuations.append(a_star_penalty(1.0, depth=depth))
            elif 'likely' in o:
                valuations.append(a_star_penalty(0.5, depth=depth))
            else:
                valuations.append(0.0)

    for p, v in list(zip(noncached_proposals, valuations)):
        cache[p] = v

    # could maybe be optimized but should be fine
    valuations = [cache[p] for p in proposals]

    return valuations, cache_hits

def parse_problem(problem, math=False):
    '''
    parsing helper function
    '''
    if not math:
        pattern = r"Choices:\s*(\[[^\]]+\])"

        # Search for the pattern in the question string
        match = re.search(pattern, problem)

        # If there's a match, process the choices into a list
        if match:
            choices_str = match.group(1)            
            choices_list = ast.literal_eval(choices_str)

            return choices_list
        else:
            print("No choices found.")
            return []
    else:
        matches = re.findall(r'\\boxed{([^}]*)}', problem)
        if matches and len(matches) > 0:
            return matches[-1]
        else:
            print("No choices found.")
            return []


def final_eval(gt, final_prop, problem):
    '''
    compare the ground truth and final proposed solution by the model
    '''
    if "Current State" in final_prop:
        final_prop = final_prop.split("Current_State")[-1]
    final_prop = final_prop.replace("Possible next step:", "").replace("Current State:", "").strip()

    if "Choices" in problem: #multiple choice scenario
        try:
            parsed = parse_problem(problem)
            gt = parsed[int(gt)]

            all_pred.append(final_prop)
            all_gt.append(gt)
            if gt in final_prop:
                return 1.0
            else: #other problem scenarios
                return 0.0
        except:
            return 0.0
    else:
        # print(gt)
        gt = parse_problem(gt, math=True)
        all_pred.append(final_prop)
        all_gt.append(gt)

        if isinstance(gt, str) and gt in final_prop:
            return 1.0
        else:
            return 0.0


def get_test_data(tokenizer, batch_size):
    '''
    Process and return the composite benchmark test data in a dataloader
    '''
    gpqa_raw = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    gpqa_choices = [[a, b, c, d] for a, b, c, d in
                    zip(gpqa_raw['train']['Correct Answer'], gpqa_raw['train']['Incorrect Answer 1'],
                        gpqa_raw['train']['Incorrect Answer 2'], gpqa_raw['train']['Incorrect Answer 3'])]
    for choices in gpqa_choices:
        random.shuffle(choices)

    gpqa_questions_proc = format_for_mm(gpqa_raw['train']['Question'], gpqa_choices)

    #math (for math)
    math_raw = load_dataset("lighteval/MATH", "all")

    # #mmlu (for gen knowledge + reasoning)
    mmlu_raw = load_dataset("cais/mmlu", "all")
    mmlu_questions_proc_test = format_for_mm(mmlu_raw['test']['question'], mmlu_raw['test']['choices'])

    #master list - test
    # sublist_input_test = gpqa_questions_proc[158:] + math_raw['test']['problem'] + mmlu_questions_proc_test
    # sublist_answer_test = gpqa_raw['train']['Correct Answer'][158:] + math_raw['test']['solution'] + mmlu_raw['test']['answer']
    # agg_test_set = benchmark_dataset(sublist_input_test, sublist_answer_test, tokenizer)
    agg_test_set = benchmark_dataset(math_raw['test']['problem'], math_raw['test']['solution'], tokenizer)

    return DataLoader(agg_test_set, batch_size=batch_size, collate_fn=collate_fn_qat)

def solve(input_ids, label, mask, model, tokenizer, device, args):
    '''
    the main tot run
    '''
    
    problem = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    # print(problem)
    selected = ""
    valuation_cache = {}    # cache for repeated valuations
    proposals = []    # persist the queue across iterations
    
    for i in range(args.depth): #args.depth number of attempts to reach the solution
        
        #propose next step/solutions per node/prompt
        rpropose = time.perf_counter()
        out = model.generate(
            input_ids,
            attention_mask=mask,
            temperature=args.temperature, 
            max_new_tokens=input_ids.shape[1]+args.max_new_tokens, 
            num_return_sequences=args.breadth,
            )
        rpropose = time.perf_counter()-rpropose
        average_proposal_time_per_sample.append(rpropose)

        #evaluate/rate the proposals
        current_state = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

        # proposals = []
        for o in out:
            string_answer = tokenizer.decode(o, skip_special_tokens=True)
            string_answer = string_answer.split("Possible next step:")[-1]
            # print(string_answer)
            # print("+++++"*50)
            # assert isinstance(string_answer, str)
            proposals.extend([string_answer])
        # exit()
        # could collect cache hit statistics if necessary
        reval = time.perf_counter()
        valuations, cache_hits = value_proposals(problem=problem, current_state=current_state, proposals=proposals, tokenizer=tokenizer, model=model, device=device, cache=valuation_cache)
        reval = time.perf_counter() - reval
        average_eval_time_per_sample.append(reval)

        #if the model believes it has reached the final solution before args.depth is up, break
        if 100.0 in valuations:
            break
        
        #select the best proposal
        val_props = list(zip(proposals, valuations))
            
        val_props.sort(key = lambda ele: ele[1], reverse=True)
        val_props = val_props[:args.greedy_n]
        selected = val_props[0][0]
        val_props = val_props[1:]    # remove the selected node from the queue to avoid reeval
        proposals = [p for vp[0] in val_props]    # update the queue to include the greedy_n highest ranking nodes

        #format the chosen proposal for the next iteration
        next_prompt = propose_prompt.format(problem=problem, current_state=selected)
        inputs = tokenizer(next_prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)


    #compare the proposed final answer vs the ground truth
    gt = tokenizer.batch_decode(label, skip_special_tokens=True)
    judgement = final_eval(gt[0], selected, problem)

    return judgement

def run(args):
    '''
    main run function
    '''
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    rtotal = time.perf_counter()
    rsetup = time.perf_counter()
    ### SETUP MODEL ###
    #bc of the way the original repo is structured, will need to load in llama models in run.py to avoid repeated loading in models.py
    if args.quantize:
        model, tokenizer = load_llama(args.quantize)
    else:
        model, tokenizer = load_llama()

    tokenizer.pad_token = tokenizer.eos_token

    ### SETUP DATA ###
    test_data = get_test_data(tokenizer, args.concurrent)

    ### OTHER SETUP ###
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    total = 0
    right = 0

    custom_stop = args.num_test_samp if args.num_test_samp else 13706
    count = 0

    rsetup = time.perf_counter()-rsetup
    
    for sample in test_data:

        #extract out the sample parts for the initial input
        input_ids = sample['input_ids'].to(device)
        label = sample['label'].to(device)
        mask = sample['attention_mask'].to(device)
        
        #cannot get multiple gpus. will run this on a single gpu one sample at a time for simplicity
        for i in range(len(input_ids)):

            rsolve = time.perf_counter()
            judgement = solve(input_ids[i].view(1,-1), label[i].view(1,-1), mask[i].view(1,-1), model, tokenizer, device, args)
            rsolve = time.perf_counter()-rsolve
            average_solving_time_per_sample.append(rsolve)

            total += 1.0
            right += judgement
            count += 1
            if count == custom_stop:
                break
        if count == custom_stop:
            break
        #keep track of the running totals
        print("Accuracy so far: ", right/total)

    print("FINAL ACCURACY: ", right/total)
    # temp_tuning[args.temperature] = right/total

    #temp save
    res = pd.DataFrame({
        "gt": all_gt,
        "pred": all_pred
    })
    if args.a_star:
        if args.quantize:
            res.to_csv(f"./results_{args.backend}_{args.quantize}_{args.temperature}_{args.num_test_samp}_astar.csv")
        elif args.lora:
            res.to_csv(f"./results_{args.backend}_lora_{args.temperature}_{args.num_test_samp}_astar.csv")
        else:
            res.to_csv("./res.csv")
    else:
        if args.quantize:
            res.to_csv(f"./results_{args.backend}_{args.quantize}_{args.temperature}_{args.num_test_samp}.csv")
        elif args.lora:
            res.to_csv(f"./results_{args.backend}_lora_{args.temperature}_{args.num_test_samp}.csv")
        else:
            res.to_csv("./res_astar.csv")

    rtotal = time.perf_counter()-rtotal

    peak = torch.cuda.max_memory_allocated()

    time_df = pd.DataFrame({
        "total_accuracy": right/total,
        "total runtime": rtotal,
        "total setup time": rsetup,
        "average solving time": sum(average_solving_time_per_sample)/len(average_solving_time_per_sample),
        "average proposal time": sum(average_proposal_time_per_sample)/len(average_proposal_time_per_sample),
        "average eval time": sum(average_eval_time_per_sample)/len(average_eval_time_per_sample),
        "peak memory usage": peak
    }, index=[0])

    if args.a_star:
        if args.quantize:
            time_df.to_csv(f"./times_{args.backend}_{args.quantize}_{args.temperature}_{args.num_test_samp}_astar.csv")
        elif args.lora:
            time_df.to_csv(f"./times_{args.backend}_lora_{args.temperature}_{args.num_test_samp}_astar.csv")
        else:
            time_df.to_csv("./times_astar.csv")
    else:
        if args.quantize:
            time_df.to_csv(f"./times_{args.backend}_{args.quantize}_{args.temperature}_{args.num_test_samp}.csv")
        elif args.lora:
            time_df.to_csv(f"./times_{args.backend}_lora_{args.temperature}_{args.num_test_samp}.csv")
        else:
            time_df.to_csv("./times.csv")
    #courtesy prints
    print("TOTAL RUNNING TIME: ", rtotal)
    print("SETUP TIME: ", rsetup)
    print(f"PEAK GPU MEM USAGE: {peak / 1e6:.2f} MB")
        

def parse_args():
    '''
    Determines the conditions for the run.
    '''
    args = argparse.ArgumentParser()

    #the arguments to use for our purposes
    args.add_argument('--backend', type=str, choices=['gpt-4o', 'llama'], default='gpt-4o')
    args.add_argument('--quantize', type=str, choices=['qat', 'ptq_int4', 'ptq_int8'])
    args.add_argument('--temperature', type=float, default=0.0)
    args.add_argument('--max_new_tokens', type=int, default=100)
    args.add_argument('--depth', type=int, default=3)
    args.add_argument('--breadth', type=int, default=3)
    args.add_argument('--greedy_n', type=int, default=1)
    args.add_argument('--concurrent', type=int, default=4)
    args.add_argument('--a_star', action='store_true')
    args.add_argument('--lora', action='store_true')
    args.add_argument('--num_test_samp', type=int)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    #test base instruct llama
    # print(args)
    # run(args)

    #test quant llama w/ qat int8
    # args.quantize="qat"
    # print(args)
    # run(args)

    #test llama w/ ptq int4
    # args.quantize="ptq_int4"
    # print(args)
    # run(args)

    #test llama w/ ptq int8
    # args.quantize="ptq_int8"
    # print(args)
    # run(args)

    #test llama w/ lora
    # args.quantize=None
    # args.lora = True
    # print(args)
    # run(args)

    #lora with the a_star run
    args.a_star = True
    args.lora=True
    print(args)
    run(args)

    #qat with the a_star run
    # args.quantize='qat'
    # args.lora=False
    # print(args)
    # run(args)

    # print("THIS IS TEMP TUNING")
    # print(temp_tuning.items())
    # temp = pd.DataFrame(temp_tuning)
    # temp.to_csv('./temp_tuning.csv')
