import os
import json
import argparse
import time
import random
import multiprocessing
import ast
import re
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig

import torch
import torch.quantization
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import AutoPeftModelForCausalLM

from src.tot.data.benchmark.bench import *
from src.tot.prompts.bench import value_prompt, propose_prompt

import torchao
from torchao.quantization.quant_api import (
    quantize_,
    int8_dynamic_activation_int8_weight,
    int4_weight_only,
    int8_weight_only
)

all_gt = []
all_pred = []
average_solving_time_per_sample = []
average_proposal_time_per_sample = []
average_eval_time_per_sample = []

temp_tuning = {} #used for tuning the temperature hyperparam during a separate sub-experiment.


def load_llama(args):
    '''Load in one of the llama models'''
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    if args.quantize and args.quantize=='ptq_int4':
        quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="auto", quantization_config=quantization_config)
        torchao.quantization.utils.recommended_inductor_config_setter()
        model = torch.compile(model, mode="max-autotune")
    elif args.quantize and args.quantize=='ptq_int8':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        quantize_(model, int8_weight_only())
        model = torch.compile(model, mode="max-autotune")
        model.to('cuda')
    elif args.quantize and args.quantize == 'qat':
        model = AutoModelForCausalLM.from_pretrained("src/tot/quant/qat_int8_20", device_map="cuda")
        model = torch.compile(model, mode="max-autotune")
    elif args.lora:
        model = AutoPeftModelForCausalLM.from_pretrained("src/tot/lora/peft_15")
    else:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    return model, tokenizer

def a_star_penalty(num, depth, k=0.1):
    '''
    custom penalty function to reflect the cost of traversing as iterations per sample increases
    '''
    return num * np.exp(-k*depth)

def value_proposals(problem, current_state, proposals, tokenizer, model, device, a_star=False, cache=None, depth=None):
    '''
    Takes in string values of problem, current state, and proposals. 
    Returns a numerical valuation of each combination of the three factors above.
    '''
    valuations = []
    prompts = []

    #evaluate proposals that are not in cache.
    #during evaluation, assign a numerical value to the proposals based on how 'promising' they are for reaching the solution

    noncached_proposals = [p for p in proposals if p not in cache]
    cache_hits = len(proposals) - len(noncached_proposals) #a courtesy statistic
    
    for p in noncached_proposals:
        prompts.append(value_prompt.format(problem=problem, current_state=current_state, proposal=p))
    
    if len(prompts) > 0: 
        values = tokenizer(prompts, return_tensors='pt', max_length=150, padding='max_length', truncation=True)
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

    valuations = [cache[p] for p in proposals]

    return valuations, cache_hits

def parse_problem(problem, math=False):
    '''
    Helper function to parse the answer out of the ground-truth paragraph.
    (note: math=false param was used for a previous experiment iteration with other datasets. currently, parse_problem always expected to be called w/ math=True)
    '''
    matches = re.findall(r'\\boxed{([^}]*)}', problem)
    if matches and len(matches) > 0:
        return matches[-1]
    else:
        print("No choices found.")
        return []


def final_eval(gt, final_prop, problem):
    '''
    Compare the ground truth and final proposed solution by the model
    '''
    #a bit of parsing and cleanup
    if "Current State" in final_prop:
        final_prop = final_prop.split("Current_State")[-1]
    final_prop = final_prop.replace("Possible next step:", "").replace("Current State:", "").strip()
    gt = parse_problem(gt, math=True)
    
    #log the answers
    all_pred.append(final_prop)
    all_gt.append(gt)

    #check if the model's answer is right or not
    if isinstance(gt, str) and gt in final_prop:
        return 1.0
    else:
        return 0.0


def get_test_data(tokenizer, batch_size):
    '''
    Process and return the MATH benchmark test data in a dataloader
    '''
    math_raw = load_dataset("lighteval/MATH", "all")
    agg_test_set = benchmark_dataset(math_raw['test']['problem'], math_raw['test']['solution'], tokenizer)

    return DataLoader(agg_test_set, batch_size=batch_size, collate_fn=collate_fn_qat)

def solve_astar(input_ids, label, mask, model, tokenizer, device, args):
    '''
    The ToT run for a single sample for A*
    '''
    #some inits
    problem = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
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

        current_state = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        
        for o in out:
            string_answer = tokenizer.decode(o, skip_special_tokens=True)
            string_answer = string_answer.split("Possible next step:")[-1]
            proposals.extend([string_answer])
        
        #evaluate/rate the proposals
        reval = time.perf_counter()
        valuations, cache_hits = value_proposals(problem=problem, current_state=current_state, proposals=proposals, tokenizer=tokenizer, model=model, device=device, cache=valuation_cache, a_star=True, depth=i)
        reval = time.perf_counter() - reval
        average_eval_time_per_sample.append(reval)

        #if the model believes it has reached the final solution before args.depth is up, break
        if 100.0 in valuations:
            break
        
        #select the best proposal
        val_props = list(zip(proposals, valuations))
        val_props.sort(key = lambda ele: ele[1], reverse=True)
        selected = val_props[:args.greedy_n][0][0]
        proposals = [p[0] for p in val_props[:args.q_size]] #maintain a sorted q of promising proposals that carry forward to future iterations

        #format the chosen proposal for the next iteration
        next_prompt = propose_prompt.format(problem=problem, current_state=selected)
        inputs = tokenizer(next_prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)


    #compare the proposed final answer vs the ground truth
    gt = tokenizer.batch_decode(label, skip_special_tokens=True)
    judgement = final_eval(gt[0], selected, problem)

    return judgement

def solve(input_ids, label, mask, model, tokenizer, device, args):
    '''
    The ToT run for a single sample for baseline BFS
    '''
    #some inits
    problem = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    valuation_cache = {}    # cache for repeated valuations
    selected = ""
    
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

        proposals = []
        for o in out:
            string_answer = tokenizer.decode(o, skip_special_tokens=True)
            string_answer = string_answer.split("Possible next step:")[-1]
            proposals.extend([string_answer])

        reval = time.perf_counter()
        
        valuations, cache_hit = value_proposals(problem=problem, current_state=current_state, proposals=proposals, tokenizer=tokenizer, model=model, device=device, cache=valuation_cache)
        reval = time.perf_counter() - reval
        average_eval_time_per_sample.append(reval)

        #if the model believes it has reached the final solution before args.depth iterations is up, break
        if 100.0 in valuations:
            break
        
        #select the best proposal
        val_props = list(zip(proposals, valuations))
        val_props.sort(key = lambda ele: ele[1], reverse=True)
        selected = val_props[:args.greedy_n][0][0]

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
    #cleanup before recording memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    rtotal = time.perf_counter()
    rsetup = time.perf_counter()

    #load in the model and tokenizer
    model, tokenizer = load_llama(args)
    tokenizer.pad_token = tokenizer.eos_token

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    #construct the dataloader
    test_data = get_test_data(tokenizer, args.concurrent)

    #other setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    total = 0
    right = 0

    #for running smaller subsets of data
    custom_stop = args.num_test_samp if args.num_test_samp else 2000
    total = 0
    rsetup = time.perf_counter()-rsetup

    #main loop    
    for sample in test_data:

        #extract out the sample parts for the initial input
        input_ids = sample['input_ids'].to(device)
        label = sample['label'].to(device)
        mask = sample['attention_mask'].to(device)
        
        #cannot get multiple gpus. will run this on a single gpu one sample at a time for simplicity
        for i in range(len(input_ids)):

            rsolve = time.perf_counter()

            #use ToT to solve for given sample
            if args.a_star:
                judgement = solve_astar(input_ids[i].view(1,-1), label[i].view(1,-1), mask[i].view(1,-1), model, tokenizer, device, args)
            else:
                judgement = solve(input_ids[i].view(1,-1), label[i].view(1,-1), mask[i].view(1,-1), model, tokenizer, device, args)

            rsolve = time.perf_counter()-rsolve
            average_solving_time_per_sample.append(rsolve)

            total += 1.0
            right += judgement
            if total == custom_stop:
                break
        if total == custom_stop:
            break

        #keep track of the running totals
        print(f"Accuracy So Far after {total} samples: {right/total}")

    print("Final Accuracy: ", right/total)
    # temp_tuning[args.temperature] = right/total

    rtotal = time.perf_counter()-rtotal

    #save the profiling statistics
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
            time_df.to_csv(f"./times_{args.backend}_{args.temperature}_{args.num_test_samp}_astar.csv")
    else:
        if args.quantize:
            time_df.to_csv(f"./times_{args.backend}_{args.quantize}_{args.temperature}_{args.num_test_samp}.csv")
        elif args.lora:
            time_df.to_csv(f"./times_{args.backend}_lora_{args.temperature}_{args.num_test_samp}.csv")
        else:
            time_df.to_csv(f"./times_{args.backend}_{args.temperature}_{args.num_test_samp}.csv")
    
    return time_df

def run_multiples(args):
    '''
    Helper function for running the same trial multiple times for stability. Saves a df of stats across identical trials.
    Returns the average as a courtesy.
    '''
    #run the trials
    profiling_stats = run(args)
    for i in range(args.num_repeat-1):
        profiling_stats = profiling_stats.merge(run(args), how='outer')

    #save the agg trial stats
    if args.a_star:
        if args.quantize:
            profiling_stats.to_csv(f"./AGGprofiling_stats_{args.backend}_{args.quantize}_{args.temperature}_{args.num_test_samp}_astar.csv")
        elif args.lora:
            profiling_stats.to_csv(f"./AGGprofiling_stats_{args.backend}_lora_{args.temperature}_{args.num_test_samp}_astar.csv")
        else:
            profiling_stats.to_csv(f"./AGGprofiling_stats_{args.backend}_{args.temperature}_{args.num_test_samp}_astar.csv")
    else:
        if args.quantize:
            profiling_stats.to_csv(f"./AGGprofiling_stats_{args.backend}_{args.quantize}_{args.temperature}_{args.num_test_samp}.csv")
        elif args.lora:
            profiling_stats.to_csv(f"./AGGprofiling_stats_{args.backend}_lora_{args.temperature}_{args.num_test_samp}.csv")
        else:
            profiling_stats.to_csv(f"./AGGprofiling_stats_{args.backend}_{args.temperature}_{args.num_test_samp}.csv")

    #print and return info as dict.
    print(f"FINAL RESULTS FROM {args.num_repeat} RUNS:")
    print(args)
    print(profiling_stats.mean().to_dict()) #can also easily be switched to median if desired. median is easy to see from the saved csvs above
    print("*****"*10)

    return profiling_stats.mean().to_dict() #can also easily be switched to median if desired
        
def parse_args():
    '''
    Determines the conditions for the run.
    '''
    args = argparse.ArgumentParser()

    #the arguments to use for our purposes
    args.add_argument('--backend', type=str, choices=['gpt-4o', 'llama'], default='llama')
    args.add_argument('--quantize', type=str, choices=['qat', 'ptq_int4', 'ptq_int8'])
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--max_new_tokens', type=int, default=100)
    args.add_argument('--depth', type=int, default=3)
    args.add_argument('--breadth', type=int, default=3)
    args.add_argument('--greedy_n', type=int, default=1)
    args.add_argument('--concurrent', type=int, default=4) #was planning to use this as batch size w/ multiple gpus but due to gpu constraints, the plan changed. see above for modificaion.
    args.add_argument('--a_star', action='store_true')
    args.add_argument('--lora', action='store_true')
    args.add_argument('--num_test_samp', type=int, default=50)
    args.add_argument('--q_size', type=int, default=5)
    args.add_argument('--num_repeat', type=int, default=3)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    print(args)
    run_multiples(args) #can also just call run(args) directly if only a single trial is desired

    # Originally run for temperature hyperparam tuning for a separate prior experiment iteration.
    # print(temp_tuning.items())
    # temp = pd.DataFrame(temp_tuning)
    # temp.to_csv('./temp_tuning.csv')
