import os
import json
import argparse
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.quantization

from src.tot.data.benchmark.bench import *
from src.tot.prompts.bench import value_prompt, propose_prompt

from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import multiprocessing

def load_llama(quant=None):
    '''Load in one of the llama models'''
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    if args.quantize and args.quantize=='ptq_int4':
        model = AutoModelForCausalLM.from_pretrained("src/tot/quant/hf_quant_int4", device_map="cuda", weights_only=False)
        model = torch.compile(model, mode="max-autotune")
    elif args.quantize and args.quantize=='ptq_int8':
        model = AutoModelForCausalLM.from_pretrained("src/tot/quant/ptq_int8", device_map="cuda")
        model = torch.compile(model, mode="max-autotune")
    elif args.quantize and args.quantize == 'qat':
        model = AutoModelForCausalLM.from_pretrained("src/tot/quant/qat_int8", device_map="cuda", weights_only=False)
        model = torch.compile(model, mode="max-autotune")
    else:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    return model, tokenizer

def value_proposals(problem, current_state, proposals, tokenizer, model, device):
    '''
    Takes in string values of problem, current state, and proposals. 
    Returns a numerical valuation of each combination of the three factors above.
    '''
    valuations = []
    prompts = []
    for p in proposals:
        prompts.append(value_prompt.format(problem=problem, current_state=current_state, proposal=p))
    
    values = tokenizer(prompts, return_tensors='pt')
    value_inputs = values['input_ids'].to(device)
    value_masks = values['attention_mask'].to(device)

    outputs = model.generate(value_inputs, attention_mask=value_masks, max_new_tokens=5)
    readable_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for o in readable_outputs:
        if 'sure' in o and 'current state is the solution' in o:
            valuations.append(100.0)
        elif 'sure' in o and 'current state is the solution' not in o:
            valuations.append(1.0)
        elif 'likely' in o:
            valuations.append(0.5)
        else:
            valuations.append(0.0)

    return valuations

def final_eval(gt, final_prop):
    '''
    compare the ground truth and final proposed solution by the model
    '''
    print("THIS IS THE FINAL PROP")
    print(final_prop)
    print("THIS IS THE GT")
    
    if gt in final_prop:
        return 1.0
    else:
        return 0.0

def get_test_data(tokenizer, batch_size):
    '''
    Process and return the composite benchmark test data in a dataloader
    '''
    # print(tokenizer)

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
    sublist_input_test = gpqa_questions_proc[158:] + math_raw['test']['problem'] + mmlu_questions_proc_test
    sublist_answer_test = gpqa_raw['train']['Correct Answer'][158:] + math_raw['test']['solution'] + mmlu_raw['test']['answer']
    agg_test_set = benchmark_dataset(sublist_input_test, sublist_answer_test, tokenizer)

    return DataLoader(agg_test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_qat)

def solve(input_ids, label, mask, model, tokenizer, device, args):
    '''
    the main tot run
    '''
    
    problem = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    # print(problem)
    selected = ""
    for i in range(args.depth): #args.depth number of attempts to reach the solution
        
        #propose next step/solutions per node/prompt

        out = model.generate(
            input_ids,
            attention_mask=mask,
            temperature=args.temperature, 
            max_new_tokens=args.max_new_tokens, 
            num_return_sequences=args.breadth,
            )

        
        #evaluate/rate the proposals
        current_state = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

        proposals = []
        for o in out:
            string_answer = tokenizer.decode(o[-args.max_new_tokens:], skip_special_tokens=True)
            assert isinstance(string_answer, str)
            proposals.extend([string_answer])

        valuations = value_proposals(problem=problem, current_state=current_state, proposals=proposals, tokenizer=tokenizer, model=model, device=device)

        #if the model believes it has reached the final solution before args.depth is up, break
        if 100.0 in valuations:
            break
        
        #select the best proposal
        val_props = list(zip(proposals, valuations))
        val_props.sort(key = lambda ele: ele[1], reverse=True)
        selected = val_props[:args.greedy_n][0][0]

        # print("THIS IS SELCTED")
        # print(selected)

        #format the chosen proposal for the next iteration
        next_prompt = propose_prompt.format(problem=problem, current_state=selected)
        inputs = tokenizer(next_prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)


    #compare the proposed final answer vs the ground truth
    gt = tokenizer.batch_decode(label, skip_special_tokens=True)
    judgement = final_eval(gt[0], selected)

    return judgement

def run(args):
    '''
    main run function
    '''
    ### SETUP MODEL ###
    #bc of the way the original repo is structured, will need to load in llama models in run.py to avoid repeated loading in models.py
    if args.backend == 'llama':
        if args.quantize:
            model, tokenizer = load_llama(args.quantize)
        else:
            model, tokenizer = load_llama()
    else: #gpt4 will be used later in this case
        model = None
        tokenizer = None

    ### SETUP DATA ###
    test_data = get_test_data(tokenizer, args.concurrent)

    ### OTHER SETUP ###
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    total = 0
    right = 0

    for sample in test_data:

        #extract out the sample parts for the initial input
        input_ids = sample['input_ids'].to(device)
        label = sample['label'].to(device)
        mask = sample['attention_mask'].to(device)
        
        #cannot get multiple gpus. will run this on a single gpu one sample at a time for simplicity
        for i in range(len(input_ids)):
            judgement = solve(input_ids[i].view(1,-1), label[i].view(1,-1), mask[i].view(1,-1), model, tokenizer, device, args)
            total += 1.0
            right += judgement

        #keep track of the running totals
        print("Accuracy so far: ", right/total)

    print("FINAL ACCURACY: ", right/total)

        

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

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)