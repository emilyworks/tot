import os
import json
import argparse
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.quantization

from src.tot.prompts.bench import value_prompt, propose_prompt

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
    print(gt)

    if gt in final_prop:
        return 1.0
    else:
        return 0.0


def run(args):
    '''
    main run function
    '''
    #load in specific llama model, if applicable
    #bc of the way the original repo is structured, will need to load in llama models in run.py to avoid repeated loading in models.py
    if args.backend == 'llama':
        if args.quantize:
            model, tokenizer = load_llama(args.quantize)
        else:
            model, tokenizer = load_llama()
    else: #gpt4 will be used later in this case
        model = None
        tokenizer = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    #set up
    test_data = torch.load('src/tot/data/agg_dl_test.pt')

    total = 0
    right = 0

    for samples in test_data:
        
        for sample in samples: #going to do this one problem at a time for now.

            #extract out the sample parts for the initial input
            input_ids = sample['input_ids'].to(device)
            label = sample['label'].to(device)
            mask = sample['attention_mask'].to(device)

            problem = tokenizer.decode(sample, skip_special_tokens=True)
            
            #start solving via tot
            start_timer = perf.counter()
            
            selected = ""
            for i in range(args.depth): #args.depth number of attempts to reach the solution
                
                #propose next step/solutions per node/prompt

                out = model.generate(
                    input_ids,
                    attention_mask=mask,
                    temperature=args.temperature, 
                    max_new_tokens=args.max_tokens, 
                    num_return_sequences=args.breadth)

                
                #evaluate/rate the proposals
                current_state = tokenizer.decode(input_ids, skip_special_tokens=True)

                proposals = []
                for o in out:
                    string_answer = tokenizer.decode(o)
                    proposals.extend([string_answer])

                valuations = value_proposals(problem=problem, current_state=current_state, proposals=proposals, tokenizer=tokenizer, model=model, device=device)

                #if the model believes it has reached the final solution before args.depth is up, break
                if 100.0 in valuations:
                    break
                
                #select the best proposal
                val_props = list(zip(proposals, valuations))
                val_props.sort(key = lambda ele: ele[1], descending=True)
                selected = val_props[:args.greedy_n]

                #format the chosen proposal for the next iteration
                next_prompt = propose_prompt.format(problem=problem, current_state=selected)
                inputs = tokenizer(next_prompt, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                mask = inputs['attention_mask'].to(device)


            #compare the proposed final answer vs the ground truth
            gt = tokenizer.decode(label, skip_special_token=True)
            judgement = final_eval(gt, selected)

            #keep track of the running totals
            total += 1.0
            right += judgement
            print("Accuracy so far: ", total/right)

    total_accuracy = right/total

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

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)