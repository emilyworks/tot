import os
import json
import argparse
import time

from src.tot.tasks import get_task
from src.tot.methods.bfs import solve, naive_solve
from src.tot.models import gpt_usage

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.quantization


def run(args):
    '''
    main run function
    '''
    #load in non-gpt model in this driver function for now to avoid repeated loading later on
    if args.backend == 'llama':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        
        if args.quantize and args.quantize=='ptq':
            model.train()
            model.qconfig = torch.quantization.get_default_qconfig('x86')
            torch.quantization.prepare(model, inplace=True)
            for _, mod in model.named_modules():
                if isinstance(mod, torch.nn.Embedding):
                    mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
            model = torch.quantization.convert(model, inplace=True)
            model.load_state_dict(torch.load('quant_experiments/quantized_model.pth'))
            model.eval()
        elif args.backend == 'qat':
            pass
            # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
            # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8")
        else:
            pass
    else:
        model = None
        tokenizer = None

    #set up
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    #run the specified range of tasks
    for i in range(args.task_start_index, args.task_end_index):

        # solve
        start_timer = time.perf_counter()
        if args.naive_run:
            ys, info = naive_solve(args, task, i, model, tokenizer) 
        else:
            ys, info = solve(args, task, i, model, tokenizer)

        runtime = time.perf_counter()-start_timer
        # print(runtime)

        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far (gpt only)': gpt_usage(args.backend), 'total_runtime': runtime})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
        
        # log main metric
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
    
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', gpt_usage(args.backend))


def parse_args():
    '''
    Determines the conditions for the run.
    '''
    args = argparse.ArgumentParser()

    #what model to use
    args.add_argument('--backend', type=str, choices=['gpt-4o', 'llama'], default='gpt-4o')
    args.add_argument('--quantize', type=str, choices=['qat', 'ptq', 'spinquant'])

    #what temperature to use
    args.add_argument('--temperature', type=float, default=0.0)

    #the problem task
    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])

    #which tasks from the data file to solve
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    
    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run
    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)