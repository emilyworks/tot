import os
import json
import argparse
import time

from src.tot.tasks import get_task
from src.tot.methods.bfs import solve, naive_solve, solve_bench
# from src.tot.methods.a_star import solve, naive_solve, solve_bench
from src.tot.models import gpt_usage

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.quantization


def run(args):
    '''
    main run function
    '''
    #load in specific llama model, if applicable
    #bc of the way the original repo is structured, will need to load in llama models in run.py to avoid repeated loading in models.py
    if args.backend == 'llama':
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

    else: #gpt4 will be used later in this case
        model = None
        tokenizer = None

    #set up
    if args.task == 'comp_bench':
        test_data = torch.load('src/tot/data/agg_dl_test.pt')

        for sample in test_data:

            start_timer = perf.counter()
            
            WHILE WE GO THROUGH THE TREE
            #solve
            ys, info = solve_bench(args, model, tokenizer)

            runtime = time.perf_counter()-start_timer
            print(f"""For iteration {i} -- 
            Total Time to Solve: {runtime} seconds""")
            
            #evaluate/rate the proposals
            

            #compare the proposed final answer vs the ground truth
            gt = tokenizer.decode(sample['label'][0], return_tensors='pt').replace("<|eot_id|>", "").replace("<|begin_of_text|>", "").strip()
            # log
            # infos = [task.test_output(i, y) for y in ys]
            # info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far (gpt only)': gpt_usage(args.backend), 'total_runtime': runtime})
            # logs.append(info)
            # with open(file, 'w') as f:
            #     json.dump(logs, f, indent=4)
            
            # log main metric
            # accs = [info['r'] for info in infos]
            # cnt_avg += sum(accs) / len(accs)
            # cnt_any += any(accs)
            # print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
        
        # n = args.task_end_index - args.task_start_index
        # print(cnt_avg / n, cnt_any / n)
        # print('usage_so_far', gpt_usage(args.backend))

    else: #the original tot tasks from the paper
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
            print(f"""For iteration {i} -- 
            Total Time to Solve: {runtime} seconds""")

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

    #the arguments to use for our purposes
    args.add_argument('--backend', type=str, choices=['gpt-4o', 'llama'], default='gpt-4o')
    args.add_argument('--quantize', type=str, choices=['qat', 'ptq_int4', 'ptq_int8'])
    args.add_argument('--temperature', type=float, default=0.0)
    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')

    #other args from the original repo 
    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords', 'comp_bench'])

    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    
    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)