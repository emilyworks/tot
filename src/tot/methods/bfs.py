import itertools
import numpy as np
from functools import partial
from ..models import inference_model
import time 

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt], 0.0 #0 for inference latency bc cache was used
    value_outputs, eval_time = inference_model(value_prompt, n=n_evaluate_sample, stop=None)

    value = task.value_outputs_unwrap(x, y, value_outputs)

    if cache_value:
        task.value_cache[value_prompt] = value

    return [value, eval_time[0]] #assumes one value, one time element

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):

    values = []
    times = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
            time = 0
        else:    
            value, time = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value

        values.append(value)
        times.append(time)
    # print(values)
    return values, times

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.cot_prompt_wrap(x, ys)
    vote_outputs = inference_model(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    final_proposals = []
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals, generate_times = inference_model(propose_prompt, n=5, stop=None)
    for prop in proposals:
        final_proposals.extend(prop.split('\n'))
    return ([y + _ + '\n' for _ in final_proposals], generate_times)

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = inference_model(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, model, tokenizer, to_print=True):
    global inference_model
    inference_model = partial(inference_model, model=model, tokenizer=tokenizer, temperature=args.temperature)
    print(inference_model)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []


    for step in range(task.steps):
        # print("Started Steps!!")

        # generation
        print("Started Generation...")
        new_ys_with_times = []
        # TODO: be mindful with n_generate_sample and n_select_sample to avoid node explosion
        # n_select_sample * n_generate_sample is the highest possible number of additional evaluations each step
        # total task.steps * n_select_sample * n_generate_sample
        if args.method_generate == 'sample':
            for y in ys:
                generated_ys, generate_times = get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step])
                new_ys_with_times.extend(zip(generated_ys, generate_times))
        else:
            for y in ys:
                generated_ys, generate_times = get_proposals(task, x, y)
                new_ys_with_times.extend(zip(generated_ys, generate_times))

        new_ys, generate_times = zip(*new_ys_with_times)
        new_ys = list(new_ys)
        generate_times = list(generate_times)

        # new_ys = list(itertools.chain(new_ys))
        new_ys = ys + new_ys    # extend the current queue with the frontier
        
        ids = list(range(len(new_ys)))

        # evaluation
        # shouldn't worry about reevaluating the same ys as the values should be saved in the task cache
        # but could potentially add logic to remove expanded from queue
        print("Finished Generation...Started Eval!")

        start_time = time.perf_counter()

        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values, eval_times = get_values(task, x, new_ys, args.n_evaluate_sample)
        
        eval_time = time.perf_counter()-start_time
        print(f"Node Eval Time: {eval_time} seconds")
        
        # selection
        print("Finished Eval...Started Selection...")

        start_time = time.perf_counter()

        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        selection_time = time.perf_counter()-start_time()
        print(f"Selection Time: {selection_time} seconds")

        # log
        print("Finished Selection...Logging...")
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n-- generate times --: {generate_times}\n-- eval times --: {eval_times}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys, 'generate_times': generate_times, 'eval_times': eval_times})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global inference_model
    inference_model = partial(inference_model, model=args.backend, temperature=args.temperature)
    print(inference_model)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}

def solve_bench(args, model, tokenizer, to_print=True, depth=5, breadth=2):
    global inference_model
    inference_model = partial(inference_model, model=model, tokenizer=tokenizer, temperature=args.temperature)

    ys = ['']  # current output candidates
    infos = []

    for step in range(5): 
        # print("Started Steps!!")

        # generation
        print("Started Generation...")
        new_ys_with_times = []

        generated_ys, generate_times = get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step])
        new_ys_with_times.extend(zip(generated_ys, generate_times))

        new_ys, generate_times = zip(*new_ys_with_times)
        new_ys = list(new_ys)
        generate_times = list(generate_times)

        # new_ys = list(itertools.chain(new_ys))
        new_ys = ys + new_ys    # extend the current queue with the frontier
        
        ids = list(range(len(new_ys)))

        # evaluation
        # shouldn't worry about reevaluating the same ys as the values should be saved in the task cache
        # but could potentially add logic to remove expanded from queue
        print("Finished Generation...Started Eval!")

        start_time = time.perf_counter()

        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values, eval_times = get_values(task, x, new_ys, args.n_evaluate_sample)
        
        eval_time = time.perf_counter()-start_time
        print(f"Node Eval Time: {eval_time} seconds")
        
        # selection
        print("Finished Eval...Started Selection...")

        start_time = time.perf_counter()

        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        selection_time = time.perf_counter()-start_time()
        print(f"Selection Time: {selection_time} seconds")

        # log
        print("Finished Selection...Logging...")
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n-- generate times --: {generate_times}\n-- eval times --: {eval_times}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys, 'generate_times': generate_times, 'eval_times': eval_times})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}