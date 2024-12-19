import os
import argparse
import time
from src.tot.prompts.bench import *
import random
import multiprocessing
import ast
import re
import pandas as pd
import json
from openai import OpenAI, OpenAIError, APITimeoutError
from dotenv import load_dotenv
from schemas import *
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken
from functools import partial

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

all_gt = []
all_pred = []

average_solving_time_per_sample = []
average_proposal_time_per_sample = []
average_eval_time_per_sample = []

temp_tuning = {}


def num_tokens_from_string(string, encoding_name="o200k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def a_star_penalty(num, depth, k=0.1):
    return num * np.exp(-k * depth)


def traverse_tree(tree, proposal):
    branch = []
    while True:
        parent = tree[proposal]
        if parent is None:
            break
        branch.append(proposal)
        proposal = parent

    return branch[::-1], len(branch) - 1


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((OpenAIError, APITimeoutError))
)
def gpt_completion_with_retry(
        messages, response_format, n=1, temperature=0.7, model="gpt-4o", max_tokens=4000):
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            n=n,
            response_format=response_format,
            max_tokens=max_tokens,
            timeout=60,  # Add timeout in seconds
        )
        return [choice.message.parsed for choice in response.choices]
    except (OpenAIError, APITimeoutError) as e:
        print(f"OpenAI API error: {str(e)}")
        raise e
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise OpenAIError(f"Unexpected error: {str(e)}")


def gpt_completion(messages, response_format, n=1,
                   temperature=0.7, model="gpt-4o"):
    try:
        return gpt_completion_with_retry(
            messages=messages,
            response_format=response_format,
            n=n,
            temperature=temperature,
            model=model,
        )
    except Exception as e:
        print(f"Failed after retries: {str(e)}")
        return None


def gpt_no_structure(messages, n=1, temperature=0.7, model='gpt-4o'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
    )
    return [choice.message.content for choice in response.choices]


def propose(problem, current, tree, num_return_sequences=3):
    messages = [solve_system_message] + solve_example_1  # + solve_example_2
    messages.append({'role': 'user', 'content': problem})
    tokens_used = sum([num_tokens_from_string(m['content']) for m in messages])

    if current != problem:
        branch, _ = traverse_tree(tree, current)
        while tokens_used + num_tokens_from_string(branch[0]) > 3500:
            branch.pop(0)
        step_messages = [{'role': 'assistant', 'content': step}
                         for step in branch]
        messages.extend(step_messages)

    proposals = gpt_completion(messages, SolutionStep, n=num_return_sequences)

    if proposals is None:
        print("Failed to get proposals - returning empty list")
        return []

    return proposals


def simple_cot(problem):
    messages = [simple_solve_system_message] + simple_solve_example_1
    messages.append({'role': 'user', 'content': problem})
    response = gpt_completion(messages, SimpleSolutionChain, n=1)
    if response is None:
        print("Failed to get solution - returning empty list")
        return None
    return response[0]


def gpt_eval(value_messages):
    responses = []
    for value_message in value_messages:
        response = gpt_completion(
            [value_system_message] + value_example_1 + [value_message], Evaluation)
        if response is None:
            responses.append(None)
        else:
            responses.append(response[0])

    return responses


def create_value_message(problem, steps, proposal):
    num_tokens = num_tokens_from_string(
        value_user_message_template) + num_tokens_from_string(problem) + num_tokens_from_string(proposal)
    steps_string = '\n'.join(steps)
    while num_tokens + num_tokens_from_string(steps_string) > 3500:
        steps_string = '\n'.join(steps_string.split('\n')[1:])

    message = value_user_message_template.format(
        steps=steps_string, problem=problem, proposal=proposal)

    return {'role': 'user', 'content': message}


def value_proposals(problem, proposals, tree, cache, a_star=True):
    valuations = []
    value_messages = []
    depths = []

    noncached_proposals = [p for p in proposals if p not in cache]
    cache_hits = len(proposals) - len(noncached_proposals)

    for p in noncached_proposals:
        branch, depth = traverse_tree(tree, p)
        value_messages.append(create_value_message(problem, branch[:-1], p))
        depths.append(depth)

    responses = gpt_eval(value_messages)
    values_depths = list(zip(responses, depths))

    if not a_star:
        for v, _ in values_depths:
            if v is None:  # Handle failed evaluation
                valuations.append(0.0)
            elif v.likelihood == 'sure' and v.is_solution:
                valuations.append(100.0)
            elif v.likelihood == 'sure' and not v.is_solution:
                valuations.append(1.0)
            elif v.likelihood == 'likely':
                valuations.append(0.5)
            else:
                valuations.append(0.0)
    else:
        for v, d in values_depths:
            if v is None:  # Handle failed evaluation
                valuations.append(0.0)
            elif v.likelihood == 'sure' and v.is_solution:
                valuations.append(100.0)
            elif v.likelihood == 'sure' and not v.is_solution:
                valuations.append(a_star_penalty(1.0, depth=d))
            elif v.likelihood == 'likely':
                valuations.append(a_star_penalty(0.5, depth=d))
            else:
                valuations.append(0.0)

    for p, v in list(zip(noncached_proposals, valuations)):
        cache[p] = v

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


def extract_answer(latex_string):
    pattern = r'boxed{([^}]*)}'
    match = re.search(pattern, latex_string)
    if match:
        return match.group(1)
    return None


def final_eval(solution_text, final_step):
    expected_answer = extract_answer(solution_text)
    model_answer = extract_answer(final_step)
    if expected_answer and model_answer:
        all_pred.append(model_answer)
        all_gt.append(expected_answer)

        try:
            if abs(float(model_answer) - float(expected_answer)) < 1e-6:
                return 1.0
            else:
                return 0.0
        except BaseException:
            if expected_answer == model_answer:
                return 1.0
            else:
                return 0.0
    else:
        return 0.0


def solve(problem, solution_text, args):
    '''
    the main tot run
    '''

    selected = ""
    valuation_cache = {}    # cache for repeated valuations
    tree = {}
    tree[problem] = None
    proposals = []    # persist the queue across iterations
    current = problem

    for i in range(
            args.depth):  # args.depth number of attempts to reach the solution

        # propose next step/solutions per node/prompt
        rpropose = time.perf_counter()

        out = propose(problem, current, tree, args.breadth)

        rpropose = time.perf_counter() - rpropose
        average_proposal_time_per_sample.append(rpropose)

        for o in out:
            proposal = o.reasoning
            if proposal not in tree:
                proposals.append(proposal)
                tree[proposal] = current

        reval = time.perf_counter()
        valuations, cache_hits = value_proposals(
            problem, proposals, tree, valuation_cache, a_star=args.a_star)
        reval = time.perf_counter() - reval
        average_eval_time_per_sample.append(reval)

        # select the best proposal
        val_props = list(zip(proposals, valuations))

        val_props.sort(key=lambda ele: ele[1], reverse=True)
        val_props = val_props[:args.queue_size]
        selected = val_props[0][0]
        if val_props[0][1] == 100.0:
            break

        # remove the selected node from the queue to avoid reeval
        val_props = val_props[1:]
        # update the queue to include the greedy_n highest ranking nodes
        proposals = [vp[0] for vp in val_props]

        current = selected
        print(f"Created {len(tree) - 1} nodes total")

    # compare the proposed final answer vs the ground truth
    # judgement = final_eval(solution_text, selected)
    branch, _ = traverse_tree(tree, selected)
    # print(f"Solution branch: {branch}")
    # print(f"Judgement: {judgement}")

    return branch


def get_test_data():
    with open('inputs.csv', 'r') as f:
        data = pd.read_csv(f)

    return data


def run(args):
    '''
    main run function
    '''

    rtotal = time.perf_counter()
    rsetup = time.perf_counter()

    ### SETUP DATA ###
    test_data = get_test_data()

    rsetup = time.perf_counter() - rsetup
    print(rsetup)

    for idx, row in test_data.iterrows():
        problem = row['question']
        solution_text = row['answer']
        print("-" * 100)
        print(f"Solving problem {idx} of {len(test_data)}")
        print(f"Problem: {problem}")
        print(f"Solution: {solution_text}")

        
        if args.agent == 'tot':    # tot
            rsolve = time.perf_counter()
            solution = solve(problem, solution_text, args)
            rsolve = time.perf_counter() - rsolve
        elif args.agent == 'cot':    # cot
            rsolve = time.perf_counter()
            solution = simple_cot(problem)
            rsolve = time.perf_counter() - rsolve
        else:    # plain gpt-4o
            rsolve = time.perf_counter()
            user_message = {
                'role': 'user', 'content': 'Solve the following math problem: ' + problem}
            solution = gpt_no_structure([no_cot_system_message, user_message],
                                        n=1, temperature=0.7, model='gpt-4o')[0]
            rsolve = time.perf_counter() - rsolve

        print(solution)

        average_solving_time_per_sample.append(rsolve)


def parse_args():
    '''
    Determines the conditions for the run.
    '''
    args = argparse.ArgumentParser()

    # the arguments to use for our purposes
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--depth', type=int, default=15)
    args.add_argument('--breadth', type=int, default=3)
    args.add_argument('--greedy_n', type=int, default=1)
    args.add_argument('--a_star', action='store_true')
    args.add_argument('--queue_size', type=int, default=7)
    args.add_argument('--agent', choices=['zeroshot', 'cot', 'tot'], default='tot', help=f"Choose from: {', '.join(allowed_options)} (default: zeroshot)")
)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)
