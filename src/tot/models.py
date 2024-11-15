import os
import openai
from openai import OpenAI
import backoff 
import torch
import transformers

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

#######################
### Model Inference ###
#######################

def inference_model(prompt, model, tokenizer, temperature=0.7, max_tokens=1000, n=5, stop=None) -> list:
    '''
    Driver function for model inference.
    '''
    if model: #will modify this later to include support for other variations
        return hf_model(model, tokenizer, prompt, temperature, max_tokens, n, stop)
    else:
        model = "gpt-4o"
        messages = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}]
        return chatgpt(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def hf_model(model, tokenizer, prompt, temperature=0.7, max_tokens=1000, n=5, stop=None):
    """
    Given a model (Huggingface) and input tokens, generate an output
    """
    outputs = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #tokenize inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(device)
    # print(inputs)
    model.to(device)

    while n > 0:
        cnt = min(n, 20) 
        n -= cnt

        #actual generation
        out = model.generate(**inputs, temperature=temperature, max_new_tokens=max_tokens, num_return_sequences=cnt) #might add stopping criteria depending on heuristics experimentation

        for o in out:
            string_answer = tokenizer.decode(o)
            outputs.extend([string_answer])

    return outputs

# @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def chatgpt(model, messages, temperature=0.7, max_tokens=1000, n=5, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    client = OpenAI()

    while n > 0:
        cnt = min(n, 20) 
        n -= cnt

        res = client.chat.completions.create(model=model, messages=messages, temperature=temperature, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])

        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens

    return outputs
    
def gpt_usage(backend="gpt-4o"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
