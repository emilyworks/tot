import os
import openai
import backoff 

from transformers import AutoTokenizer, AutoModelForCausalLM

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

def inference_model(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None, vllm=False, quant=False) -> list:
    '''
    Driver function for model inference.
    '''
    if model == "llama_3.2" and vllm:
        return llama_32(prompt, quant, vllm, temperature, max_tokens, n, stop)
    else:
        messages = [{"role": "user", "content": prompt}]
        return chatgpt(prompt, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def llama_32(prompt, quant, vllm, temperature, max_tokens, n, stop): #will add vllm support later
    '''
    Use llama3.2 for inference
    '''
    if quant:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8")
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = hf_model(model, inputs, temperature, max_tokens, n, stop)
    
    return outputs

def hf_model(model, input_tokens, temperature=0.7, max_tokens=1000, n=1, stop=None):
    """
    Given a model (Huggingface) and input tokens, generate an output
    """
    outputs = []

    while n > 0:
        cnt = min(n, 20) #never generate more than 20 outputs per same input
        n -= cnt
        outputs = model.generate(**input_tokens, temperature=temperature, max_new_tokens=max_tokens, num_return_sequences=cnt) #might add stopping criteria depending on heuristics experimentation
        #need to take a look at the specific output format once i get access to the gated repo
        #need to outputs.extend()

    return outputs

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20) #never generate more than 20 outputs per same input
        n -= cnt
        
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])

        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]

    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
