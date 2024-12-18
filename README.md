## Overview
This project is an exploratory extension of [Tree-of-Thought (Yao et al., 2023)](https://github.com/princeton-nlp/tree-of-thought-llm) (referred to as ToT in this repository).

We were specifically driven to extend upon ToT by the following:

**Problem Motivation**: Complex math problems can be difficult for LLMs to solve reliably. At the same time, ToT has shown promise for helping LLMs tackle complex reasoning problems.

**Problem Goal**: Explore the efficacy and efficiency of solving complex math problems with Tree-of-Thought.

(**NOTE:** We originally intended to adapt the ToT source code repository for this project but due to a lack of fit, we eventually deforked and wrote our own repository implementations from scratch instead. However, please note that the deforking process has still preserved the commit history and Github contributor list from when the repository was forked. Please look to recent commits to see the code and contributors relevant to this specific project)

## Quickstart
From the root directory, run ```python run_bench.py``` followed by your desired parameters. 

E.g., ```python run_bench.py --backend llama --num_test_samp 50 --temperature 0.7``` to run the unmodified Llama3.2-3B-Instruct model on the first 50 (filtered) samples from the MATH test dataset. 

A list of all available args is listed below.

Notes:
- Please note that loading in the original Llama3.2-3b-Instruct model requires logging into huggingface, as the model sits in a gated repo.

## Args
Please note all args are optional. There are several args used in this project that control the nature of each ToT run. For readability, they are grouped by purpose as follows:

### Type of model
- ```backend``` indicates which model to use.
- ```quantize``` supports three options: 
    - "qat" for the (int8) quantization-aware-trained Llama model, 
    - "ptq_int4" for the int4 (weights only) post-training-quantized Llama model, and
    -  "ptq_int8" for the int8 (weights only) post-training-quantized Llama model.
    - all other values will result in the code running the unmodified Llama3.2-3b-Instruct model (unless the LoRA flag is present. See below)
- ```lora``` . If this flag is present, the LoRA version of the Llama model will be used. (note: QLoRA or other combination versions of quantization and LoRA are not supported in this code. If both args.quantize and args.lora are not None, the code will run the selected quantized model and ignore the lora flag.)

### Model generation configurations
- ```temperature``` modifies the token probabilities. Default is 0.7 to encourage meaningfully diverse 'branches' from each parent node in the tree.
- ```max_new_tokens``` sets the ceiling for the number of new tokens that can be generated per model call. The default is 100.

### Tree structure and traversal
- ```a_star``` . If this flag is present, the code will run ToT using the A* traversal method.
- ```q_size``` controls the maximum size of the priority queue used to help implement the A* traversal method in this repository. The top q-size ranked proposals are selected after each proposal-evaluation iteration and carried forward into the next. Default is 5. 
- ```depth``` controls the "depth" of the ToT tree and, effectively, the maximum number of attempts a model is allowed to reach its final solution. Default is 3.
- ```breadth``` controls the branching factor of each node in the tree and, effectively, the number of proposals generated by the model per call. Default is 3.
- ```greedy_n``` controls the number of best proposals selected from each iteration for the next. E.g., greedy_n = 1 will have the code select the top-1 ranked proposal from the current propose-evaluate iteration to carry into the next. For consistency with the original source code repository, default is 1. (Note: if the traversal method is A*, proposals from the priority queue will also be carried forward)
- ```concurrent``` was an argument originally set up to help support a distributed learning/multiple-gpu run. Due to gpu access issues, this parameter is no longer used but has been left in place in case access changes.

### Other

- ```num_test_samp``` the number of samples from the filtered MATH test set to use. If "None", 2000 samples will be run. Default is 50. For replicability, comparability, and consistency, the test samples are not shuffled. I.e. num_test_samp=n selects the first n samples in the test set after filtering.
- ```num_repeat``` the number of times to repeat the same trial. This argument is available for replicability and consistency, as the temperature is expected to be high in order to ensure sufficiently different proposals/children nodes from each parent node.

## Repository Structure
The main branch contains all code necessary to run Llama3.2-3B-Instruct with ToT on the processed MATH dataset.

For GPT-4o, please switch to the branch **TBA**