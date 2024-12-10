import torch
import torch.quantization
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import multiprocessing
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

class benchmark_dataset(torch.utils.data.Dataset):
  '''formats the data for dataloader'''

  def __init__(self, input, labels, tokenizer, filter_n=150):
    '''constructor. input samples and output labels'''

    self.input = input
    self.labels = labels
    self.tokenizer = tokenizer

    self.filter_len(filter_n)

  def filter_len(self, n):

    new_input = []
    new_label = []

    for q, a in zip(self.input, self.labels):

      matches = re.findall(r'\\boxed{([^}]*)}', a)
      if len(matches) <= 0:
        continue

      tk_len_q = len(tokenizer(str(q), return_tensors='pt')['input_ids'][0])
      tk_len_a = len(tokenizer(str(a), return_tensors='pt')['input_ids'][0])

      if tk_len_q <= n and tk_len_a <= n:
        new_input.append(q)
        new_label.append(a)

    print(f"""
    Len of Original Input: {len(self.input)}
    Len of Original Labels: {len(self.labels)}
    Len of New_Input: {len(new_input)}
    Len of New_Label: {len(new_label)}

    Sample Input, Label: {new_input[1], new_label[1]}

    """)

    self.input = new_input
    self.labels = new_label

  def __len__(self):
    return len(self.input)

  def __getitem__(self, idx):

    return {"question": self.input[idx], "answer": self.labels[idx]}

def format_for_mm(question, choices):
  '''
  Formats questions and choices into one multiple-choice-question string
  '''
  return [f"""Choose the choice that best answer the following question:
  Question:
  {q.strip()}
  Choices:
  {c}
  """
  for q, c in zip(question, choices)]

def collate_fn_qat(batch):

    # Now collate into mini-batches
    inputs = tokenizer([i['question'] for i in batch], return_tensors='pt', padding='max_length', truncation=True, max_length=150)
    # labels = tokenizer([str(i['answer']) for i in batch], return_tensors='pt', padding='max_length', truncation=True, max_length=65)
    labels = tokenizer([str(i['answer']) for i in batch], return_tensors='pt', padding='max_length', truncation=True, max_length=150)

    # labels = [ele[-100:] for ele in labels['input_ids']]

    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'label': labels['input_ids']}
