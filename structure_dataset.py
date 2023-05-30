import numpy as np
import torch

from transformers import BertTokenizerFast
from tqdm import tqdm
from datasets import load_dataset

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', pad_token='[PAD]', eos_token='[EOS]', sos_token='[SOS]')
vocab_size = tokenizer.vocab_size + 1

SOS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[SOS]')
EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[EOS]')
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids('[PAD]')
print('[SOS]', SOS_TOKEN_ID)
print('[EOS]', EOS_TOKEN_ID)
print('[PAD]', PAD_TOKEN_ID)

# load tinystories dataset

dataset = load_dataset("skeskinen/TinyStories-hf")

# tokenize dataset

train_dataset = [tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=1024, truncation=True)['input_ids'].long().squeeze(0) for sentence in tqdm(dataset['train'][:1_000_000]['text'])]
val_dataset = [tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=1024, truncation=True)['input_ids'].long().squeeze(0) for sentence in tqdm(dataset['validation'][:3000]['text'])]

# save the tokenized dataset

torch.save(train_dataset, './dataset/train_dataset.pt')
torch.save(val_dataset, './dataset/val_dataset.pt')