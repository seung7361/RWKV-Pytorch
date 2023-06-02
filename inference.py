import numpy as np
import torch

from transformers import BertTokenizerFast
from RWKV import RWKVModel

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', pad_token='[PAD]', eos_token='[EOS]', sos_token='[SOS]')
vocab_size = tokenizer.vocab_size + 1

SOS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[SOS]')
EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[EOS]')
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids('[PAD]')
print('[SOS]', SOS_TOKEN_ID)
print('[EOS]', EOS_TOKEN_ID)
print('[PAD]', PAD_TOKEN_ID)


### hyperparameters

n_layers = 6
hidden_size = 768
learning_rate = 2e-4
batch_size = 32
num_epochs = 15

num_warmup_steps = 1000
num_training_steps = batch_size * num_epochs

model = RWKVModel(vocab_size=vocab_size, n_layers=n_layers,
                  hidden_size=hidden_size, tokenizer=tokenizer).cuda()
model.load_state_dict(torch.load('./checkpoint/model_epoch_5.pt'))
print('model load done')

sentence = 'Once upon a time'
print(model.generate(sentence))