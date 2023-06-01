from RWKV import *

import numpy as np
import torch

from tqdm import tqdm
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
import deepspeed

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

model = RWKVModel(vocab_size=vocab_size, n_layers=n_layers, hidden_size=hidden_size).cuda()

print("model parameters: {:_}".format(sum(p.numel() for p in model.parameters())))
print("model init done")

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# initialize deepspeed

deepspeed.init_distributed(dist_backend='nccl')

# load the dataset

train_dataset = torch.load('./dataset/train_dataset.pt')
val_dataset = torch.load('./dataset/val_dataset.pt')

model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=scheduler,
                                                    model_parameters=model.parameters(), config='ds_config.json',
                                                    training_data=train_dataset)

# training loop

for epoch in range(num_epochs):
    model_engine.train()

    # save checkpoint
    torch.save(model.state_dict(), './checkpoint/model_epoch_{}.pt'.format(epoch))
    print('saving checkpoint complete.')

    pbar = tqdm(train_dataloader)
    for batch in pbar:
        model_engine.zero_grad()
        batch = batch.cuda()

        input_ids, labels = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        outputs = model_engine(input_ids)

        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        
        model_engine.backward(loss)
        model_engine.step()

        pbar.set_description("loss: {:.4f}".format(loss.item()))

### save model

torch.save(model.state_dict(), 'model.pt')
print('saving model complete.')