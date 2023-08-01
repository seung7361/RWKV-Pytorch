from RWKV import RWKVLMHeadModel

import numpy as np
import torch

from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import deepspeed

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", pad_token='<|pad|>', eos_token='<|endoftext|>')

### hyperparameters

vocab_size = tokenizer.vocab_size + 2
n_layers = 12
hidden_size = 1024
learning_rate = 5e-5
batch_size = 32
max_len = 256
num_epochs = 5

model = RWKVLMHeadModel(vocab_size, hidden_size, n_layers).cuda()

print("model parameters: {:_}".format(sum(p.numel() for p in model.parameters())))
print("model init done")

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# initialize deepspeed

deepspeed.init_distributed(dist_backend='nccl')

# load the dataset

wikitext = load_dataset('wikitext', 'wikitext-103-v1')['train']['text']
train_dataset = []
for sentence in wikitext:
    if len(sentence) > 15:
        train_dataset.append("{} {} {}".format(tokenizer.eos_token, sentence, tokenizer.eos_token))

model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(model=model, optimizer=optimizer,
                                                    model_parameters=model.parameters(), config='ds_config.json',
                                                    training_data=train_dataset)

# training loop

for epoch in range(num_epochs):
    model_engine.train()

    # save checkpoint
    torch.save(model.state_dict(), 'checkpoint.pt')
    print('saving checkpoint complete.')
    step = 0

    pbar = tqdm(train_dataloader)
    pbar.set_description("epoch: {}, loss: {:.4f}".format(epoch, 0.0))
    for batch in pbar:
        model_engine.zero_grad()
        model_engine.train()
        
        batch = tokenizer(batch, padding='max_length', max_length=max_len,
                          truncation=True, return_tensors='pt').input_ids.cuda()

        input_ids, labels = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        outputs = model_engine(input_ids)

        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        
        model_engine.backward(loss)
        model_engine.step()

        pbar.set_description("epoch: {}, loss: {:.4f}".format(epoch, loss.item()))

        step += 1
        if step % 10 == 0:
            model_engine.eval()
            sentence = "{}".format(tokenizer.eos_token)
            token = tokenizer(sentence, return_tensors='pt')['input_ids'].cuda()
            print(
                tokenizer.decode(
                    model.generate(token, max_len=64)[0].cpu().numpy(),
                    skip_special_tokens=True
                )
            )

        if step % 100 == 0:
            torch.save(model.state_dict(), 'checkpoint.pt')
            print('saving checkpoint complete.')

### save model

torch.save(model.state_dict(), 'model.pt')
print('saving model complete.')