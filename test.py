import torch
from transformers import GPT2TokenizerFast
from RWKV import RWKVLMHeadModel

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", pad_token='<|pad|>', eos_token='<|endoftext|>')
sentence = "{}".format(tokenizer.eos_token)
token = tokenizer(sentence, return_tensors='pt')['input_ids'].cuda()

print(token.shape)

model = RWKVLMHeadModel(50257, 1024, 12).cuda()
model.eval()

print(
    tokenizer.decode(
        model.generate(token, max_len=64)[0].cpu().numpy(),
        skip_special_tokens=True
    )
)