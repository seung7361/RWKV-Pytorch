import numpy as np
import torch
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = tokenizer.vocab_size + 1
SOS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[SOS]')
EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[EOS]')
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids('[PAD]')
print('[SOS]', SOS_TOKEN_ID)
print('[EOS]', EOS_TOKEN_ID)
print('[PAD]', PAD_TOKEN_ID)

class RWKVSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, layer_id):
        super().__init__()

        self.layer_id = layer_id

        self.hidden_size = hidden_size

        self.time_decay = torch.nn.Parameter(torch.ones(hidden_size))
        self.time_first = torch.nn.Parameter(torch.ones(hidden_size))
        self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1)) # shifts data one step to the right

        self.time_mix_key = torch.nn.Parameter(torch.ones(1, 1, hidden_size))
        self.time_mix_value = torch.nn.Parameter(torch.ones(1, 1, hidden_size))
        self.time_mix_receptance = torch.nn.Parameter(torch.ones(1, 1, hidden_size))
        
        self.ln_out = torch.nn.Linear(hidden_size, hidden_size)

        self.receptance = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        shifted = self.time_shift(x)
        _, seq_len, _ = x.shape
        # shifted shape == x shape == (batch_size, seq_len, hidden_size)


        key = x * self.time_mix_key + shifted * (1 - self.time_mix_key)
        value = x * self.time_mix_value + shifted * (1 - self.time_mix_value)
        receptance = self.receptance(x * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance))

        key = self.key( key )
        value = self.value( value )
        receptance = self.sigmoid( receptance )

        # key shape == value shape == receptance shape == (batch_size, seq_len, hidden_size)

        # initialization before the for loop
        num_state = torch.zeros_like(key[:, 0])
        den_state = torch.zeros_like(key[:, 0])

        output = torch.zeros_like(x)
        # num_state shape == den_state shape == (batch_size, hidden_size)
        time_decay = torch.exp(-torch.exp(self.time_decay))

        # calculate wkv_t
        for cur in range(seq_len):
            current_key = key[:, cur, :] # shape == (batch_size, hidden_size)
            current_value = value[:, cur, :] # shape == (batch_size, hidden_size)

            output[:, cur] = (num_state + torch.exp(self.time_first + current_key) * current_value) \
                              / (den_state + torch.exp(self.time_first + current_key))

            # update num_state and den_state for the next loop
            num_state = time_decay * num_state + torch.exp(current_key) * current_value
            den_state = time_decay * den_state + torch.exp(current_key)
    

        output = self.ln_out(receptance * output)
        return output

class RWKVFeedForward(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1)) # shifts data one step to the right
        self.time_mix_key = torch.nn.Parameter(torch.ones(1, 1, hidden_size))
        self.time_mix_receptance = torch.nn.Parameter(torch.ones(1, 1, hidden_size))

        self.key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        shifted = self.time_shift(x)

        key = self.key( x * self.time_mix_key + shifted * (1 - self.time_mix_key) )
        receptance = self.sigmoid(self.receptance( x * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance) ))
        value = self.value(key)

        return receptance * value

class RWKVBlock(torch.nn.Module):
    def __init__(self, layer_id, hidden_size):
        super().__init__()

        self.layer_id = layer_id

        self.ln1 = torch.nn.LayerNorm(hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size)

        self.attention = RWKVSelfAttention(hidden_size=hidden_size, layer_id=layer_id)
        self.ffn = RWKVFeedForward(hidden_size=hidden_size)
    
    def forward(self, x):
        attention = self.attention(self.ln1(x)) # attention block
        x = attention + x # residual connection

        ffn = self.ffn(self.ln2(x)) + x # feed forward network
        x = ffn + x # residual connection

        # x shape == attention shape == ffn shape == (batch_size, seq_len, hidden_size)

        return x

class RWKVModel(torch.nn.Module):
    def __init__(self, vocab_size, n_layers, hidden_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.ln_in = torch.nn.LayerNorm(hidden_size)
        self.layers = torch.nn.ModuleList([RWKVBlock(layer_id=i, hidden_size=hidden_size) for i in range(n_layers)])
        self.ln_out = torch.nn.LayerNorm(hidden_size)

        self.linear = torch.nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        out = self.embedding(input_ids)
        out = self.ln_in(out)
        for idx, layer in enumerate(self.layers):

            out = layer(out)

        
        out = self.ln_out(out)
        out = self.linear(out)

        return out

    def generate(self, text, max_length=64, top_p=0.9):
        input_ids = tokenizer(text, return_tensors='pt')['input_ids'].cuda()
        
        for i in range(max_length):
            outputs = self(input_ids)
            next_token_logits = outputs[0][-1, :]
            
            # top-p sampling
            # apply a softmax to convert the logits to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # sort the probabilities in descending order and compute their cumulative sum
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # sample from the filtered distribution
            next_token_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1).item()
            
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            
            # stop when end-of-text token is generated
            if next_token_id == PAD_TOKEN_ID or next_token_id == EOS_TOKEN_ID:
                break
        
        return tokenizer.decode(input_ids[0])
