import numpy as np
import torch

class RWKVSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, layer_id):
        super().__init__()

        self.layer_id = layer_id

        self.hidden_size = hidden_size

        self.time_decay = torch.nn.Parameter(torch.exp(-torch.ones(hidden_size)))
        self.time_first = torch.nn.Parameter(torch.exp(-torch.ones(hidden_size)))
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


        key = x * self.time_mix_key + shifted * self.time_mix_key
        value = x * self.time_mix_value + shifted * self.time_mix_value
        receptance = self.receptance(x * self.time_mix_receptance + shifted * self.time_mix_receptance)

        key = self.key( key )
        value = self.value( value )
        receptance = self.sigmoid( receptance )

        # key shape == value shape == receptance shape == (batch_size, seq_len, hidden_size)

        # initialization before the for loop
        num_state = torch.ones_like(key[:, 0])
        den_state = torch.ones_like(key[:, 0])

        output = torch.zeros_like(x)
        # num_state shape == den_state shape == (batch_size, hidden_size)

        # calculate wkv_t
        for cur in range(seq_len):
            current_key = key[:, cur] # shape == (batch_size, hidden_size)
            current_value = value[:, cur] # shape == (batch_size, hidden_size)

            output[:, cur] = ((num_state + self.time_decay * current_key * den_state) / (den_state + self.time_decay * current_key))

            # update num_state and den_state for the next loop
            num_state = self.time_decay * num_state + torch.exp(current_key) * current_value
            den_state = self.time_decay * den_state + torch.exp(current_key)

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

        key = self.key( x * self.time_mix_key + shifted * self.time_mix_key )
        receptance = self.sigmoid(self.receptance( x * self.time_mix_receptance + shifted * self.time_mix_receptance ))
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
    
    def forward(self, input_ids):
        out = self.embedding(input_ids)
        out = self.ln_in(out)
        for idx, layer in enumerate(self.layers):

            out = layer(out)

        
        out = self.ln_out(out)

        return out

