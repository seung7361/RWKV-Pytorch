from RMKV import *

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
                                          pad_token='[PAD]', unk_token='[UNK]',
                                          eos_token='[EOS]', sos_token='[SOS]')
vocab_size = len(tokenizer)
SOS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[SOS]')
EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[EOS]')
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids('[PAD]')
print('[SOS]', SOS_TOKEN_ID)
print('[EOS]', EOS_TOKEN_ID)
print('[PAD]', PAD_TOKEN_ID)


model = RWKVModel(vocab_size=vocab_size, n_layers=3, hidden_size=2048)