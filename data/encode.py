from transformers import AutoTokenizer 
import torch

model = 'mistralai/Mistral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model)

with open('politic4k.txt') as fp:
	seq = torch.tensor(tokenizer.encode(fp.read())).to(torch.long)
	torch.save(seq, 'politic4k.pt')
	del seq
with open('politic50k.txt') as fp:
	seq = torch.tensor(tokenizer.encode(fp.read())).to(torch.long)
	torch.save(seq, 'politic50k.pt')

