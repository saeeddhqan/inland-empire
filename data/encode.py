from transformers import AutoTokenizer 
import torch

model = 'mistralai/Mistral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model)
split = 4

with open('politic4k.txt') as fp:
	seq = torch.tensor(tokenizer.encode(fp.read(), add_special_tokens=False)).to(torch.long)
	torch.save(seq, 'politic4k.pt')
	del seq

with open('politic50k.txt') as fp:
	seq = torch.tensor(tokenizer.encode(fp.read(), add_special_tokens=False)).to(torch.long)
	torch.save(seq, 'politic50k.pt')
	cut = seq.size(0) // split
	for c in range(split + 1):
		torch.save(a[cut * c:(cut + cut * c)].clone(), f'politic50k_p{c}.pt')

