from collections import Counter
import numpy
import torch

text = open('politic_50k.txt').read()
text2 = open('politic_1k.txt').read()

cnt = Counter(text.split(' '))
chars = sorted(list(set(text)))
top50k = [x[0] for x in cnt.most_common(15000)]
stoi = {top50k[x]: x for x in range(len(top50k))}
for x in range(len(chars)):
	c = chars[x]
	if c not in stoi:
		stoi[c] = len(stoi)
space = stoi[' ']
itos = {i:c for c,i in stoi.items()}

encode = lambda s: [stoi[x] for x in s]
decode = lambda s: ''.join([itos[x.item()] for x in s])

# def decode(s):
# 	out = []
# 	begin = 0
# 	for x in s:
# 		x = x.item()
# 		if x != space:
# 			out.append(itos[x])
# 			if begin == 0:
# 				out.append(' ')
# 		else:
# 			begin = 0 if begin == 1 else 1
# 			out.append(' ')
# 	return ''.join(out)

def create():
	seq = [stoi['\n']]
	mx = 0
	for doc in text2.split('\n'):
		if doc == '':
			continue

		for token in doc.split():
			if token in stoi:
				seq.append(stoi[token])
			else:
				for c in token:
					seq.append(stoi[c])
			seq.append(stoi[' '])
		seq.append(stoi['\n'])
	numpy.save('politic_1k.npy', numpy.array(seq))

create()

loaded_array = torch.from_numpy(numpy.load('politic_1k.npy'))

print(loaded_array.shape)
print(decode(loaded_array[10:1000]))
