from collections import Counter
import numpy
import torch


a = open('pol_clean.txt').read()
cnt = Counter(a.split(' '))
chars = sorted(list(set(a)))
top15k = [x[0] for x in cnt.most_common(20000)]
stoi = {top15k[x]: x for x in range(len(top15k))}
for x in range(len(chars)):
	c = chars[x]
	if c not in stoi:
		print(len(stoi), c)
		stoi[c] = len(stoi)
itos = {i:c for c,i in stoi.items()}

encode = lambda s: [stoi[x] for x in s]
decode = lambda s: ''.join([itos[x.item()] for x in s])


def encode():
	seq = []
	mx = 0
	al = 0
	ldoc = 0
	for doc in a.split('\n'):
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
	numpy.save('pol20k.npy', numpy.array(seq))

encode()

loaded_array = torch.from_numpy(numpy.load('pol20k.npy'))

print(loaded_array.shape)
print(decode(loaded_array[10:1000]))