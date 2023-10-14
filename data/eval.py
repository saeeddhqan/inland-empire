from collections import Counter
import numpy
import torch


punc_pos = 'j0' # Punctuation part of speech
line_pos = '0' # Line part of speech

text = open('pol_clean.txt').read()
text_pos = open('pol_clean_pos.txt').read()
ptoi = {i:x for x,i in enumerate(list(set(text_pos.split())))}
ptoi['0'] = len(ptoi)

cnt = Counter(text.split(' '))
chars = sorted(list(set(text)))
top15k = [x[0] for x in cnt.most_common(20000)]
stoi = {top15k[x]: x for x in range(len(top15k))}
for x in range(len(chars)):
	c = chars[x]
	if c not in stoi:
		stoi[c] = len(stoi)
itos = {i:c for c,i in stoi.items()}

encode = lambda s: [stoi[x] for x in s]
decode = lambda s: ''.join([itos[x.item()] for x in s])


def encode():
	seq = []
	seq_pos = []
	for doc, poss in zip(text.split('\n'), text_pos.split('\n')):
		if doc == '':
			continue
		for token, pos in zip(doc.split(), poss.split()):
			if token in stoi:
				seq.append(stoi[token])
				seq_pos.append(ptoi[pos])
			else:
				for c in token:
					seq.append(stoi[c])
					seq_pos.append(ptoi[pos])
			seq.append(stoi[' '])
			seq_pos.append(ptoi[punc_pos])
		seq.append(stoi['\n'])
		seq_pos.append(ptoi[line_pos])
	numpy.save('pol20k.npy', numpy.array(seq))
	numpy.save('pol20k_pos.npy', numpy.array(seq_pos))

encode()

loaded_array = torch.from_numpy(numpy.load('pol20k.npy'))
loaded_array_pos = torch.from_numpy(numpy.load('pol20k_pos.npy'))

print(loaded_array.shape)
print(loaded_array_pos.shape)
print(decode(loaded_array[10:1000]))
print(loaded_array_pos[10:1000])
