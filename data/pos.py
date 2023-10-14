import spacy

with open('pol_clean.txt') as nf:
	nlp = spacy.load('en_core_web_sm')
	with open('pol_clean_pos.txt', 'w') as wf:
		cnt = 0
		for doc in nlp.pipe(nf.read().split('\n'), disable=['ner', 'lemmatizer'], n_process=4):
			seq = []
			for token in doc:
				seq.append(f"{token.pos_}_{int(token.is_stop)}")
			wf.write(' '.join(seq) + '\n')
			print(cnt)
			cnt += 1

with open('pol_clean_pos.txt') as nf:
	text = nf.read()
	b = list(set(text.split()))
	alph = ''.join([chr(x + 97) for x in range(26)])
	alph += alph.upper()

	rep = {y:alph[x]+y.split('_')[1] for x,y in enumerate(b)}
	for tok in rep:
		text = text.replace(tok, rep[tok])

with open('pol_clean_pos.txt', 'w') as wf:
	wf.write(text)
