import re

with open('lowercased.txt') as fp:
	a = open('pol.txt', 'w')
	mx = 50000
	count = 0
	for i in fp:
		if 'politic' in i:
			a.write(i + '\n')
			count += 1
			if count == mx:
				break
	 

with open('pol.txt') as nf:
	nf = re.sub(r'[^A-z0-9\s\(\)\.,:\|;\-!=+\s]+', '', nf.read())
	nf = re.sub(r'[-]+', ' ', nf)
	nf = re.sub(r'`', ' ', nf)
	nf = re.sub(r'\\', ' ', nf)
	nf = re.sub(r'\d', ' <N> ', nf)
	nf = re.sub(r'\s<N>\s', '$', nf)
	nf = re.sub(r'[$]+', ' $ ', nf)
	nf = re.sub(r'\s+\$\s+', '$', nf)
	nf = re.sub(r'[$]+', ' <N> ', nf)
	nf = re.sub(r'[ ]+', ' ', nf)
	wf = open('pol_clean.txt', 'w')
	for line in nf.split('\n'):
		line = ' '.join(line.split(' ')[:430]).replace('\n', '')
		wf.write(line + '\n')

