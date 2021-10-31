from collections import Counter

import sklearn.model_selection
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

data_set = []
classes = ['spam', 'ham']

tokenizer = RegexpTokenizer(r'\w+')
snowball = SnowballStemmer(language='english')
# Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
with open('SMSSpamCollection', 'r', encoding='utf-8') as file:
    for line in file:
        label, msg = line.rstrip().split('\t', 2)
        tokenized_msg = tokenizer.tokenize(msg)
        stemmed_msg = [snowball.stem(token) for token in tokenized_msg]
        data_set.append((label, stemmed_msg))

train_set, test_set = sklearn.model_selection.train_test_split(data_set, train_size=0.8)
print(f'Train len: {len(train_set)}. Test len: {len(test_set)}')

spam = 0
ham = 0

spamVec = []
hamVec = []

Alldict = dict()
Spamdict = dict()
Hamdict = dict()

AlldictSize = 0

Allwords = 0
Spamwords = 0
Hamwords = 0

for i in range(4459):
	if train_set[i][0] == "ham":
		ham+=1
	else:
		spam+=1
	for j in range(len(train_set[i][1])):
		if train_set[i][1][j] in Alldict:
			Alldict[train_set[i][1][j]] +=1
			Allwords+=1
		else:
			Alldict[train_set[i][1][j]]=1
			AlldictSize+=1
			Allwords+=1
		if train_set[i][1][j] in Spamdict and train_set[i][0] == "spam":
			Spamdict[train_set[i][1][j]] +=1
			Spamwords+=1
		elif train_set[i][0] == "spam":
			Spamdict[train_set[i][1][j]] =1
			Spamwords+=1
		if train_set[i][1][j] in Hamdict and train_set[i][0] == "ham":
			Hamdict[train_set[i][1][j]] +=1
			Hamwords+=1
		elif train_set[i][0] == "ham":
			Hamdict[train_set[i][1][j]] =1
			Hamwords+=1




a = 0.01

spamCounter = 0
HamCounter = 0

good = 0
bad = 0

HamP = Hamwords/(Allwords)
SpamP = Spamwords/Allwords

for j in range(1115):
	PHAM = HamP
	PSPAM = SpamP
	for k in range(len(test_set[j][1])):
		if test_set[j][1][k] in Spamdict:
			PSPAM*=(Spamdict[test_set[j][1][k]] + a)/((Spamwords+a) * AlldictSize)
		else:
			PSPAM*= (a/(Spamwords * AlldictSize))
	for k in range(len(test_set[j][1])):
		if test_set[j][1][k] in Hamdict:
			PHAM*=(Hamdict[test_set[j][1][k]] + a)/((Hamwords+a) * AlldictSize)
		else:
			PHAM*=(a/((Hamwords) * AlldictSize))

	if PSPAM >= PHAM:
		spamCounter+=1
		if test_set[j][0] == "spam":
			good+=1
		else:
			bad+=1
	else:
		HamCounter+=1
		if test_set[j][0] == "ham":
			good+=1
		else:
			bad+=1

print("Znalezione Ham:",HamCounter)
print("Znalezione Spam:",spamCounter)

print("Znalezione dobrze sklasyfikowane:",good)
print("Znalezione źle sklasyfikowane:",bad)

print("Accuracy :",good/(good+bad))