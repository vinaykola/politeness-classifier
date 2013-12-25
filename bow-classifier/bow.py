import csv
from collections import defaultdict
import cPickle as pickle
import numpy as np
from operator import itemgetter
import nltk.data
from nltk import word_tokenize
from sklearn import svm, cross_validation

wiki_requests = []
test_wiki_requests = []

request_count = 0

# Load the test data and pickle it

with open('../Stanford_politeness_corpus/wikipedia.annotated.csv') as corpus:
	inputReader = csv.reader(corpus, quotechar='"')
	request_count = 0
	for row in inputReader:
		break
	for row in inputReader:
		tempdict = {}
		tempdict['text'] = row[2]
                test_wiki_requests.append(tempdict)
		request_count += 1

# Load the annotated data

with open('../Stanford_politeness_corpus/wikipedia.annotated.csv') as targetFile:
	inputReader = csv.reader(targetFile, quotechar='"')
	'''
	14 rows.
	row[2] -> request
	row[13] -> score
	'''
	request_count = 0
	for row in inputReader:
            break
	for row in inputReader:
            request_count += 1
            tempdict = {}
            tempdict['text'] = row[2]
            tempdict['score'] = float(row[13])
            wiki_requests.append(tempdict)
	
print request_count

# Classify the top and bottom quartiles as 1 and 0 respectively, as per the binary notion of
# politeness in the paper.

binary_wiki_requests = [] # The final set of instances to be used for training

sorted_wiki_requests = sorted(wiki_requests,key=itemgetter('score'))
size =  len(sorted_wiki_requests)
quart_size = size/4 + 1 # 1089 instances, as mentioned in the paper

# Set the bottom quartile as 0
request_count = 0
for request in sorted_wiki_requests:
    request['class'] = 0
    binary_wiki_requests.append(request)
    request_count += 1
    if request_count == quart_size:
        break

# Set the top quartile as 1
request_count = 0
for request in reversed(sorted_wiki_requests):
    request['class'] = 1
    binary_wiki_requests.append(request)
    request_count += 1
    if request_count == quart_size:
        break

	
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

request_count = 0
for request in binary_wiki_requests:
	request['sentences'] = sent_detector.tokenize(request['text'].strip())
	request['words'] = []
	for sentence in request['sentences']:
                request['words'] += word_tokenize(sentence)
        request_count += 1

word_dict = defaultdict(int)
	
for request in binary_wiki_requests:
    for word in request['words']:
        word_dict[word] += 1

filtered_word_dict = {k:v for k,v in word_dict.iteritems() if v >= 10}

print len(filtered_word_dict)

words = [k for k,v in filtered_word_dict.iteritems()]

wiki_requests_data = np.zeros((len(binary_wiki_requests),len(words)))
wiki_requests_target = np.zeros(2178)

request_count = 0
for request in binary_wiki_requests:
    for word in request['words']:
        if word in words:
            idx = words.index(word)
            wiki_requests_data[request_count][idx] += 1
    wiki_requests_target[request_count] = request['class']
    request_count += 1

loo = cross_validation.LeaveOneOut(len(binary_wiki_requests))
print len(loo)

accuracy = 0.0
count = 0

for train_index, test_index in loo:
    X_train, X_test = wiki_requests_data[train_index], wiki_requests_data[test_index]
    y_train, y_test = wiki_requests_target[train_index], wiki_requests_target[test_index]

    clf = svm.LinearSVC()
    clf.fit(X_train,y_train)

    prediction = clf.predict(X_test)
    
    if prediction == y_test:
        accuracy += 1.0

    print count, y_test, prediction
    count += 1

print accuracy/2178


