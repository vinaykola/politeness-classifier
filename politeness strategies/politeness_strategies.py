import re
import csv
from collections import defaultdict
import cPickle as pickle
import numpy as np
from operator import itemgetter
from sklearn import svm, cross_validation
#from stanford_core_nlp import StanfordCoreNLPWrapper
from collections import defaultdict
import time
import jsonrpc

import jsonrpclib
from simplejson import loads
server = jsonrpclib.Server("http://ec2-54-213-165-168.us-west-2.compute.amazonaws.com/:8080")
#server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),jsonrpc.TransportTcpIp(addr=("http://ec2-54-213-165-168.us-west-2.compute.amazonaws.com/", 8080)))
server = jsonrpclib.Server("http://127.0.0.1:8080")

hedges=["think","thought","thinking",
      "almost",
      "apparent", "apparently", "appear", "appeared", "appears", "approximately", "around",
      "assume", "assumed", "certain amount", "certain extent", "certain level", "claim",
      "claimed", "doubt", "doubtful", "essentially", "estimate",
      "estimated", "feel", "felt", "frequently", "from our perspective", "generally", "guess",
      "in general", "in most cases", "in most instances", "in our view", "indicate", "indicated",
      "largely", "likely", "mainly", "may", "maybe", "might", "mostly", "often", "on the whole",
      "ought", "perhaps", "plausible", "plausibly", "possible", "possibly", "postulate",
      "postulated", "presumable", "probable", "probably", "relatively", "roughly", "seems",
      "should", "sometimes", "somewhat", "suggest", "suggested", "suppose", "suspect", "tend to",
      "tends to", "typical", "typically", "uncertain", "uncertainly", "unclear", "unclearly",
      "unlikely", "usually", "broadly", "tended to", "presumably", "suggests",
      "from this perspective", "from my perspective", "in my view", "in this view", "in our opinion",
      "in my opinion", "to my knowledge", "fairly", "quite", "rather", "argue", "argues", "argued",
      "claims", "feels", "indicates", "supposed", "supposes", "suspects", "postulates"]

negativewords=[]
fin=open('liu-negative-words.txt')
for line in fin:
    negativewords.append(line.strip())

negativewords=set(negativewords)

positivewords=[]
fin=open('liu-positive-words.txt')
for line in fin:
    positivewords.append(line.strip())

positivewords=set(positivewords)


getdeptag=lambda p: p.split("(")[0]
getleft=lambda p: re.findall(r"([-\w]+)-(\d+)",p)[0][0].lower()
getleftpos=lambda p: int(re.findall(r"([-\w]+)-(\d+)",p)[0][1])
getright=lambda p: re.findall(r"([-\w]+)-(\d+)",p)[1][0].lower()
getrightpos=lambda p: int(re.findall(r"([-\w]+)-(\d+)",p)[1][1])

def removenumbers(parse_element):
  return re.sub(r"\-(\d+)","",parse_element)

def getrel(document):
  return [removenumbers(w.lower()) for w in document['parse']]

please=lambda p: len(set([getleft(p),getright(p)]).intersection(["please"]))>0 and 1 not in [getleftpos(p),getrightpos(p)]
please.__name__="Please"
pleasestart=lambda p: (getleftpos(p)==1 and getleft(p)=="please") or (getrightpos(p)==1 and getright(p)=="please")
pleasestart.__name__="Please start"
hashedges=lambda p:   getdeptag(p)=="nsubj" and  getleft(p) in hedges
hashedges.__name__="Hedges"
deference=lambda p: (getleftpos(p)==1 and getleft(p) in ["great","good","nice","good","interesting","cool","excellent","awesome"]) or (getrightpos(p)==1 and getright(p) in ["great","good","nice","good","interesting","cool","excellent","awesome"]) 
deference.__name__="Deference"
gratitude=lambda p: getleft(p).startswith("thank") or getright(p).startswith("thank") or "(appreciate, i)" in removenumbers(p).lower()
gratitude.__name__="Gratitude"
apologize=lambda p: getleft(p) in ["sorry","woops","oops"] or getright(p) in ["sorry","woops","oops"] or removenumbers(p).lower()=="dobj(excuse, me)" or removenumbers(p).lower()=="nsubj(apologize, i)" or removenumbers(p).lower()=="dobj(forgive, me)"
apologize.__name__="Apologizing"
groupidentity=lambda p: len(set([getleft(p),getright(p)]).intersection(["we","our","us","ourselves"]))>0
groupidentity.__name__="1st person pl."
firstperson=lambda p: 1 not in [getleftpos(p),getrightpos(p)] and len(set([getleft(p),getright(p)]).intersection(["i","my","mine","myself"]))>0
firstperson.__name__="1st person"
secondperson_start=lambda p: (getleftpos(p)==1 and getleft(p) in ["you","your","yours","yourself"]) or (getrightpos(p)==1 and getright(p) in ["you","your","yours","yourself"]) 
secondperson_start.__name__="2nd person start"
firstperson_start=lambda p: (getleftpos(p)==1 and getleft(p) in ["i","my","mine","myself"]) or (getrightpos(p)==1 and getright(p) in ["i","my","mine","myself"]) 
firstperson_start.__name__="1st person start"
hello=lambda p: (getleftpos(p)==1 and getleft(p) in ["hi","hello","hey"]) or (getrightpos(p)==1 and getright(p) in ["hi","hello","hey"]) 
hello.__name__="Indirect (greeting)"
really=lambda p: (getright(p)=="fact" and getdeptag(p)=="prep_in") or removenumbers(p) in ["det(point, the)","det(reality, the)","det(truth, the)"] or len(set([getleft(p),getright(p)]).intersection(["really","actually","honestly","surely"]))>0  
really.__name__="Factuality"
why=lambda p: (getleftpos(p) in [1,2] and getleft(p) in ["what","why","who","how"]) or (getrightpos(p) in [1,2] and getright(p) in ["what","why","who","how"])
why.__name__="Direct question"
conj=lambda p: (getleftpos(p) in [1] and getleft(p) in ["so","then","and","but","or"]) or (getrightpos(p) in [1] and getright(p) in ["so","then","and","but","or"])
conj.__name__="Direct start"
btw=lambda p: getdeptag(p)=="prep_by" and getright(p)=="way" and getrightpos(p)==3
btw.__name__="Indirect (btw)"
secondperson=lambda p: 1 not in [getleftpos(p),getrightpos(p)] and len(set([getleft(p),getright(p)]).intersection(["you","your","yours","yourself"]))>0
secondperson.__name__="2nd person"


def getparse(document):
  return document['parse']

def select(document,processfun,unittest):
    for l in processfun(document):
        try:
            testres=unittest(l)
            if testres:
                return True
        except Exception, e:
            print e,l
            testres=False
    return False



funset=[please,pleasestart,btw,hashedges,really,deference,gratitude,apologize,groupidentity,firstperson,secondperson,secondperson_start,firstperson_start,hello,why,conj]

def getPolitenessFeaturesFromParse(parse, verbose=False):
  #nlp = StanfordCoreNLPWrapper()
  #parse_string_list = nlp.getDependencyParseStringList(parse)

  
  parse_string_list = [sentence[u'indexeddependencies'] for sentence in parse[u'sentences']][0]
  indexeddependencies = [parse[u'sentences'][sentence_id]['indexeddependencies'] for sentence_id in xrange(len(parse[u'sentences']))]
  #print indexeddependencies
  
  parse_string_list = []
  for sentence in indexeddependencies:
      deps = []
      for dep in sentence:
          deps.append(dep[0] + '(' + dep[1] + ', ' + dep[2] + ')')
      parse_string_list.append(deps)
  #print parse_string_list

  
  # parse_string_list looks like: [[u'root(ROOT-0, help-2)', u'discourse(help-2, Please-1)', u'dobj(help-2, me-3)', u'prt(help-2, out-4)', u'tmod(help-2, here-5)'], [u'root(ROOT-0, Hi-1)', u'appos(Hi-1, woops-3)', u'nsubj(appreciate-7, I-5)', u'advmod(appreciate-7, really-6)', u'dep(Hi-1, appreciate-7)', u'poss(honesty-10, your-8)', u'amod(honesty-10, unlikely-9)', u'dobj(appreciate-7, honesty-10)']]
  featurelist_dict = defaultdict(list)
  if len(parse['sentences']) == 0:
    # Make sure we still output something.
    all_politeness_features = ['feature_politeness_==1st_person==', 'feature_politeness_==1st_person_pl.==', 'feature_politeness_==1st_person_start==', 'feature_politeness_==2nd_person==', 'feature_politeness_==2nd_person_start==', 'feature_politeness_==Apologizing==', 'feature_politeness_==Deference==', 'feature_politeness_==Direct_question==', 'feature_politeness_==Direct_start==', 'feature_politeness_==Factuality==', 'feature_politeness_==Gratitude==', 'feature_politeness_==Hedges==', 'feature_politeness_==INDICATIVE==', 'feature_politeness_==Indirect_(btw)==', 'feature_politeness_==Indirect_(greeting)==', 'feature_politeness_==Please==', 'feature_politeness_==Please_start==', 'feature_politeness_==SUBJONCTIVE==', 'feature_politeness_=HASHEDGE=','feature_politeness_=HASPOSITIVE=','feature_politeness_=HASNEGATIVE=']
    result = {}
    for f in all_politeness_features:
      result[f] = 0
    return result
  for sentence_id in xrange(len(parse['sentences'])):
    # Fake dictionary with parse and text
    d = {}
    d['parse'] = parse_string_list[sentence_id]
    d['text'] = parse['sentences'][sentence_id]['text']
    verbose = True
    if verbose:
      print '\t\t', sentence_id, d['text']
      print '\t\t', sentence_id, d['parse']

    for fun in funset:
      featurelist_dict["=="+fun.__name__.replace(" ","_")+"=="].append(int(select(d,getparse,fun)))
    featurelist_dict["==SUBJONCTIVE=="].append(int("could you" in d['text'].lower() or "would you" in d['text'].lower()))
    featurelist_dict["==INDICATIVE=="].append(int("can you" in d['text'].lower() or "will you" in d['text'].lower()))
    featurelist_dict["=HASHEDGE="].append((len(set(d['text'].lower().split()).intersection(hedges))>0))
    featurelist_dict["=HASPOSITIVE="].append(int(len(positivewords.intersection(positivewords))>0))
    featurelist_dict["=HASNEGATIVE="].append(int(len(negativewords.intersection(negativewords))>0))
  # Aggregate all binary features by OR/MAX. Alternatively, one could also sum them up.
  aggregated_features = {}
  for k,v in featurelist_dict.items():
    aggregated_features['feature_politeness_'+k] = max(v)
  return aggregated_features
  
def getWordsFromParse(parse):
    word_list = []
    words =  [sentence[u'words'] for sentence in parse[u'sentences']] 
    for sentence in words:
        word_list += [word[0] for word in sentence]
    return word_list

# Load the annotated data
def loadData():

  wiki_requests = []  
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
          for i in xrange(1,21):
              tempdict[i] = 0
          wiki_requests.append(tempdict)
  return wiki_requests  

def binarizeRequests(wiki_requests):
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
  return binary_wiki_requests  


def train(binary_wiki_requests):
  with open('parses.pkl','rb') as handle:
      parses = pickle.load(handle)

  request_count = 0
  for request in binary_wiki_requests:
      #request['parse'] = loads(server.parse(request['text']))
      #parses[request['text']] = request['parse']
      request['parse'] = parses[request['text']]
      request_count += 1
      print request_count


  request_count = 0
  for request in binary_wiki_requests:
      request['words'] = getWordsFromParse(request['parse'])
      #print request['words']
      print request_count
      request_count += 1


# Add the unigram features
	
  word_dict = defaultdict(int)
  for request in binary_wiki_requests:
      for word in request['words']:
          word_dict[word] += 1

# Filter out unigrams with frequency < 10
  filtered_word_dict = {k:v for k,v in word_dict.iteritems() if v >= 10}
  words = [k for k,v in filtered_word_dict.iteritems()]

  wiki_requests_data = np.zeros((len(binary_wiki_requests),len(words)+21))
 
  wiki_requests_target = np.zeros(len(binary_wiki_requests))

  request_count = 0
  for request in binary_wiki_requests:
      
      f = sorted(getPolitenessFeaturesFromParse(request['parse']).iteritems())

      for i in xrange(21):
          wiki_requests_data[request_count][len(words)+i] = f[i][1]
          if wiki_requests_data[request_count][len(words)+i] == True:
              wiki_requests_data[request_count][len(words)+i] = 1
          if wiki_requests_data[request_count][len(words)+i] == False:
              wiki_requests_data[request_count][len(words)+i] = 0
      
      for word in request['words']:
          if word in words:
              idx = words.index(word)
              wiki_requests_data[request_count][idx] += 1

      wiki_requests_target[request_count] = request['class']

      request_count += 1
  return wiki_requests_data,wiki_requests_target    
    
if __name__ == '__main__':
  text = 'Please help me out here. Hi, woops, I really appreciate your unlikely honesty.'
  #nlp = StanfordCoreNLPWrapper()
  #parse = nlp.getParse(text)
  
  parse = loads(server.parse(text))
  
  f = getPolitenessFeaturesFromParse(parse)

  getWordsFromParse(parse)
  
  wiki_requests = loadData()

  request_count = 0

  binary_wiki_requests = binarizeRequests(wiki_requests)


	

  with open('parses.pkl','rb') as handle:
      parses = pickle.load(handle)

  request_count = 0
  for request in binary_wiki_requests:
      #request['parse'] = loads(server.parse(request['text']))
      #parses[request['text']] = request['parse']
      request['parse'] = parses[request['text']]
      request_count += 1
      print request_count


  request_count = 0
  for request in binary_wiki_requests:
      request['words'] = getWordsFromParse(request['parse'])
      #print request['words']
      print request_count
      request_count += 1


# Add the unigram features
	
  word_dict = defaultdict(int)
  for request in binary_wiki_requests:
      for word in request['words']:
          word_dict[word] += 1

# Filter out unigrams with frequency < 10
  filtered_word_dict = {k:v for k,v in word_dict.iteritems() if v >= 10}
  words = [k for k,v in filtered_word_dict.iteritems()]

  wiki_requests_data = np.zeros((len(binary_wiki_requests),len(words)+21))
 
  wiki_requests_target = np.zeros(len(binary_wiki_requests))

  request_count = 0
  for request in binary_wiki_requests:
      
      f = sorted(getPolitenessFeaturesFromParse(request['parse']).iteritems())

      for i in xrange(21):
          wiki_requests_data[request_count][len(words)+i] = f[i][1]
          if wiki_requests_data[request_count][len(words)+i] == True:
              wiki_requests_data[request_count][len(words)+i] = 1
          if wiki_requests_data[request_count][len(words)+i] == False:
              wiki_requests_data[request_count][len(words)+i] = 0
      
      for word in request['words']:
          if word in words:
              idx = words.index(word)
              wiki_requests_data[request_count][idx] += 1

      wiki_requests_target[request_count] = request['class']

      request_count += 1

  loo = cross_validation.LeaveOneOut(len(binary_wiki_requests))

  accuracy = 0.0
  count = 0

  print len(binary_wiki_requests)

  clf = svm.LinearSVC()
  
  start_time = time.time()

  for train_index, test_index in loo:
      X_train, X_test = wiki_requests_data[train_index], wiki_requests_data[test_index]
      y_train, y_test = wiki_requests_target[train_index], wiki_requests_target[test_index]


      clf.fit(X_train,y_train)

      prediction = clf.predict(X_test)
    
      if prediction == y_test:
          accuracy += 1.0

      print count, y_test, prediction
      count += 1

  print accuracy/len(binary_wiki_requests)

  end_time = time.time()
  print("Elapsed time was %g seconds" % (end_time - start_time))
  
  '''
  start_time = time.time()

  was_right = cross_validation.cross_val_score(clf,wiki_requests_data, wiki_requests_target, cv=loo)
  total_acc = np.mean(was_right)

  end_time = time.time()

  print total_acc
  '''
  
  #  print("Elapsed time was %g seconds" % (end_time - start_time))

