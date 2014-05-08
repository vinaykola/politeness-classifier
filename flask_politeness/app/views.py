from app import app
from flask import request
from politeness import politeness_strategies
import json
import cPickle as pickle

from politeness import politeness_strategies

clf,vocab = politeness_strategies.getClassifierandVocab()
print 'loaded classifier'
                                                                                  
@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


@app.route('/politeness',methods=['GET', 'POST'])
def politeness():
    if request.method == 'POST':
        try:
            sentence = request.form[u'sentence']
            pred_class,pred_probs = politeness_strategies.get_proba(sentence,vocab,clf)
            print pred_class
            print pred_probs
            result = {}
            result['class'] = pred_class[0]
            probs = {}
            probs[0] = pred_probs[0][0]
            probs[1] = pred_probs[0][1]
            result['probs'] = probs
            print result
            return json.dumps(result)
        except KeyError:
            return "USing POST. Give me a fucking sentence!"
    else:
        return "Using GET. Fuck off. "
