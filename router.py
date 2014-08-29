from flask import Flask,jsonify,render_template,request,send_from_directory

import os

import settings, classifiers

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/<path:filename>')
def send_foo(filename):
    return send_from_directory(os.path.join(settings.APP_STATIC,'paper'), 'BA_ValentinDeyringer.pdf')

@app.route('/_get_sentiment', methods=['POST'])
def get_sentiment():
    
    #get input text
    text = request.form.get("txt")
    
    #color settings
    settings.colPos = request.form.get('colorPositive')
    settings.colNeg = request.form.get('colorNegative')
    settings.colMix = request.form.get('colorMixed')
    settings.colNegpol = request.form.get('colorNegation')
    
    #training data
    settings.trainingData = request.form.get('trainingData')
    
    #additional words for rules
    settings.posWordsAdd = set([s.strip().lower() for s in request.form.get('posWords').split(',')])
    settings.negWordsAdd = set([s.strip().lower() for s in request.form.get('negWords').split(',')])
    
    #set features for classifiers
    classifiers.set_feats(text)
    
    #compute results; tuples returned
    ml = classifiers.classify_ml() #(res_bayes,text_bayes,res_svm,text_svm)
    bayes = ml[0]
    svm = ml[2]
    rules = classifiers.classify_rule()
    
    #overall result
    res = ''
    
    if bayes == svm == rules[0]:
        res = bayes
    else:
        res = 'Mixed'
        
    #'translate' res to html; no color for neutral result needed
    if res == 'Mixed':
        res = '<span style="color:' + settings.colMix + '">' + res + '</span>'
    elif res == 'Positive':
        res = '<span style="color:' + settings.colPos + '">' + res + '</span>'
    elif res == 'Negative':
        res = '<span style="color:' + settings.colNeg + '">' + res + '</span>'
    
    #return results jsonified
    return jsonify(result=res,bayesHTML=ml[1],svmHTML=ml[3],rulesHTML=rules[1])


