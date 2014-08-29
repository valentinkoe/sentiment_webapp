#basic python imports
from __future__ import unicode_literals
import codecs, os, cPickle
from copy import copy

#text processing imports
import nltk
from textblob_aptagger import PerceptronTagger #much faster than nltk tagger

#app imports
import settings

#table of features as list of tuples
#(original_text,negated,negated_lemma,start_index,length)
FEATS = []
prev_text = ''


#######################################
########## text processing ############
#######################################

neg_pol_items = ['not', "n't", 'no']
#TODO: better negative polarity items, better negation handling ('never','none','nothing',neither + nor, ...)
delims = '?.,!:;'
#TODO: parenthesis, quotation and other possible delimiters?
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
tagger = PerceptronTagger()

def set_feats(text):

    global FEATS, prev_text
    
    #don't do computation, if user enters same text twice; saves at least some time for this rare case
    #TODO: memoize would maybe be nice
    if prev_text == text:
        return
    prev_text = text
    
    #reset FEATS from previous computations
    FEATS = []
    
    add_index = 0
    sents = nltk.sent_tokenize(text)
    
    for sent in sents:
    
        words = nltk.word_tokenize(sent)
        #TODO: emoticon handling: twokenize.tokenize(sent)
        #TODO: nltk tokenization methods seem to fail to split "something like this.Too bad!" (no space after dot)
        lemmas = lemmatize(sent.lower()) #maps because of same tokenization method!
        negated = False
        prev = None
        prev_lem = None
        pprev = None
        pprev_lem = None
        
        #get all relevant features
        for i in range(len(words)):
            unigram = 'not_' + words[i].lower() if negated else words[i].lower()
            lemma = 'not_' + lemmas[i] if negated else lemmas[i]
            FEATS.append((words[i],unigram,lemma,i+add_index,1)) #add unigram
            if prev:
                bigram = prev + ' ' + unigram
                bigram_lem = prev_lem + ' ' + lemma
                FEATS.append((words[i-1] + ' ' + words[i],bigram,bigram_lem,i+add_index-1,2)) #add bigram
                if pprev:
                    trigram = pprev + ' ' + bigram
                    trigram_lem = pprev_lem + ' ' + bigram_lem
                    FEATS.append((words[i-2] + ' ' + words[i-1] + ' ' + words[i],trigram,trigram_lem,i+add_index-2,3)) #add trigram
                pprev = prev
                pprev_lem = prev_lem
            prev = unigram
            prev_lem = lemma
            
            #change negation flag
            if words[i] in neg_pol_items: #do not use any() -- words containing 'no'
                negated = not negated

            if any(d in words[i] for d in delims): #reset negation flag
                negated = False
        
        add_index += len(words)

#lemmatize a text
def lemmatize(text):
    tagged = tagger.tag(text) #also uses nltk.word_tokenize TODO: combination possible to avoid double call?
    ret = []
    for tagged_word in tagged:
        #handle 're and n't
        if (tagged_word[0] == "'re"):
            ret.append('be')
        elif (tagged_word[0] == "n't"):
            ret.append('not')
        else:
            lemma = ''
            if tagged_word == ("'s",'VBZ'):
                lemma = 'be'
            else:
                lemma = lemmatizer.lemmatize(tagged_word[0],wordnet_pos(tagged_word[1]))
            ret.append(lemma)
    return ret

#translate POS Tags to WordNet Tags for correct lemmatization
def wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    return nltk.corpus.wordnet.NOUN #default value for wordnet lemmatizer anyway

#######################################
###### rule based classification ######
#######################################

#dictionaries for rule based classification
pos_rule = set()
neg_rule = set()
mix_rule = set()

#load inquierer dictionary for rule classification
with codecs.open(os.path.join(settings.APP_ROOT,'resources','inquirer_pos.txt'),'r','utf-8') as f:
    for w in f.readlines():
        w = w.strip()
        pos_rule.add(w)
        neg_rule.add('not_' + w)

with codecs.open(os.path.join(settings.APP_ROOT,'resources','inquirer_neg.txt'),'r','utf-8') as f:
    for w in f.readlines():
        w = w.strip()
        neg_rule.add(w)
        pos_rule.add('not_' + w)
    
    #fill mixed dictionary and remove mixed words from pos and neg
    mix_rule = pos_rule.intersection(neg_rule)
    pos_rule = pos_rule - mix_rule
    neg_rule = neg_rule - mix_rule

def classify_rule():
    
    #build dictionaries to actually check
    check_pos = copy(pos_rule) #shallow copies of sets are fine
    check_neg = copy(neg_rule)
    #add user defined words
    #Ignoring neg pol items!! #TODO: offer possibility to edit neg_pol_items
    settings.posWordsAdd = settings.posWordsAdd - set(neg_pol_items)
    settings.negWordsAdd = settings.negWordsAdd - set(neg_pol_items)
    check_pos.update([w for w in settings.posWordsAdd])
    check_neg.update(['not_' + w for w in settings.posWordsAdd])
    check_neg.update([w for w in settings.negWordsAdd])
    check_pos.update(['not_' + w for w in settings.negWordsAdd])
    #update mixed dict
    check_mix = mix_rule
    check_mix.update(check_pos.intersection(check_mix))
    check_pos = check_pos - check_mix
    check_neg = check_neg - check_mix
    
    neg_count = 0
    pos_count = 0
    mix_count = 0
    text = ''
    
    #build html text
    for f in FEATS:
        if f[4] == 1: #only unigrams by now
            if f[2] in check_pos:
                text += '<span style="color:' + settings.colPos + '">' + f[0] + '</span> '
                pos_count += 1
            elif f[2] in check_neg:
                text += '<span style="color:' + settings.colNeg + '">' + f[0] + '</span> '
                neg_count += 1
            elif f[2] in check_mix:
                text += '<span style="color:' + settings.colMix + '">' + f[0] + '</span> '
                mix_count += 1
            else:
                if f[0] in neg_pol_items:
                    text += '<span style="color:' + settings.colNegpol + '">' + f[0] + '</span> '
                else:
                    text += f[0] + ' '
    
    res = 'Neutral'
    if pos_count > neg_count:
        res = 'Positive'
    elif neg_count > pos_count:
        res = 'Negative'
    elif pos_count != 0: #same number of pos and neg words, but at least one
        res = 'Mixed'
    
    if res == 'Positive':
        text = '<span style="color:' + settings.colPos + '">Positive (+' + str(pos_count - neg_count) + ')</span><br>' + text.strip()
    elif res == 'Negative':
        text = '<span style="color:' + settings.colNeg + '">Negative (-' + str(neg_count - pos_count) + ')</span><br>' + text.strip()
    elif res == 'Mixed':
        text = '<span style="color:' + settings.colNeg + '">Mixed (-' + str(pos_count + mix_count) + ')</span><br>' + text.strip()
    else:
        text = 'Neutral (0)<br>' + text.strip()
    
    return (res,text)

#######################################
########## ml classification ##########
#######################################

#load classifiers
svm_classifier_imdb = cPickle.load(open(os.path.join(settings.APP_ROOT,'resources', 'classifier_svm_imdb.pickle')))
svm_classifier_yelp = cPickle.load(open(os.path.join(settings.APP_ROOT,'resources', 'classifier_svm_yelp.pickle')))
svm_classifier_comb = cPickle.load(open(os.path.join(settings.APP_ROOT,'resources', 'classifier_svm_comb.pickle')))
bayes_classifier_imdb = cPickle.load(open(os.path.join(settings.APP_ROOT,'resources', 'classifier_bayes_imdb.pickle')))
bayes_classifier_yelp = cPickle.load(open(os.path.join(settings.APP_ROOT,'resources', 'classifier_bayes_yelp.pickle')))
bayes_classifier_comb = cPickle.load(open(os.path.join(settings.APP_ROOT,'resources', 'classifier_bayes_comb.pickle')))

#load dicts of features extracted on training data
fcount_imdb = cPickle.load(open(os.path.join(settings.APP_ROOT,'resources','fcount_imdb.pickle')))
fcount_yelp = cPickle.load(open(os.path.join(settings.APP_ROOT,'resources','fcount_yelp.pickle')))
fcount_comb = cPickle.load(open(os.path.join(settings.APP_ROOT,'resources','fcount_comb.pickle')))

#check f[1] for for feat in FEATS

def classify_ml():
    
    #get classifiers and feature count of selected training data set
    bayes_classifier = None
    svm_classifier = None
    feat_count_pos = None
    feat_count_neg = None
    if settings.trainingData == 'imdb':
        bayes_classifier = bayes_classifier_imdb
        svm_classifier = svm_classifier_imdb
        feat_count_pos = fcount_imdb[0]
        feat_count_neg = fcount_imdb[1]
    elif settings.trainingData == 'yelp':
        bayes_classifier = bayes_classifier_yelp
        svm_classifier = svm_classifier_yelp
        feat_count_pos = fcount_yelp[0]
        feat_count_neg = fcount_yelp[1]
    elif settings.trainingData == 'comb':
        bayes_classifier = bayes_classifier_comb
        svm_classifier = svm_classifier_comb
        feat_count_pos = fcount_comb[0]
        feat_count_neg = fcount_comb[1]
    else:
        raise Exception('Something went terribly wrong!')
    
    global FEATS
    
    rel_feats = {}
    #map indices of words to counts of corresponding feature appearing in one of the fcounts
    pos_indices = {}
    neg_indices = {}
    words = []
    
    #extract relevant features and indices for vizualisig in html output
    for f in FEATS:
        if f[1] in feat_count_pos: #f[1] are negated uni- bi- and trigrams
            rel_feats[f[1]] = 1 #each feature only once; no incrementing but directly setting count to one
            for j in range(f[3],f[3]+f[4]): #indices of words that feature contains
                try:
                    pos_indices[j] += feat_count_pos[f[1]]
                except KeyError:
                    pos_indices[j] = feat_count_pos[f[1]]
        if  f[1] in feat_count_neg:
            rel_feats[f[1]] = 1
            for j in range(f[3],f[3]+f[4]):
                try:
                    neg_indices[j] += feat_count_neg[f[1]]
                except KeyError:
                    neg_indices[j] = feat_count_neg[f[1]]
        if f[4] == 1:
            words.append(f[0])
    
    #no feats to test on
    if len(rel_feats) == 0:
        neut_text = 'Neutral (no matching features found)<br>' + ' '.join(f[0] for f in FEATS if f[4] == 1)
        return ('Neutral',neut_text,'Neutral',neut_text)
    
    text = ''
    for i in range(len(words)):
        if words[i] in neg_pol_items:
            text += '<span style="color:' + settings.colNegpol + '">' + words[i] + '</span> '
        elif i in pos_indices or i in neg_indices:
            if pos_indices[i] > neg_indices[i]: #word appears in positive features more often
                text += '<span style="color:' + settings.colPos + '">' + words[i] + '</span> '
            elif neg_indices[i] > pos_indices[i]: #word appears in negative features more often
                text += '<span style="color:' + settings.colNeg + '">' + words[i] + '</span> '
            else: #counts the same -- mixed
                text += '<span style="color:' + settings.colMix + '">' + words[i] + '</span> '
        else:
            text += words[i] + ' '
    
    
    #classify according to relevant features
    res_bayes = bayes_classifier.classify(rel_feats)
    res_svm = svm_classifier.classify(rel_feats)
    
    #complete html output; ml only has levels of positive and negative, no mixed or neutral
    #bayes
    if res_bayes == 'Positive':
        text_bayes = '<span style="color:' + settings.colPos + '">Positive</span><br>' + text.strip()
    elif res_bayes == 'Negative':
        text_bayes = '<span style="color:' + settings.colNeg + '">Negative</span><br>' + text.strip()
    #svm
    if res_svm == 'Positive':
        text_svm = '<span style="color:' + settings.colPos + '">Positive</span><br>' + text.strip()
    elif res_svm == 'Negative':
        text_svm = '<span style="color:' + settings.colNeg + '">Negative</span><br>' + text.strip()
    
    return (res_bayes,text_bayes,res_svm,text_svm)
