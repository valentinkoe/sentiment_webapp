import os
# __file__ refers to the file settings.py 
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')

colPos = '#00000'
colNeg = '#00000'
colMix = '#00000'
colNegpol = '#00000'

trainingData = ''
posWordsAdd = set()
negWordsAdd = set()
