import os

#directories
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')

#color settings
colPos = '#00000'
colNeg = '#00000'
colMix = '#00000'
colNegpol = '#00000'

#other settings
trainingData = ''
posWordsAdd = set()
negWordsAdd = set()
