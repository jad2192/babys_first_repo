# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:39:55 2018

@author: DIOTJA
"""
import numpy as np
import string
import nltk
import re
import sklearn
import pickle
from nltk.corpus import stopwords


stop_wds = set(stopwords.words('english'))
translator = str.maketrans('', '', '''!"$%&\()*+,-./:;<=>?@[\\]^'_`{|}~''')
lemmatizer = nltk.WordNetLemmatizer()

def process_tweet(tweet):
    tweet = tweet.replace('&amp;', 'and')
    tweet = tweet.replace('\x92', "'")
    tweet_split = [wd for wd in tweet.lower().split()]
    for k in range(len(tweet_split)):
        if  '@' in tweet_split[k]:
            tweet_split[k] = ''
        if 'https:' in tweet_split[k]:
            tweet_split[k] = 'hyperlink'
    tweet = ' '.join(tweet_split)
    return ' '.join(tweet.translate(translator).split())
    

tf_path = 'foo.pickle'
log_reg_path = 'foo.pickle'
    

# Loading pre-trained vectorizer and logistic regression model.

with open(log_reg_path, 'rb') as handle:
    clf = pickle.load(handle)
    
with open(tf_path, 'rb') as handle:
    vectorizer = pickle.load(handle)
    

def predict(tweet):
    
    tweet_p = process_tweet(tweet)
    t_idf = vectorizer.transform([tweet_p])
    
    return clf.predict_proba(t_idf)[0]
