# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:08:59 2018

@author: DIOTJA
"""

import keras
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import string
import nltk
import re
import sklearn
import pickle
from keras.models import Sequential
from keras.layers import Embedding, Activation
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.layers import LSTM, Conv1D, Flatten, Dense, Dropout
from keras.layers import MaxPooling1D, BatchNormalization
from keras.optimizers import adam
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

translator = str.maketrans('', '', '''!"$%&\()*+,-./:;<=>?@[\\]^'_`{|}~''')

def process_tweet(tweet):
    tweet = tweet.replace('&amp;', 'and')
    tweet = tweet.replace('\x92', "'")
    tweet_split = [wd for wd in tweet.lower().split()]
    for k in range(len(tweet_split)):
        if  '@' in tweet_split[k]:
            tweet_split[k] = 'mention'
        if 'https:' in tweet_split[k]:
            tweet_split[k] = 'hyperlink'
    tweet = ' '.join(tweet_split)
    return ' '.join(tweet.translate(translator).split())
    
tok_path = 'tokenizer.pickle'
tf_path = 'tfidf.pickle'
model_path = 'model_weights.pickle'
log_reg_path = 'log_reg.pickle'
    
with open(tok_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(log_reg_path, 'rb') as handle:
    clf = pickle.load(handle)
    
with open(tf_path, 'rb') as handle:
    vectorizer = pickle.load(handle)
    
with open(model_path, 'rb') as handle:
    model_weights = pickle.load(handle)
    
###############################################################################    
#model = keras.models.load_model(model_path)
TIME_STEPS = 20
SINGLE_ATTENTION_VECTOR = False

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) 
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul',
                                 mode='mul')
    return output_attention_mul

embedding_layer = Embedding(100972,
                            100,
                            input_length=30,
                            trainable=True)

inputs = Input((30,))
emb_v = embedding_layer(inputs)

out = Conv1D(64,
                 3,
                 activation='elu',
                 padding='valid',
                 strides=1)(emb_v)

out = Conv1D(64,
                 3,
                 activation='elu',
                 padding='valid',
                 strides=1)(out)

out = Conv1D(64,
                 3,
                 activation='elu',
                 padding='valid',
                 strides=1)(out)

out = Conv1D(64,
                 3,
                 activation='elu',
                 padding='valid',
                 strides=1)(out)

out = Conv1D(64,
                 3,
                 activation='elu',
                 padding='valid',
                 strides=1)(out)

out = BatchNormalization()(out)
out = Bidirectional(LSTM(512, return_sequences=True))(out)
out = attention_3d_block(out)
out = Flatten()(out)
out = Dense(1,  activation='sigmoid')(out)

model = Model(inputs, out)

sgd = adam(0.0005)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.set_weights(model_weights)
###############################################################################
def predict(tweet):
    
    tweet_p = process_tweet(tweet)
    ttp = tokenizer.texts_to_sequences([tweet_p])
    ttpt = pad_sequences(ttp, maxlen=30, padding='post', truncating='post')
    t_idf = vectorizer.transform([tweet_p])
    
    return (( 0.5 * model.predict(ttpt)[0,0]) +
            ( 0.5 * clf.predict_proba(t_idf)[0][1]))
