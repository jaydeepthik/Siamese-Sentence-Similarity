# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:27:15 2019

@author: jaydeep thik
"""

import pandas as pd
import numpy as np
from nltk import TreebankWordTokenizer
import os
import string
from keras.preprocessing.sequence import pad_sequences

from keras import layers
from keras import Input
from keras.models import Model, model_from_json
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt




def load_glove(dim):
    glove_dir = "F:/machine learning/code/NLP_word_embedding/data"
    
    embedding = {}
    f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'), encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coeff = np.asarray(values[1:], dtype = 'float32')
        embedding[word]=coeff
    f.close()
    return embedding


def get_vector(word):
    if word in embeddings:
        return embeddings[word]

def vectorise_sentence(sentence):
    tokens = tokenizer.tokenize(sentence)
    vector = []
    for token in tokens:
        if token not in string.punctuation:
            token_vector = get_vector(token)
            if token_vector is not None:
                vector.append(token_vector)
    return vector

def vectorize_df(df):
     vectors_a = [vectorise_sentence(sentence) for sentence in df.sentence_A ]   
     vectors_b = [vectorise_sentence(sentence) for sentence in df.sentence_B ]
     scores = ((df.relatedness_score-1)/4).tolist()
     return vectors_a, vectors_b, scores

def test_custom(sentence1, sentence2, max_len,model):
    vectors_a = [vectorise_sentence(sentence1)]   
    vectors_a = pad_sequences(vectors_a, padding='post', dtype='float32', maxlen=max_len)
    
    vectors_b = [vectorise_sentence(sentence2)]
    vectors_b = pad_sequences(vectors_b, padding='post', dtype='float32', maxlen=max_len)
    
    normalized_score = model.predict([vectors_a, vectors_b])
    return normalized_score*4 + 1



embeddings =load_glove(50)

tokenizer = TreebankWordTokenizer()
    
if __name__ == "__main__":
    
    file_path = "SICK/SICK.txt"
    
    df = pd.read_csv(file_path, sep='\t')[['sentence_A', 'sentence_B', 'relatedness_score', 'SemEval_set']]
    
    
    train_data = df[df['SemEval_set']=="TRAIN"]
    dev_data = df[df['SemEval_set']=="TRIAL"]
    test_data = df[df['SemEval_set']=="TEST"]
    
    #get train data and similarity score
    train_a_vectors, train_b_vectors, train_org_scores = vectorize_df(train_data)
    
    train_max_a_length = len(max(train_a_vectors, key=len))
    train_max_b_length = len(max(train_b_vectors, key=len))
    
    max_len = max([train_max_a_length, train_max_b_length])

    train_a_vectors = pad_sequences(train_a_vectors, padding='post', dtype='float32', maxlen=max_len)
    train_b_vectors = pad_sequences(train_b_vectors, padding='post', dtype='float32', maxlen=max_len)
    
    #get dev data and similarity score
    dev_a_vectors, dev_b_vectors, dev_org_scores = vectorize_df(dev_data)
    
    dev_a_vectors = pad_sequences(dev_a_vectors, padding='post', dtype='float32', maxlen=max_len)
    dev_b_vectors = pad_sequences(dev_b_vectors, padding='post', dtype='float32', maxlen=max_len)
    
    #get test data and similarity score
    test_a_vectors, test_b_vectors, test_org_scores = vectorize_df(test_data)
    
    test_a_vectors = pad_sequences(test_a_vectors, padding='post', dtype='float32', maxlen=max_len)
    test_b_vectors = pad_sequences(test_b_vectors, padding='post', dtype='float32', maxlen=max_len)
    
    #------------------------------------------------------------------------------
    
    
    input_dim = 50
    h_units = 100
    
    # LSTM siamese model 
    
    lstm = layers.LSTM(50, unit_forget_bias=True, kernel_initializer='he_normal', kernel_regularizer='l2', name='lstm_layer')
    
    left_input =Input(shape=(None, input_dim), name='input_1')
    left_output = lstm(left_input)
    
    right_input =Input(shape=(None, input_dim), name='input_2')
    right_output = lstm(right_input)
    
    manhattan = lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), keepdims=True, axis=1))
    
    merged = layers.merge([left_output, right_output], mode = manhattan, output_shape = lambda x: (x[0][0],1), name='l1_distance')
    #predictions = layers.Dense(1,activation='linear', name='similarity_layer')(merged)
    
    model = Model([left_input, right_input],merged)
    
    model.compile(optimizer=Adadelta(), loss='mse', metrics=['mse'])
    # Train the model 
    history = model.fit([train_a_vectors, train_b_vectors], train_org_scores, validation_data=([dev_a_vectors, dev_b_vectors], dev_org_scores), epochs = 50, batch_size = 64)#, callbacks = callbacks)
    