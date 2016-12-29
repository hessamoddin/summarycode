from __future__ import division, print_function, absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
from mlxtend.preprocessing import one_hot
import sklearn.mixture.gmm as gm
from tempfile import TemporaryFile
from sklearn import mixture
import numpy as np
import scipy.io as matreader
from sklearn.cross_validation import train_test_split
from scipy.stats import multivariate_normal
import pprint
import logging
#import cv2
from os import listdir
import os.path
from os.path import isfile, join, splitext
from os import path
import tflearn
import pylab,os
import imageio
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import daisy
import matplotlib.cbook as cbook
from sklearn.cluster import KMeans
import math
import logging
import pandas as pd
import random
from random import sample
import csv
import cPickle as pickle
import numpy as np
from glove import Corpus, Glove
import array
import scipy.sparse as sp
from glove.glove import check_random_state





 
# Define functions
def generate_training_corpus(num_sentences,
                             vocabulary_size=30000,
                             sentence_min_size=2,
                             sentence_max_size=30,
                             seed=None):

    rs = check_random_state(seed)

    for _ in range(num_sentences):
        sentence_size = rs.randint(sentence_min_size,
                                   sentence_max_size)
        yield [str(x) for x in
               rs.randint(0, vocabulary_size, sentence_size)]


def build_coocurrence_matrix(sentences,window=10):

    dictionary = {}
    rows = []
    cols = []
    data = array.array('f')

 
    for sentence in sentences:
        for i, first_word in enumerate(sentence):
            first_word_idx = dictionary.setdefault(first_word,
                                                   len(dictionary))
            for j, second_word in enumerate(sentence[i:i + window + 1]):
                second_word_idx = dictionary.setdefault(second_word,
                                                        len(dictionary))

                distance = j

                if first_word_idx == second_word_idx:
                    pass
                elif first_word_idx < second_word_idx:
                    rows.append(first_word_idx)

                    cols.append(second_word_idx)
                    data.append(np.double(1.0) / distance)
                else:
                    rows.append(second_word_idx)
                    cols.append(first_word_idx)
                    data.append(np.double(1.0) / distance)

    return sp.coo_matrix((np.double(data), (rows, cols)),
                         shape=(len(dictionary),
                                len(dictionary)),
                         dtype=np.double).tocsr().tocoo()
 
def _reproduce_input_matrix(glove_model):

    wvec = glove_model.word_vectors
    wbias = glove_model.word_biases

    out = np.dot(wvec, wvec.T)

    for i in range(wvec.shape[0]):
        for j in range(wvec.shape[0]):
            if i == j:
                out[i, j] = 0.0
            elif i < j:
                out[i, j] += wbias[i] + wbias[j]
            else:
                out[i, j] = 0.0

    return np.asarray(out)
 
 

bovw_shape=(3,5)
bovw_bins = np.random.randint(9,13, size=bovw_shape)
bovw_weights = np.random.randint(2, size=bovw_shape)
 


print('Bovw bins')
print(bovw_bins)
print('Bovw weights')
print(bovw_weights)
 




dictionary = {}
rows = []
cols = []
data = array.array('f')
 
k=0 
#print(bovw_bins)

for frame in bovw_bins:
        for i, first_word in enumerate(frame):
            first_word_idx = dictionary.setdefault(first_word,
                                                   len(dictionary))
            w1=bovw_weights[k,i]                                    
            for j, second_word in enumerate(frame):
                second_word_idx = dictionary.setdefault(second_word,
                                                        len(dictionary))
                w2=bovw_weights[k,j]            
                distance = 1
                w=w1*w2

                if first_word_idx == second_word_idx:
                    pass
                elif first_word_idx < second_word_idx:
                    rows.append(first_word_idx)

                    cols.append(second_word_idx)
                    data.append(np.double(w*np.double(1.0) / distance))
                else:
                    rows.append(second_word_idx)
                    cols.append(first_word_idx)
                    data.append(np.double(w*np.double(1.0) / distance))
        k=k+1
     
                    
 

x=sp.coo_matrix((data, (rows, cols)),
                         shape=(len(dictionary),
                                len(dictionary)),
                         dtype=np.double).tocsr().tocoo()      
print(dictionary)           
dic_keys=dictionary.keys()
dic_values=dictionary.values()

              
xarr=x.toarray()                         
xarr/=np.amax(xarr)
print("coocurancem matrix")
print(xarr)
xsparse=sp.coo_matrix(xarr)   

glove_model = Glove(no_components=5, learning_rate=0.05)
glove_model.fit(xsparse,
                    epochs=500,
                    no_threads=2)


new_word_representation=glove_model.word_vectors
print("New word representation")
print(new_word_representation)

print("*** Query ***")
query=10
query_pos=dic_values[dic_keys.index(query)]

target=12
target_pos=dic_values[dic_keys.index(target)]
sim=np.dot(glove_model.word_vectors[query_pos],glove_model.word_vectors[target_pos])
print(sim)
