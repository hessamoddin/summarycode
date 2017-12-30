from __future__ import print_function
from __future__ import division, print_function, absolute_import


from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.layers.wrappers import  Bidirectional

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.cluster import MiniBatchKMeans, KMeans
from random import sample
from sklearn.metrics.pairwise import cosine_similarity
from glove import Glove
from sklearn import preprocessing
from glove import Corpus
from keras.layers import Dropout


 

import array
import logging
import sklearn.mixture.gmm as gm
import numpy as np
import scipy.sparse as sp
import tables as tb
import warnings
import os,glob
import os.path
import math
import pandas as pd
from collections import OrderedDict,defaultdict

from numpy import dot
from numpy.linalg import norm

from keras.layers import Input
from keras import layers
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
import torchfile
from sklearn.metrics.pairwise import cosine_similarity

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


 
"""
vgg_feature_file_list=[]
for vgg_feature_file in files:
    if os.stat(vgg_feature_file).st_size>100000:
        vgg_feature_file_list.append(vgg_feature_file)
        
        
for vgg_feature_file in vgg_feature_file_list:      
        
"""        
        
        
        
        
"""       
Parameters
"""
bovw_size=20
num_LSTMs=5
train_frac=0.90
longest_allowed_frames=500
batch_size = 1
nb_epochs = 100
hidden_units = 64
learning_rate = 1e-6
clip_norm = 1.0
embedding_size=300
N=4


def word_embedding(sentences,embedding_size,windows_len):
    """
    Verify that the square error diminishes with fitting
    """

     

    corpus_model = Corpus()

    corpus_model.fit(sentences,window=windows_len)

    # Check that the performance is poor without fitting
    glove_model = Glove(no_components=embedding_size, learning_rate=0.05)
    glove_model.fit(corpus_model.matrix,
                    epochs=0,
                    no_threads=2)

    log_cooc_mat = corpus_model.matrix.copy()
    log_cooc_mat.data = np.log(log_cooc_mat.data)
    log_cooc_mat = np.asarray(log_cooc_mat.todense())
    
    
    
    corpus_dict=corpus_model.dictionary
    corpus_inverse_dict=dict(map(reversed, corpus_dict.items()))

        
    

    return glove_model,corpus_dict,corpus_inverse_dict
 





def embedding_func(gridded_words_overall,embedding_size):
    
    """***************
     GLOVE for Video
     ***************"""
     
    
    glove_bins=np.asarray(gridded_words_overall)
    glove_shape=glove_bins.shape
    glove_weights=np.ones((glove_shape))
    #bovw_shape=(3,5)
    #bovw_bins = np.random.randint(9,13, size=bovw_shape)
    #bovw_weights = np.random.randint(2, size=bovw_shape)
    
    
    
    
    #print('Bovw bins')
    #print(bovw_bins)
    #print('Bovw weights')
    #print(bovw_weights)
     
    
    
    
    
    dictionary = {}
    rows = []
    cols = []
    data = array.array('f')
     
    k=0 
    #print(bovw_bins)
    
    for frame in glove_bins:
            for i, first_word in enumerate(frame):
                first_word_idx = dictionary.setdefault(first_word,
                                                       len(dictionary))
                w1=glove_weights[k,i]                                    
                for j, second_word in enumerate(frame):
                    second_word_idx = dictionary.setdefault(second_word,
                                                            len(dictionary))
                    w2=glove_weights[k,j]            
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
        
           
 
    
                  
    xarr=x.toarray()                         
    xarr/=np.amax(xarr)
    xsparse=sp.coo_matrix(xarr)   
    
    glove_model = Glove(no_components=embedding_size, learning_rate=0.05)
    glove_model.fit(xsparse,
                        epochs=500,
                        no_threads=2)
    
    
    new_word_representation=glove_model.word_vectors


    return new_word_representation,dictionary
    

def learn_kmeans_codebook(X, codebook_size=1000, seed=None):
    """ Learn a codebook.
    source: https://github.com/KitwareMedical/TubeTK/blob/master/Base/Python/pyfsa/core/fsa.py
    """
    logger = logging.getLogger()
    logger.info("Learning codebook with %d words ..." % codebook_size)
    # Run vector-quantization
                
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=codebook_size, batch_size=30,
                      n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit(X)

                
    return mbk

def chunks(l, n):
    """
    To split video into different evenly sized set of frames to feed into LSTMs    
    Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

 
def calc_bovw(X, cb):
    """Compute a (normalized) BoW histogram.
   source: https://github.com/rkwitt/pyfsa/blob/master/core/fsa.py
    """
    # Get nr. codewords
    n,d = cb.cluster_centers_.shape
    # Compute closest cluster centers
    assignments = cb.predict(X)
    # Compute (normalized) BoW histogram
    B = range(0,n+1)
    return np.histogram(assignments,bins=B,density=True)[0]

 
def LSTM_Func(hidden_units,vgg_matrix_train,num_cats,learning_rate,y_train,nb_epochs,vgg_matrix_test, y_test):

    model = Sequential()

 
 

    model.add(LSTM(output_dim=hidden_units,activation='relu',input_shape=vgg_matrix_train.shape[1:]))
    model.add(Dense(num_cats))
    model.add(Dropout(0.2))

    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

    model.fit(vgg_matrix_train, y_train, nb_epoch=nb_epochs,verbose=0)

    scores = model.evaluate(vgg_matrix_test, y_test, verbose=0)    
    print('Accuracy:', scores[1])

    ranked_classes_all=np.argsort(-model.predict(vgg_matrix_test),axis=1)

    AP=0
    sum_ind=0
 
    for i in xrange(num_test_samples):
        #predicted_classes=np.argmax(model.predict(vgg_matrix_test),axis=1)
        expected_class=int(labels_arr_test[i])
        predicted_classes=ranked_classes_all[i,:]
        x=(predicted_classes==expected_class)
        r=x*1
        ind=np.where(predicted_classes==expected_class)[0][0]
        sum_ind=sum_ind+ind
        #AP=AP+1./(1+ind)
        AP=AP+sum([sum(r[:z + 1]) / (z + 1.)  for z, y in enumerate(r) if y])
    
    mAP=AP/num_test_samples

    print('Mean Average Precision:',mAP)
    return scores,mAP



training_data_torch = torchfile.load('/home/hessam/Activity_recognition/hmdb_stip/data_feat_train_RGB_centerCrop_25f_sp1.t7')


testing_data_torch=torchfile.load('/home/hessam/Activity_recognition/hmdb_stip/data_feat_test_RGB_centerCrop_25f_sp1.t7')
vgg_matrix_train=training_data_torch['featMats']
vgg_matrix_train=np.swapaxes(vgg_matrix_train,1,2)

labels_arr_train=training_data_torch['labels']
 

#labels_arr_train=np.hstack((labels_arr_train2,labels_arr_train))
#vgg_matrix_train=np.vstack((vgg_matrix_train2,vgg_matrix_train))

vgg_matrix_test=testing_data_torch['featMats']
vgg_matrix_test=np.swapaxes(vgg_matrix_test,1,2)
 

labels_arr_test=testing_data_torch['labels']

 

y_train=np_utils.to_categorical(labels_arr_train,int(max(labels_arr_train)+1 ))
                
y_test=np_utils.to_categorical(labels_arr_test,int(max(labels_arr_test)+1))

num_cats=int(max(labels_arr_train))+1
num_train_samples=vgg_matrix_train.shape[0]
num_test_samples=vgg_matrix_test.shape[0]


print("Evaluating VGG by LSTM")



 
 



LSTM_Func(hidden_units,vgg_matrix_train,num_cats,learning_rate,y_train,nb_epochs,vgg_matrix_test, y_test)
     


print("Evaluating GLOVE  by LSTM")




glove_kmeans_train=vgg_matrix_train.reshape(vgg_matrix_train.shape[0]*vgg_matrix_train.shape[1],-1)
kmeans_codebook_size=int(math.sqrt(math.floor(glove_kmeans_train.shape[0])))
kmeans_codebook=learn_kmeans_codebook(glove_kmeans_train, kmeans_codebook_size)



bovw_matrix_train=np.zeros((vgg_matrix_train.shape[0],vgg_matrix_train.shape[1]))
bovw_matrix_test=np.zeros((vgg_matrix_test.shape[0],vgg_matrix_test.shape[1]))


for i in xrange(bovw_matrix_train.shape[0]):
    for j in xrange(bovw_matrix_train.shape[1]):
        bovw_matrix_train[i,j]=kmeans_codebook.predict(vgg_matrix_train[i,j,:].reshape(1,-1))[0]
 
for i in xrange(bovw_matrix_test.shape[0]):
    for j in xrange(bovw_matrix_test.shape[1]):
        bovw_matrix_test[i,j]=kmeans_codebook.predict(vgg_matrix_test[i,j,:].reshape(1,-1))[0]



glove_matrix_train=np.zeros((vgg_matrix_train.shape[0],vgg_matrix_train.shape[1],embedding_size))
glove_matrix_test=np.zeros((vgg_matrix_test.shape[0],vgg_matrix_test.shape[1],embedding_size))


glove_cocurance_train=[]

for i in xrange(int(max(labels_arr_train))):
    inds=np.where(labels_arr_train==i+1)[0]
    list_labels=[]
    for ind in inds:
        for ind_row in bovw_matrix_train[ind]:
            list_labels.append(int(ind_row))
    glove_cocurance_train.append(list_labels)
        
         
         
model = Sequential()
model.add(Embedding(2048, hidden_units, input_shape=bovw_matrix_train.shape[1:]))
model.add(Dropout(0.1))

#model.add(LSTM(hidden_units, return_sequences=True))
model.add(LSTM(hidden_units))
model.add(Dropout(0.3))
model.add(Dense(num_cats))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())



model.fit(bovw_matrix_train, y_train, nb_epoch=nb_epochs,verbose=0)

scores = model.evaluate(bovw_matrix_test, y_test, verbose=0)
print('Accuracy:',scores[1])

  
ranked_classes_all=np.argsort(-model.predict(bovw_matrix_test),axis=1)

AP=0
sum_ind=0
 
for i in xrange(num_test_samples):
        #predicted_classes=np.argmax(model.predict(vgg_matrix_test),axis=1)
    expected_class=int(labels_arr_test[i])
    predicted_classes=ranked_classes_all[i,:]
    x=(predicted_classes==expected_class)
    r=x*1
    ind=np.where(predicted_classes==expected_class)[0][0]
    sum_ind=sum_ind+ind
    #AP=AP+1./(1+ind)
    AP=AP+sum([sum(r[:z + 1]) / (z + 1.)  for z, y in enumerate(r) if y])
    
mAP=AP/num_test_samples
print('Mean Average Precision:',mAP)

        
