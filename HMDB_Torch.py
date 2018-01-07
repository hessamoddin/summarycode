from __future__ import print_function
from __future__ import division, print_function, absolute_import
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB

from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.layers.wrappers import  Bidirectional
import nltk
import sklearn
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.cluster import MiniBatchKMeans, KMeans
from random import sample
from glove import Glove
from sklearn import preprocessing
from glove import Corpus
from keras.layers import Dropout
from gensim import utils, corpora, matutils, models
import glove
import gensim
import scipy

from sklearn.svm import SVC
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
from gensim.models.doc2vec import Doc2Vec
from sklearn.svm import LinearSVC,SVC
 
from numpy import dot
from nltk.cluster.kmeans import KMeansClusterer

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
from sklearn.metrics.pairwise import euclidean_distances
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
 
def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    return cosine_similarity(X,Y)


def hist_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

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
 
 
def Text2Glove(sentences,embedding_size):

    id2word = corpora.Dictionary(sentences)
    #id2word.filter_extremes(keep_n=285)
    word2id = dict((word, id) for id, word in id2word.iteritems())
    # Filter all wiki documents to contain only those 30k words.
    filter_text = lambda text: [word for word in text if word in word2id]
    filtered_wiki = lambda: (filter_text(text) for text in sentences)  # generator
    
    # Get the word co-occurrence matrix -- needs lots of RAM!!
    cooccur = glove.Corpus()
    cooccur.fit(filtered_wiki())
    
    # and train GloVe model itself, using 10 epochs
    model_glove = glove.Glove(no_components=embedding_size, learning_rate=0.05)
    model_glove.fit(cooccur.matrix, epochs=10)
    model_glove.add_dictionary(cooccur.dictionary)
    return model_glove,word2id



 


def Bovw_Cocurance_Func(bovw_matrix,labels_arr):
    bovw_cocurance=[]

    for i in xrange(int(max(labels_arr))):
        inds=np.where(labels_arr==i+1)[0]
        list_labels=[]
        for ind in inds:
            ind_row= bovw_matrix[ind]
            for w in ind_row:
                list_labels.append(int(w))
        bovw_cocurance.append(list_labels)
    return bovw_cocurance



def Array2MatStr(bovw_matrix,word2vec_model,embedding_size):
    word2vec_matrix=np.zeros((bovw_matrix.shape[0],bovw_matrix.shape[1],embedding_size))
    for i in xrange(word2vec_matrix.shape[0]):
        for j in xrange(word2vec_matrix.shape[1]):
            w_int=int(bovw_matrix[i,j])
            try:
                w_str='{:.2f}'.format(w_int)
                word2vec_matrix[i,j,:]=word2vec_model.wv[w_str]
            except:
                pass
    return word2vec_matrix
    


    
    
    
    
    

def Array2Text(bovw_matrix_train):
    wiki=[]
    for i in xrange(len(bovw_matrix_train)):
        current_cat_glove=bovw_matrix_train[i]
        current_cat_string =   ['{:.2f}'.format(g) for g in current_cat_glove]
        wiki.append(current_cat_string)
    return wiki


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



def Sim_Train_Test(cocurance_train,cocurance_test,kmeans_codebook_size):

    sim_mat=np.zeros((len(cocurance_test),len(cocurance_train)))

    for Query_ind in xrange(len(cocurance_test)):
        for Target_ind in xrange(len(cocurance_train)):
            Query_hist=np.histogram(cocurance_test[Query_ind],bins=range(0,1+kmeans_codebook_size),density=True)[0]
            Target_hist=np.histogram(cocurance_train[Target_ind],bins=range(0,1+kmeans_codebook_size),density=True)[0]
            sim_mat[Query_ind,Target_ind]=hist_intersection(Query_hist,Target_hist)
    
         
    predicted_labels=np.argmax(sim_mat,axis=1)
    count=0

    for i in xrange(len(predicted_labels)):
        
        if predicted_labels[i]==i:
            count=count+1

    return float(count)/51



def Sim_Queried(matrix_test,cocurance_train,kmeans_codebook_size):
    sim_mat=np.zeros((matrix_test.shape[0],len(cocurance_train)))
    for Query_ind in xrange(matrix_test.shape[0]):
        for Target_ind in xrange(len(cocurance_train)):
            q=np.histogram(matrix_test[Query_ind] ,bins=range(0,1+kmeans_codebook_size),density=True)[0]
            t=np.histogram(cocurance_train[Target_ind],bins=range(0,1+kmeans_codebook_size),density=True)[0]
            sim_mat[Query_ind,Target_ind]=hist_intersection(q,t)
        
  
    
    predicted_labels=np.argmax(sim_mat,axis=1)
    actual_labels=labels_arr_test-1
    count=0

    for i in xrange(len(predicted_labels)):
        if predicted_labels[i]==actual_labels[i]:
            count=count+1
        
        
    accuracy=float(count)/len(predicted_labels)    
    return accuracy


 
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
training_data_torch2 = torchfile.load('/home/hessam/Activity_recognition/hmdb_stip/data_feat_train_RGB_centerCrop_25f_sp2.t7')



testing_data_torch=torchfile.load('/home/hessam/Activity_recognition/hmdb_stip/data_feat_test_RGB_centerCrop_25f_sp1.t7')
testing_data_torch2=torchfile.load('/home/hessam/Activity_recognition/hmdb_stip/data_feat_test_RGB_centerCrop_25f_sp2.t7')

vgg_matrix_train=training_data_torch['featMats']
vgg_matrix_train=np.swapaxes(vgg_matrix_train,1,2)



vgg_matrix_train2=training_data_torch2['featMats']
vgg_matrix_train2=np.swapaxes(vgg_matrix_train2,1,2)
vgg_matrix_train2=np.vstack((vgg_matrix_train,vgg_matrix_train2))



vgg_matrix_test=testing_data_torch['featMats']
vgg_matrix_test=np.swapaxes(vgg_matrix_test,1,2)
num_test_samples=vgg_matrix_test.shape[0]





labels_arr_train=training_data_torch['labels']
labels_arr_train2=training_data_torch2['labels']
labels_arr_train2=np.hstack((labels_arr_train,labels_arr_train2))
labels_arr_test=testing_data_torch['labels']

 
y_train=np_utils.to_categorical(labels_arr_train,int(max(labels_arr_train)+1 ))
y_train2=np_utils.to_categorical(labels_arr_train2,int(max(labels_arr_train2)+1 ))              
y_test=np_utils.to_categorical(labels_arr_test,int(max(labels_arr_test)+1))
  



vgg_kmeans_train_samples=vgg_matrix_train.reshape(vgg_matrix_train.shape[0]*vgg_matrix_train.shape[1],-1)
vgg_kmeans_codebook_size=int(math.sqrt(math.floor(vgg_kmeans_train_samples.shape[0])))
vgg_kmeans_codebook=learn_kmeans_codebook(vgg_kmeans_train_samples, vgg_kmeans_codebook_size)

bovw_matrix_train=np.zeros((vgg_matrix_train.shape[0],vgg_matrix_train.shape[1]))
bovw_matrix_test=np.zeros((vgg_matrix_test.shape[0],vgg_matrix_test.shape[1]))  


for i in xrange(bovw_matrix_train.shape[0]):
    for j in xrange(bovw_matrix_train.shape[1]):
        bovw_matrix_train[i,j]=vgg_kmeans_codebook.predict(vgg_matrix_train[i,j,:].reshape(1,-1))[0]
        
         
for i in xrange(bovw_matrix_test.shape[0]):
    for j in xrange(bovw_matrix_test.shape[1]):
        bovw_matrix_test[i,j]=vgg_kmeans_codebook.predict(vgg_matrix_test[i,j,:].reshape(1,-1))[0]

 
         
 
 
bovw_cocurance_train=Bovw_Cocurance_Func(bovw_matrix_train,labels_arr_train)
bovw_cocurance_test=Bovw_Cocurance_Func(bovw_matrix_test,labels_arr_test)


print("Accuracy for Categorized BovW:")
accuracy_cat=Sim_Train_Test(bovw_cocurance_train,bovw_cocurance_test,vgg_kmeans_codebook_size)
print(accuracy_cat)

print("Accuracy for Individual BovW")    
accuracy_query=Sim_Queried(bovw_matrix_test,bovw_cocurance_train,vgg_kmeans_codebook_size)
print(accuracy_query)





print("Representation: Glove")


sentences_train=Array2Text(bovw_cocurance_train)
sentences_test=Array2Text(bovw_cocurance_test)




model_glove,word2id=Text2Glove(sentences_train,embedding_size)


glove_matrix_train=np.zeros((bovw_matrix_train.shape[0],bovw_matrix_train.shape[1],embedding_size))
glove_matrix_test=np.zeros((bovw_matrix_test.shape[0],bovw_matrix_test.shape[1],embedding_size))


for i in  xrange(glove_matrix_train.shape[0]):
    for j in xrange(glove_matrix_train.shape[1]):
        word_bovw=bovw_matrix_train[i,j]
        glove_matrix_train[i,j,:]=model_glove.word_vectors[word2id['{:.2f}'.format(word_bovw)]]


for i in  xrange(glove_matrix_test.shape[0]):
    for j in xrange(glove_matrix_test.shape[1]):
        word_bovw=bovw_matrix_test[i,j]
        glove_matrix_test[i,j,:]=model_glove.word_vectors[word2id['{:.2f}'.format(word_bovw)]]




print("Representation: Word2Vec")

word2vec_model=gensim.models.Word2Vec(sentences_train, size=embedding_size, window=5, min_count=1, workers=4)

#print('IRNN test score:', scores[0])

word2vec_matrix_train=np.zeros((bovw_matrix_train.shape[0],bovw_matrix_train.shape[1],embedding_size))
word2vec_matrix_train=Array2MatStr(bovw_matrix_train,word2vec_model,embedding_size)

word2vec_matrix_test=np.zeros((bovw_matrix_test.shape[0],bovw_matrix_test.shape[1],embedding_size))
word2vec_matrix_test=Array2MatStr(bovw_matrix_test,word2vec_model,embedding_size)


word2vec_2D_train=np.mean(word2vec_matrix_train,axis=1)
word2vec_2D_test=np.mean(word2vec_matrix_test,axis=1)


print("Activity classification")

#https://radimrehurek.com/gensim/models/keyedvectors.html
np.abs(cosine_similarity(glove_matrix_train[111,1,:].reshape(1,-1),glove_matrix_train[3,1,:].reshape(1,-1))[0][0])

np.abs(cosine_similarity(word2vec_matrix_train[111,1,:].reshape(1,-1),word2vec_matrix_train[3,1,:].reshape(1,-1))[0][0])


Print("Method= Linear SVC")
svc_model=LinearSVC(random_state=0)
svc_model.fit(word2vec_2D_train,labels_arr_train)
results=svc_model.predict(word2vec_2D_test)
print("SVC accuracy = "+repr(sklearn.metrics.accuracy_score(labels_arr_test,results)))



Print("Method= My LSTM")

    model = Sequential()

 
 

    model.add(LSTM(output_dim=hidden_units,activation='relu'))
    model.add(Dense(52))
    model.add(Dropout(0.2))

    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

    model.fit(word2vec_2D_train, y_train, nb_epoch=nb_epochs,verbose=0)

    scores = model.evaluate(word2vec_2D_test, y_test, verbose=0)    
    print('Accuracy:', scores[1])

    ranked_classes_all=np.argsort(-model.predict(word2vec_2D_test),axis=1)

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



 
Print("MEthod= LSTM") 
np.random.seed(0)
model = Sequential()
model.add(Dense(embedding_size, input_dim=word2vec_matrix_train.shape[2], init='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(embedding_size, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(52, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit( word2vec_matrix_train , y_train , nb_epoch=30, batch_size=2)
results = model.predict_classes( word2vec_2D_test )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( labels_arr_test , results )  ))
print (sklearn.metrics.classification_report( labels_arr_test , results ))







print ("Method = Stack of two LSTMs")
max_features=5000
embeddings_dim=embedding_size
np.random.seed(0)
model = Sequential()
model.add(Embedding(max_features, embeddings_dim, mask_zero=True))
model.add(Dropout(0.25))
model.add(LSTM(output_dim=embeddings_dim , activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(output_dim=embeddings_dim , activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(52))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam')  
model.fit( word2vec_2D_train , y_train , nb_epoch=30, batch_size=4)

results = model.predict_classes( y_test )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))








print ("Method = Linear SVM with doc2vec features")
np.random.seed(0)
class LabeledLineSentence(object):
  def __init__(self, data ): self.data = data
  def __iter__(self):
    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["S_%s" % uid] )
model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
sentences = LabeledLineSentence( train_texts + test_texts )
model.build_vocab( sentences )
model.train( sentences )
for w in model.vocab.keys():
  try: model[w] = embeddings[w] 
  except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
model = LinearSVC( random_state=0 )
model.fit( train_rep , train_labels )
results = model.predict( test_rep )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))












import numpy as np
import csv
import keras
import sklearn
import gensim
import random
import scipy
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential , Graph
from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC , SVC
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from word_movers_knn import WordMoversKNN

# size of the word embeddings
embeddings_dim = 300

# maximum number of words to consider in the representations
max_features = 1000

# maximum length of a sentence

# percentage of the data used for model training
percent = 0.75

# number of classes
num_classes = 2

print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
embeddings = Word2Vec.load_word2vec_format( "GoogleNews-vectors-negative300.bin.gz" , binary=True ) 

print ("Reading text data for classification and building representations...")
data = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("test-data.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
random.shuffle( data )
train_size = int(len(data) * percent)
train_texts = [ txt.lower() for ( txt, label ) in data[0:train_size] ]
test_texts = [ txt.lower() for ( txt, label ) in data[train_size:-1] ]
train_labels = [ label for ( txt , label ) in data[0:train_size] ]
test_labels = [ label for ( txt , label ) in data[train_size:-1] ]
num_classes = len( set( train_labels + test_labels ) )
tokenizer = Tokenizer(nb_words=max_features, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
tokenizer.fit_on_texts(train_texts)
train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sent_len )
test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sent_len )
train_matrix = tokenizer.texts_to_matrix( train_texts )
test_matrix = tokenizer.texts_to_matrix( test_texts )
for word,index in tokenizer.word_index.items():
  if index < max_features:
    try: embedding_weights[index,:] = embeddings[word]
    except: embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )
le = preprocessing.LabelEncoder( )
le.fit( train_labels + test_labels )
train_labels = le.transform( train_labels )
test_labels = le.transform( test_labels )
print "Classes that are considered in the problem : " + repr( le.classes_ ) 





print ("Method = CNN from the paper 'Convolutional Neural Networks for Sentence Classification'")
np.random.seed(0)
nb_filter = embeddings_dim
model = Graph()
model.add_input(name='input', input_shape=(max_sent_len,), dtype=int)
model.add_node(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ), name='embedding', input='input')
model.add_node(Dropout(0.25), name='dropout_embedding', input='embedding')
for n_gram in [3, 5, 7]:
    model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=n_gram, border_mode='valid', activation='relu', subsample_length=1, input_dim=embeddings_dim, input_length=max_sent_len), name='conv_' + str(n_gram), input='dropout_embedding')
    model.add_node(MaxPooling1D(pool_length=max_sent_len - n_gram + 1), name='maxpool_' + str(n_gram), input='conv_' + str(n_gram))
    model.add_node(Flatten(), name='flat_' + str(n_gram), input='maxpool_' + str(n_gram))
model.add_node(Dropout(0.25), name='dropout', inputs=['flat_' + str(n) for n in [3, 5, 7]])
model.add_node(Dense(1, input_dim=nb_filter * len([3, 5, 7])), name='dense', input='dropout')
model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
model.add_output(name='output', input='sigmoid')
if num_classes == 2: model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')
else: model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam') 
model.fit({'input': train_sequences, 'output': train_labels}, batch_size=32, nb_epoch=30)
results = np.array(model.predict({'input': test_sequences}, batch_size=32)['output'])
if num_classes != 2: results = results.argmax(axis=-1)
else: results = (results > 0.5).astype('int32')
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))




print ("Method = Bidirectional LSTM")
np.random.seed(0)
model = Graph()
model.add_input(name='input', input_shape=(max_sent_len,), dtype=int)
model.add_node(Embedding( max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights] ), name='embedding', input='input')
model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True), name='forward1', input='embedding')
model.add_node(Dropout(0.25), name="dropout1", input='forward1')
model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid'), name='forward2', input='forward1')
model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True), name='backward1', input='embedding')
model.add_node(Dropout(0.25), name="dropout2", input='backward1') 
model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid', go_backwards=True), name='backward2', input='backward1')
model.add_node(Dropout(0.25), name='dropout', inputs=['forward2', 'backward2'])
model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
model.add_output(name='output', input='sigmoid')
if num_classes == 2: model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')
else: model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam')
model.fit({'input': train_sequences, 'output': train_labels}, batch_size=32, nb_epoch=30)
results = np.array(model.predict({'input': test_sequences}, batch_size=32)['output'])
if num_classes != 2: results = results.argmax(axis=-1)
else: results = (results > 0.5).astype('int32')
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))



print ("Method = CNN-LSTM")
np.random.seed(0)
filter_length = 3
nb_filter = embeddings_dim
pool_length = 2
model = Sequential()
model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, weights=[embedding_weights]))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='relu', subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(embeddings_dim))
model.add(Dense(1))
model.add(Activation('sigmoid'))
if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
else: model.compile(loss='categorical_crossentropy', optimizer='adam')  
model.fit( train_sequences , train_labels , nb_epoch=30, batch_size=32)
results = model.predict_classes( test_sequences )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results ) ) )
print (sklearn.metrics.classification_report( test_labels , results ))









print ("Method = Non-linear SVM with doc2vec features")
np.random.seed(0)
class LabeledLineSentence(object):
  def __init__(self, data ): self.data = data
  def __iter__(self):
    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["S_%s" % uid] )
model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
sentences = LabeledLineSentence( train_texts + test_texts )
model.build_vocab( sentences )
model.train( sentences )
for w in model.vocab.keys():
  try: model[w] = embeddings[w] 
  except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
model = SVC( random_state=0 , kernel='poly' )
model.fit( train_rep , train_labels )
results = model.predict( test_rep )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = MLP with doc2vec features")
np.random.seed(0)
class LabeledLineSentence(object):
  def __init__(self, data ): self.data = data
  def __iter__(self):
    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["S_%s" % uid] )
model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
sentences = train_texts + test_texts
sentences = LabeledLineSentence( sentences )
model.build_vocab( sentences )
model.train( sentences )
for w in model.vocab.keys():
  try: model[w] = embeddings[w]
  except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
model = Sequential()
model.add(Dense(embeddings_dim, input_dim=train_rep.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(embeddings_dim, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
else: model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit( train_rep , train_labels , nb_epoch=30, batch_size=32)
results = model.predict_classes( test_rep )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

 
