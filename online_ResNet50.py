# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:29:32 2018

@author: hessam
"""
import os ,glob,math
import numpy as np
import tables as tb
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from gensim import corpora 
import glove
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import np_utils




database_path="/home/hessam/Activity_recognition/HMDB" 
precent_train=0.7
Num_LSTMs=8
subsample_rate=5
max_vid_len=Num_LSTMs* subsample_rate # 8 sample frames per videoss
embedding_size=200


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

def Word2Glove(bovw_matrix_train,embedding_size):
    
    sentences=[]
    for file_words in bovw_matrix_train:
        current_sentence=[]    
        for w_int in file_words:
            if w_int>0:
                w_str='{:.2f}'.format(w_int)
                current_sentence.append(w_str)
        sentences.append(current_sentence)
        
        
        
    
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



class framefeature_hdf(tb.IsDescription):
    category        = tb.StringCol(20,pos=1)   
    filepath        = tb.StringCol(250, pos=2) 
    frame_no         = tb.Int32Col((1,2048), pos=3) 
    rawfeature      = tb.Float32Col((1,2048), pos=4) 
    
print("***********")
print("ResNet->W")
print("***********")




print("Split training and testing sets of files ...")
h5_files_path=database_path+'/**/*.h5'
h5_all_files=glob.glob(h5_files_path) 
h5_train_files, h5_test_files=train_test_split(h5_all_files,train_size=precent_train)


print("Accumulating ResNet training features across training set  ...")
resnet50_train_overall=np.zeros((0,2048))

y_train_categories=[]

for current_file in h5_train_files:
    current_cat=os.path.split(os.path.dirname(current_file))[1]
    y_train_categories.append(current_cat)
    rawfeaturefileh = tb.open_file(current_file, mode='r')
    num_frames=rawfeaturefileh.root.table.nrows
    for i in xrange(num_frames):
        current_video_rawfeature=rawfeaturefileh.root.table[i]['rawfeature']
        resnet50_train_overall=np.vstack((resnet50_train_overall,current_video_rawfeature))
    rawfeaturefileh.close()
    
  
y_test_categories=[]
for current_file in h5_test_files:
    current_cat=os.path.split(os.path.dirname(current_file))[1]
    y_test_categories.append(current_cat)



print("Calculate Kmeans codebook for quantization  ...")

kmeans_codebook_size=int(math.sqrt(math.floor(resnet50_train_overall.shape[0])))*6
kmeans_codebook = KMeans( n_clusters=kmeans_codebook_size)
kmeans_codebook.fit(resnet50_train_overall)

print("***********")
print("W->Glove")
print("***********")


print("Calculate coocurance  ...")


bovw_matrix_train=np.zeros((len(h5_train_files),Num_LSTMs))
for i in xrange(len(h5_train_files)):
    rawfeaturefileh = tb.open_file(h5_train_files[i], mode='r')
    num_frames=rawfeaturefileh.root.table.nrows
    for j in xrange(num_frames):
        current_frame_rawfeature=rawfeaturefileh.root.table[j]['rawfeature']
        current_frame_w=kmeans_codebook.predict(current_frame_rawfeature)[0]
        bovw_matrix_train[i,j]=int(current_frame_w)
        
        
bovw_matrix_test=np.zeros((len(h5_test_files),Num_LSTMs))
for i in xrange(len(h5_test_files)):
    rawfeaturefileh = tb.open_file(h5_test_files[i], mode='r')
    num_frames=rawfeaturefileh.root.table.nrows
    for j in xrange(num_frames):
        current_frame_rawfeature=rawfeaturefileh.root.table[j]['rawfeature']
        current_frame_w=kmeans_codebook.predict(current_frame_rawfeature)[0]
        bovw_matrix_test[i,j]=int(current_frame_w)

 
print("Calculate Glove  ...")

model_glove,word2id=Word2Glove(bovw_matrix_train,embedding_size)


glove_matrix_train=np.zeros((bovw_matrix_train.shape[0],bovw_matrix_train.shape[1],embedding_size))
glove_matrix_test=np.zeros((bovw_matrix_test.shape[0],bovw_matrix_test.shape[1],embedding_size))




for i in  xrange(glove_matrix_train.shape[0]):
    for j in xrange(glove_matrix_train.shape[1]):
        word_int=bovw_matrix_train[i,j]
        if word_int>0:
            glove_matrix_train[i,j,:]=model_glove.word_vectors[word2id['{:.2f}'.format(word_int)]]
 


for i in  xrange(glove_matrix_test.shape[0]):
    for j in xrange(glove_matrix_test.shape[1]):
        word_int=bovw_matrix_test[i,j]
        if word_int>0:
            glove_matrix_test[i,j,:]=model_glove.word_vectors[word2id['{:.2f}'.format(word_int)]]

    
print("***********")
print("Glove->LSTM")
print("***********")






print("Construct LSTM model")


y_categories=y_train_categories+y_test_categories
data_dim=glove_matrix_train.shape[2]
timesteps = Num_LSTMs
unique_categories=np.unique(y_categories)
nb_classes = len(unique_categories)


cat_dict=dict(zip(unique_categories,range(nb_classes)))

y_train=[]
for cat in y_train_categories:
    y_train.append(cat_dict[cat])
y_train_categorized = np_utils.to_categorical(y_train)

y_test=[]
for cat in y_test_categories:
    y_test.append(cat_dict[cat])
y_test_categorized = np_utils.to_categorical(y_test)



# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
 
model.fit(glove_matrix_train, y_train_categorized,
          batch_size=16, nb_epoch=300,
          validation_data=(glove_matrix_test, y_test_categorized))




print("effect of noise removal")

print("effect of glove")





    
    
     
