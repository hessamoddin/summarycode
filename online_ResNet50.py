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
from sklearn.cluster import AffinityPropagation
 



database_path="/home/hessam/Activity_recognition/HMDB" 
precent_train=0.5
Num_LSTMs=8
subsample_rate=5
max_vid_len=Num_LSTMs* subsample_rate # 8 sample frames per videoss
embedding_size=400


def Raw_Frame_Reader(h5_train_files):
  #  raw_matrix_train=np.zeros((len(h5_train_files),Num_LSTMs,2048))

    y_train_categories=[]
    resnet50_train_overall=np.zeros((0,2048))
    for i in xrange(len(h5_train_files)):
   # for i in xrange(20):
        current_file=h5_train_files[i]
        current_cat=os.path.split(os.path.dirname(current_file))[1]
        rawfeaturefileh = tb.open_file(current_file, mode='r')
        num_frames=rawfeaturefileh.root.table.nrows
        if num_frames>0:
            y_train_categories.append(current_cat)
        

        for j in xrange(num_frames):
            current_frame_rawfeature=rawfeaturefileh.root.table[j]['rawfeature']
           # raw_matrix_train[i,j,:]=current_frame_rawfeature
            resnet50_train_overall=np.vstack((resnet50_train_overall,current_frame_rawfeature))
        rawfeaturefileh.close()
    return resnet50_train_overall,y_train_categories#,raw_matrix_train
    
    
    
     
    
     

    
    

def Evaluate_Classification(glove_matrix_train,y_train_categorized,glove_matrix_test,y_test_categorized,hidden_layers,num_stacks,batch_size,nb_epoch):
# expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    
    data_dim=glove_matrix_train.shape[2]
    
    model.add(LSTM(hidden_layers, return_sequences=True,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    
    for i in xrange(num_stacks-2):
        model.add(LSTM(hidden_layers, return_sequences=True))  # returns a sequence of vectors of dimension 32

    model.add(LSTM(hidden_layers))  # return a single vector of dimension 32
    model.add(Dense(nb_classes, activation='softmax'))

           
          
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    model.fit(glove_matrix_train, y_train_categorized,batch_size=batch_size, nb_epoch=nb_epoch,verbose=0)
          
          # Final evaluation of the model
    scores = model.evaluate(glove_matrix_test, y_test_categorized)
    return scores[1]*100
    
    

    
     
    
    
    


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
   # id2word.filter_extremes(no_below=5, no_above=0.15)
   # id2word.filter_extremes( no_below=5)
    word2id = dict((word, id) for id, word in id2word.iteritems())
    print(len(word2id))
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
#h5_all_files.sort()
#h5_all_files=h5_all_files[1:600]

h5_train_files, h5_test_files=train_test_split(h5_all_files,train_size=precent_train,random_state=42)

#h5_train_files.sort()
#h5_test_files.sort()




print("Accumulating ResNet training and testing features ...")

resnet50_train_overall,y_train_categories=Raw_Frame_Reader(h5_train_files)   
raw_matrix_train=resnet50_train_overall.reshape(resnet50_train_overall.shape[0]/Num_LSTMs,Num_LSTMs,resnet50_train_overall.shape[1])

resnet50_test_overall,y_test_categories=Raw_Frame_Reader(h5_test_files)  
raw_matrix_test=resnet50_test_overall.reshape(resnet50_test_overall.shape[0]/Num_LSTMs,Num_LSTMs,resnet50_test_overall.shape[1])
 


print("Calculate Kmeans codebook for quantization  ...")

kmeans_codebook_size=int(math.sqrt(math.floor(resnet50_train_overall.shape[0])))
kmeans_codebook = KMeans( n_clusters=kmeans_codebook_size)
kmeans_codebook.fit(resnet50_train_overall)



print("Calculate AP codebook for quantization  ...")
af = AffinityPropagation().fit(resnet50_train_overall)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
AP_codebook_size = len(cluster_centers_indices) 

bovw_matrix_train=np.zeros((raw_matrix_train.shape[0],raw_matrix_train.shape[1]))

for i in xrange(bovw_matrix_train.shape[0]):
    for j in xrange(bovw_matrix_train.shape[1]):
        current_frame_rawfeature=raw_matrix_train[i,j]
        current_frame_w=kmeans_codebook.predict(current_frame_rawfeature.reshape(1,-1))[0]
        current_frame_w=af.predict(current_frame_rawfeature.reshape(1,-1))[0]
        bovw_matrix_train[i,j]=int(current_frame_w)


bovw_matrix_test=np.zeros((raw_matrix_test.shape[0],raw_matrix_test.shape[1]))

for i in xrange(bovw_matrix_test.shape[0]):
    for j in xrange(bovw_matrix_test.shape[1]):
        current_frame_rawfeature=raw_matrix_test[i,j]
        current_frame_w=kmeans_codebook.predict(current_frame_rawfeature.reshape(1,-1))[0]
        current_frame_w=af.predict(current_frame_rawfeature.reshape(1,-1))[0]
        bovw_matrix_test[i,j]=int(current_frame_w)

 
 
 
print("***********")
print("W->Glove")
print("***********")


print("Calculate coocurance  ...")

model_glove,word2id=Word2Glove(bovw_matrix_train,embedding_size)


glove_matrix_train=np.zeros((bovw_matrix_train.shape[0],bovw_matrix_train.shape[1],embedding_size))
glove_matrix_test=np.zeros((bovw_matrix_test.shape[0],bovw_matrix_test.shape[1],embedding_size))


 
print("Calculate Glove  ...")


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

 




hidden_layers=256
num_stacks=3
batch_size=16
nb_epoch=100

glove_acc=Evaluate_Classification(glove_matrix_train,y_train_categorized,glove_matrix_test,y_test_categorized,hidden_layers,num_stacks,batch_size,nb_epoch)
print("Glove accuracy:")
print(glove_acc)

raw_acc=Evaluate_Classification(raw_matrix_train,y_train_categorized,raw_matrix_test,y_test_categorized,hidden_layers,num_stacks,batch_size,nb_epoch)
print("Raw accuracy:")
print(raw_acc)








 


print("effect of cross vaidation")


print("effect of noise removal")




    
    
     
