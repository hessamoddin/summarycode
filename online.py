from __future__ import division, print_function, absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
from scipy.stats import multivariate_normal
from sklearn.cluster import MiniBatchKMeans, KMeans
from random import sample
from glove import Glove
 

import array
import logging
import sklearn.mixture.gmm as gm
import numpy as np
import pickle
import scipy.sparse as sp
import tables as tb
 

"""       
Parameters
"""
Num_samples_per_video=5
bovw_size=15
num_LSTMs=8
train_frac=0.5
LSTM_overlap=0.25
longest_allowed_frames=500
batch_size = 1
nb_epochs = 200
hidden_units = 6
learning_rate = 1e-6
clip_norm = 1.0
embedding_size=2000
N=4


filecounter_str="file_counter4.p"
framefeatures='framefeatures4.h5'
gridfeatures='gridfeatures4.h5'
gridded_bovwfeatures='gridded_bovwfeatures4.h5'
glovefeatures='glovefeatures4.h5'
bovwfeatures='bovwfeatures4.h5'
dir_str="dirs4.p"

  
 
 
class videofile(object):
    """Class of Video file object"""
    def __init__(self, contained_bovws=None,category=None,filename=None):
        self.contained_bovws = contained_bovws
        self.category=category
        self.filename = filename
 


"""       
Define functions
"""

def embedding_func(gridded_words_overall,embedding_size):
    
    """***************
     GLOVE for Video
     ***************"""
     
    
    glove_bins=np.asarray(gridded_words_overall)
    print(glove_bins)
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
    print(dictionary)     
           
 
    
                  
    xarr=x.toarray()                         
    xarr/=np.amax(xarr)
    print("coocurance matrix")
    print(xarr)
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
                
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=codebook_size, batch_size=1000,
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

 

def estimate_gm(X,components=1000,seed=None):
    """Estimate a Gaussian mixture model.
    source: https://github.com/rkwitt/pyfsa/blob/master/core/fsa.py
    """
    logger = logging.getLogger()
    n, d = X.shape
    logger.info("Estimating %d-comp. GMM from (%d x %d) ..." %
                (components, n, d))
    gm_obj = gm.GMM (n_components=components,
                     covariance_type='diag',
                     random_state=seed)
    gm_obj.fit(X)   
    np.random.seed(1)
    return  np.float32(gm_obj.means_), np.float32(gm_obj.covars_), np.float32(gm_obj.weights_)




"""
Fisher Vector
https://github.com/jacobgil/pyfishervector/blob/master/fisher.py
"""
     
def likelihood_moment(x, ytk, moment):    
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk
    
def likelihood_statistics(samples, means, covs, weights):
    gaussians, s0, s1,s2 = {}, {}, {}, {}
#    samples = zip(range(0, len(samples)), samples)
    
    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights))]
    for index, x in samples:
        for k in range(0, len(weights)):
            gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

    return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples, means, covs, w):
    s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    return fv

    
 



videofile=[ videofile() for i in range(1000000)]
 
 

  
# The number of all (subsampled) frames in dataset                      

 
   
framefileh = tb.open_file(framefeatures, mode='r')
frametable=framefileh.root.table
  
number_frames_all=frametable.nrows


 
all_frames_ind_1=range(number_frames_all)
train_ind_1 = sample(all_frames_ind_1,int(train_frac*number_frames_all))
test_ind_1=np.delete(all_frames_ind_1,train_ind_1)
    

    

# Construct training and testing features for codeboook generation
 


#class overall_training_hdf(tb.IsDescription):
#    overall_holisitc_training = tb.Float32Col(7000, pos=1) 
#    glove_training_list = tb.Float32Col(shape=(N,N,7000), pos=2) 
    
 

print("construct training and testing features")




 

 

num_train_samples=len(train_ind_1)
num_row=frametable[0]['griddedfeature'].shape[0]
num_col=frametable[0]['griddedfeature'].shape[1]


 


# Ok, this is the summary of what happened so far:
# gridded Daisy training data: overall_gridded_training
# holistic Daisy training data: overall_holisitc_training



print("overall_holisitc_training")
 
# first method of bovw calculation: kmeans
kmeans_codebook_size_holistic=bovw_size*bovw_size
kmeans_codebook_size_gridded=kmeans_codebook_size_holistic

 

print("learn_kmeans_codebook")
# Final codebook created by Kmeans


kmeans_ind=sample(range(frametable.shape[0]),1000)
kmeans_codebook_holistic=learn_kmeans_codebook(frametable[kmeans_ind]['rawfeature'], kmeans_codebook_size_holistic)



new_dim=np.prod(frametable[kmeans_ind]['griddedfeature'].shape[0:3]),frametable[kmeans_ind]['griddedfeature'].shape[3]
gridded_nowungridded=frametable[kmeans_ind]['griddedfeature'].reshape(new_dim)
 

kmeans_gridded_ind=sample(range(gridded_nowungridded.shape[0]),1000)
kmeans_codebook_gridded=learn_kmeans_codebook(gridded_nowungridded[kmeans_gridded_ind][:], kmeans_codebook_size_holistic)

class gridfeature_hdf(tb.IsDescription):
    gridded_code = tb.Float64Col(kmeans_codebook_size_gridded , pos=1) 
    words = tb.Int64Col(num_row*num_col, pos=2) 
gridfileh = tb.open_file(gridfeatures, mode='w')
gridtable = gridfileh.create_table(gridfileh.root, 'table', gridfeature_hdf,"A table") 
 
# second method of bovw calculation: GMM (fisher vector) ... to be finished later
 
# The number of all bovws in dataset                   

filecounter_var = pickle.load( open(filecounter_str, "rb" ) )   
dirs = pickle.load( open( dir_str, "rb" ) )   


num_bovw_all=frametable[number_frames_all-1]['bovw_id']+1
# Number of all files

unique_video_files=list(set(filecounter_var))
num_videos=len(unique_video_files)

"""
Codebook generation for representation of Bag of Visual Words
Summary:
bovwcodebook[i].gridded_code:
(For video bag i, the codeword generated based on the
whole gridded Daisy features inside each frame of the bag)
framefeature[j].gridded_code:
(For frame j, the gridded daisy feature)
bovwcodebook[i].code:
(For video bag i, the holistic codebook across all containing frames)
"""


print("for i in xrange(num_bovw_all)")


 
class bovwfeature_hdf(tb.IsDescription):
    contained_frames=tb.Int32Col(bovw_size,pos=1)
    middle_frame=tb.Int32Col(pos=2)
    category        = tb.StringCol(10,pos=3)        
    filename=tb.StringCol(200, pos=4) 
    code=tb.Float32Col(kmeans_codebook_size_holistic,pos=5)
    
 
 
  
    
bovwfileh = tb.open_file(bovwfeatures, mode='w')
bovwtable = bovwfileh.create_table(bovwfileh.root, 'table', bovwfeature_hdf,"A table") 


class gridded_bovwfeature_hdf(tb.IsDescription):
    gridded_code=tb.Int32Col(kmeans_codebook_size_gridded,pos=1)
    words=tb.Int32Col(shape=(bovw_size,N*N),pos=2)


gridded_bovwfileh = tb.open_file(gridded_bovwfeatures.h5, mode='w')
gridded_bovwtable = gridded_bovwfileh.create_table(gridded_bovwfileh.root, 'table', gridded_bovwfeature_hdf,"A table") 
 
    

num_frames_overall=0
num_bags_overall=0
gridded_words_overall=[]  # All
for i in xrange(num_bovw_all):
    # which frames does the current Bovw contain
    
    current_contained_frames= [ind for ind in range(number_frames_all) if frametable[ind]['bovw_id'] == i]
    
    
    if len(current_contained_frames)==bovw_size:
         num_bags_overall=num_bags_overall+1
         num_frames_overall=num_frames_overall+len(current_contained_frames)
        
         
            # take the middle frame of the bag of visual words as its examplar
         middle_frame=current_contained_frames[len(current_contained_frames)//2]
 
         # categotry of the current bag = category of its middle frame= 
         #category of all frames in the bag= category of the video containing the bag
         
         current_category=frametable[middle_frame]['category']
         current_filename=frametable[middle_frame]['filename']


         training_list_holistic=[]
         for j in current_contained_frames:
             training_list_holistic.append(frametable[j]['rawfeature'])
     
         current_code=calc_bovw(np.asarray(training_list_holistic), kmeans_codebook_holistic)
         bovwtable.append([(current_contained_frames,middle_frame,current_category,current_filename,current_code)])
         training_gridded_intrabag=[]
         gridded_words_intrabag=[]
         for j in current_contained_frames:
             
             current_gridded_frame_feature=frametable[j]['griddedfeature']
             gridded_words_intraframe=np.zeros((num_row,num_col))
             training_gridded_intraframe=[]
             for row_id in xrange(num_row):
                 for col_id in xrange(num_col):
                     current_grid_feature=current_gridded_frame_feature[row_id,col_id,:].reshape(1,-1)
                     current_grid_word = kmeans_codebook_gridded.predict(current_grid_feature)
                     # Map the gridded daisy feature to a word
                     gridded_words_intraframe[row_id,col_id]=current_grid_word[0]
                     #temp=calc_bovw(np.transpose(current_grid_feature), kmeans_codebook_gridded)
                     training_gridded_intraframe.append(current_grid_feature)
                     training_gridded_intrabag.append(current_grid_feature)
                     
            # gridtable[j]['gridded_code']=calc_bovw(np.squeeze(np.asarray(training_gridded_intraframe),axis=(1,)), kmeans_codebook_gridded)  #saves gridded Bovw for the whole frame
             #gridtable[j]['words']=np.reshape(gridded_words_intraframe, (np.product(gridded_words_intraframe.shape),))   
             current_gridded_code=calc_bovw(np.squeeze(np.asarray(training_gridded_intraframe),axis=(1,)), kmeans_codebook_gridded)  #saves gridded Bovw for the whole frame
             current_words=np.reshape(gridded_words_intraframe, (np.product(gridded_words_intraframe.shape),)).astype(int)

             gridtable.append([(current_gridded_code,current_words)])
             gridtable.flush()
             gridded_words_intrabag.append(np.reshape(gridded_words_intraframe, (np.product(gridded_words_intraframe.shape),))) # each row contains words for each containing frame
             gridded_words_overall.append(np.reshape(gridded_words_intraframe, (np.product(gridded_words_intraframe.shape),)))
        
    current_bovwg_ridded_code=calc_bovw(np.squeeze(np.asarray(training_gridded_intrabag),axis=(1,)), kmeans_codebook_gridded)  #saves gridded Bovw for the whole bag     
    current_bovw_words=np.asarray(gridded_words_intrabag) # all words across all frames wihin the bag i           
    gridded_bovwtable.append([(current_bovwg_ridded_code,current_bovw_words)])
    
print(len(gridded_words_overall))
new_word_representation,dictionary=embedding_func(gridded_words_overall,embedding_size)


class glovefeature_hdf(tb.IsDescription):
    glove_words= tb.Float32Col(embedding_size, pos=1) 
glovefileh = tb.open_file(glovefeatures, mode='w')
glovetable = glovefileh.create_table(glovefileh.root, 'table', glovefeature_hdf,"A table") 

gridded_bovwfileh.close()
gridded_bovwfileh = tb.open_file(gridded_bovwfeatures, mode='r')
gridded_bovwtable = gridded_bovwfileh.root.table
 

for i in xrange(num_bags_overall):
    current_bag_words=np.ravel(gridded_bovwtable[i]['words'])
    
    bag_new_rep=[]
    for j in xrange(current_bag_words.size):
        bag_new_rep.append(np.ravel(new_word_representation[dictionary[current_bag_words[j]]]))
    glovetable.append(np.mean(np.transpose(bag_new_rep),axis=1))    
     
     
    


glovefileh.close()
glovefileh = tb.open_file(glovefeatures, mode='r')
glovetable = glovefileh.root.table

# dic_keys=old gridded Words
# dic_values= position of the word in dictionary
query=11
target=23
sim=np.dot(new_word_representation[dictionary[query]],new_word_representation[dictionary[target]])
print(sim)


cat_list=[]
sample_ind=0
overall_bovw_ind=[]
X_bovw_code=[]
X_raw_code=[]
X_glove_code=[]
X_sample_timestep=[]
"""
Video file level containing BOVW
"""

print("Video file level containing BOVW")
for i in xrange(num_videos):
     videofile[i].filename=unique_video_files[i]
     current_contained_bovws= [ind for ind in range(bovwtable.nrows) if bovwtable[ind]['filename'] == unique_video_files[i]]
     videofile[i].contained_bovws=current_contained_bovws
     videofile[i].category= bovwtable[current_contained_bovws[len(current_contained_bovws)//2]]['category']
     # Format the training and testing for TFlearn LSTM model
     chunks_bovws_ind=list(chunks(current_contained_bovws,num_LSTMs))
     if len(chunks_bovws_ind[len(chunks_bovws_ind)-1])<num_LSTMs:
         chunks_bovws_ind=chunks_bovws_ind[0:len(chunks_bovws_ind)-1]
     timestep_ind=0
     for current_bovw_chunk_ind in chunks_bovws_ind:
         cat_list.append(dirs.index(videofile[i].category))
         for timestep in xrange(num_LSTMs):
             overall_bovw_ind.append(current_bovw_chunk_ind[timestep])
             current_bovw_code=bovwtable[current_bovw_chunk_ind[timestep]]['code']
             current_glove_words=glovetable[current_bovw_chunk_ind[timestep]]['glove_words']
             X_bovw_code.append(current_bovw_code)
             X_raw_code.append(frametable[bovwtable[current_bovw_chunk_ind[timestep]]['middle_frame']]['rawfeature'])
             X_glove_code.append(current_glove_words)
             X_sample_timestep.append((sample_ind,timestep))
         sample_ind=sample_ind+1
          
         
 
# Training samples to LSTM (num samples X num timesteps aka LSTMS X feature dim)
X=np.zeros((sample_ind,num_LSTMs,len(X_bovw_code[0])))

for i in xrange(len(overall_bovw_ind)):
    ind1=X_sample_timestep[i][0]
    ind2=X_sample_timestep[i][1]
    X[ind1,ind2,:]=X_bovw_code[i]
    
    
# Training samples to LSTM (num samples X num timesteps aka LSTMS X feature dim)
X_raw=np.zeros((sample_ind,num_LSTMs,len(X_raw_code[0])))

for i in xrange(len(overall_bovw_ind)):
    ind1=X_sample_timestep[i][0]
    ind2=X_sample_timestep[i][1]
    X_raw[ind1,ind2,:]=X_raw_code[i]    

X_glove=np.zeros((sample_ind,num_LSTMs,len(X_glove_code[0])))

for i in xrange(len(overall_bovw_ind)):
    ind1=X_sample_timestep[i][0]
    ind2=X_sample_timestep[i][1]
    X_glove[ind1,ind2,:]=X_glove_code[i]
    
  
# Split training and testing sets for frames
all_frames_ind_2=range(len(cat_list))
train_ind_2= sample(all_frames_ind_2,int(0.5*len(cat_list)))
test_ind_2=np.delete(all_frames_ind_2,train_ind_2)


nb_classes=len(dirs)
Y = np_utils.to_categorical(np.asarray(cat_list),nb_classes )


X_test=X[test_ind_2,:]   
X_train=X[train_ind_2,:]    
Y_test=Y[test_ind_2,:]   
Y_train=Y[train_ind_2,:]  



X_raw_test=X_raw[test_ind_2,:]   
X_raw_train=X_raw[train_ind_2,:]   
 
X_glove_test=X_glove[test_ind_2,:]   
X_glove_train=X_glove[train_ind_2,:]

 
print('Evaluate IRNN with BOVW...')
model = Sequential()

model.add(LSTM(output_dim=hidden_units,activation='relu',input_shape=X_train.shape[1:]))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=nb_epochs,verbose=0)

scores = model.evaluate(X_test, Y_test, verbose=0)
#print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])










print('Evaluate IRNN with raw frames ...')
model = Sequential()

model.add(LSTM(output_dim=hidden_units,activation='relu',input_shape=X_raw_train.shape[1:]))
 
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(X_raw_train, Y_train, nb_epoch=nb_epochs,
          verbose=0)

scores = model.evaluate(X_raw_test, Y_test, verbose=0)
#print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])


 
  
print('Evaluate IRNN with Glove...')
model = Sequential()

model.add(LSTM(output_dim=hidden_units,activation='relu',input_shape=X_glove_train.shape[1:]))
 
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(X_glove_train, Y_train, nb_epoch=nb_epochs,verbose=0)

scores = model.evaluate(X_glove_test, Y_test, verbose=0)
#print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])
