############ Load Libraries ##############

from __future__ import division, print_function, absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
import sklearn.mixture.gmm as gm
import numpy as np
from scipy.stats import multivariate_normal
import logging
#import cv2
from os import listdir
import os.path
from os.path import isfile, join
from os import path
import imageio
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import daisy
from sklearn.cluster import KMeans
import math
from random import sample

  

# parameters:
bovw_size=5
num_LSTMs=10
train_frac=0.5
LSTM_overlap=0.25
longest_allowed_frames=500


batch_size = 1
nb_epochs = 200
hidden_units = 30

learning_rate = 1e-6
clip_norm = 1.0

#imageio.plugins.ffmpeg.download()
     
# Define functions


def learn_kmeans_codebook(X, codebook_size=1000, seed=None):
    """Learn a codebook.
    Run K-Means clustering to compute a codebook. K-Means
    is initialized by K-Means++, uses a max. of 500 iter-
    ations and 10 times re-initialization.
    Paramters
    ---------
    X : numpy matrix, shape (N,D)
        Input data.
    codebook_size : int (default : 200)
        Desired number of codewords.
    seed : int (default : None)
        Seed for random number generator.
    Returns
    -------
    cb : sklearn.cluster.KMeans object
        KMeans object after fitting.
    """

    logger = logging.getLogger()
    logger.info("Learning codebook with %d words ..." % codebook_size)

    # Run vector-quantization
    cb = KMeans(codebook_size,
                init="k-means++",
                n_init=10,
                max_iter=500,
                random_state=seed)
    cb.fit(X)
    return cb

# # Reference for GMM and Kmeans : https://github.com/rkwitt/pyfsa/blob/master/core/fsa.py
def calc_bovw(X, cb):
    """Compute a (normalized) BoW histogram.
    Parameters
    ----------
    X : numpy matrix, shape (N, D)
        Input data.
    cb : sklearn.cluster.KMeans
        Already estimated codebook with C codewords.
    Returns
    -------
    H : numpy array, shape (C,)
        Normalized (l2-norm) BoW histogram.
    """

    # Get nr. codewords
    n,d = cb.cluster_centers_.shape
    # Compute closest cluster centers
    assignments = cb.predict(X)
    # Compute (normalized) BoW histogram
    B = range(0,n+1)
    return np.histogram(assignments,bins=B,density=True)[0]

############ Extract frame features ##############

#source: https://github.com/rkwitt/pyfsa/blob/master/core/fsa.py

def estimate_gm(X,components=1000,seed=None):
    """Estimate a Gaussian mixture model.
    Note: Uses diagonal covariance matrices.
    Parameters
    ----------
    X : numpy matrix, shape (N,D)
        Matrix of data samples (i-th row is i-th sample vector).
    c : int (default : 3)
        Number of desired mixture components.
    seed : int (default : None)
        Seed for the random number generator.
    Returns
    -------
    gm_obj : sklearn.mixture.gmm object
        Estimated GMM.
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


     
def likelihood_moment(x, ytk, moment):	
	x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
	return x_moment * ytk
	
def likelihood_statistics(samples, means, covs, weights):
	gaussians, s0, s1,s2 = {}, {}, {}, {}
#	samples = zip(range(0, len(samples)), samples)
	
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

	


def Feature_Extractor_Fn(vid,num_frames,frame_no,new_shape=(120,180),step=50, radius=20):
    if frame_no<num_frames-1: 
        frame = vid.get_data(frame_no)  
        frame_resized=resize(frame, new_shape)
        frame_gray= rgb2gray(frame_resized)
        daisy_desc = daisy(frame_gray,step=step, radius=radius)
        daisy_1D=np.ravel(daisy_desc)
       #sift = cv2.xfeatures2d.SIFT_create()
       # (sift_kps, sift_descs) = sift.detectAndCompute(frame, None)
       # print("# kps: {}, descriptors: {}".format(len(sift_kps), sift_descs.shape))
       # surf = cv2.xfeatures2d.SURF_create()
      #  (surf_kps, surf_descs) = surf.detectAndCompute(frame, None)
       # print("# kps: {}, descriptors: {}".format(len(surf_kps), surf_descs.shape))
    else:
        print("Frame number is larger than the length of video")
  #  return (daisy_1D,surf_descs,sift_descs)
    return daisy_1D




# To split video into different evenly sized set of frames to feed into LSTMs
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

############# objects
# class of video features and other methadata
class feature(object):
    """__init__() functions as the class constructor"""
    def __init__(self, filename=None, category=None, rawfeature=None, bovw_id=None,frame_id=None):
        self.filename = filename
        self.category = category
        self.rawfeature = rawfeature
        self.bovw_id=bovw_id
        self.frame_id=frame_id
        
class bovw(object):
    """__init__() functions as the class constructor"""
    def __init__(self, middle_frame=None, category=None, bovw_id=None,contained_frames=None,filename=None,code=None):
        self.contained_frames = contained_frames
        self.category=category
        self.middle_frame=middle_frame
        self.filename=filename
        self.code=code
  
 
class video(object):
    """__init__() functions as the class constructor"""
    def __init__(self, contained_bovws=None,category=None,filename=None):
        self.contained_bovws = contained_bovws
        self.category=category
        self.filename = filename


framefeature = [ feature() for i in range(1000000)]
bovwcodebook=[ bovw() for i in range(1000000)]
videofile=[ video() for i in range(1000000)]


############ Access Action Category Folders ##############
# current working directory for the code
cwd = os.getcwd()
# The folder at which the other folders (data) is located at
parent_dir = os.path.split(cwd)[0] 
 


 


# Find the data folders
datasetpath=join(parent_dir,'Tour20/Tour20-Videos4/')
# Dir the folders; each representing a category of action
dirs = os.listdir( datasetpath )

i=0
file_counter=[]
# cat: categort of actions, also the name of the folder containing the action videos
for cat in dirs:
    print("Processing  %s Videos...." % (cat))    
    if "." not in cat:
	    cat_path=join(datasetpath,cat)
	    onlyfiles = [f for f in listdir(cat_path) if isfile(join(cat_path, f))]
	    for current_file in onlyfiles:
		# This dataset contains only mp4 video clips
	        if current_file.endswith('.mp4'):
                 print("***")
                 print(current_file)
                 videopath=path.join(cat_path,current_file)
                 # Extract raw Daisy and other features
                 try:
                     vid = imageio.get_reader(videopath,  'ffmpeg')
                     num_frames=vid._meta['nframes']
                     sampling_rate=num_frames//longest_allowed_frames+1
                     step_percent=num_frames//10
                     bovw_processable_len=bovw_size*(num_frames//bovw_size)
                     # j is the frame index for the bvw processable parts of video
                     for j in xrange(bovw_processable_len):
                         bovw_id=i//bovw_size  # every bovw_size block of frames
                        # print("** frame no %d **" % j)	
                         if j%step_percent==0:
                            print("%d %%" % (1+100*j//num_frames))	
                            # Feature extraction
                            # daisy_1D,surf_descs,sift_descs 			
                         current_feature=Feature_Extractor_Fn(vid,num_frames,j)
                         framefeature[i].filename=videopath
                         framefeature[i].category=cat
                         # Accumulating all raw features			
                         framefeature[i].rawfeature=current_feature
                         framefeature[i].bovw_id=bovw_id	
                         framefeature[i].frame_id=i
                         i=i+1
                         file_counter.append(videopath)
                         file_counter=list(set(file_counter))
                         # update feature objects for each video
                 except:
                     print("error on video")
                     print(current_file)
                     print("***")
print("Finished raw feature extraction!")


# The number of all (subsampled) frames in dataset                      
number_frames_all=i  


# Split training and testing sets for frames
all_frames_ind=range(number_frames_all)
train_ind = sample(all_frames_ind,int(train_frac*number_frames_all))
test_ind=np.delete(all_frames_ind,train_ind)
    

    



# Construct training and testing features for codeboook generation
training_list=[]
testing_list=[]
for i in train_ind:
    training_list.append(framefeature[i].rawfeature)
for i in test_ind:
    testing_list.append(framefeature[i].rawfeature)

bag_training=np.asarray(training_list)
bag_testing=np.asarray(testing_list)

# first method of bovw calculation: kmeans
kmeans_codebook_size=int(math.sqrt(math.floor(len(training_list))))
 


# Final codebook created by Kmeans

 
kmeans_codebook=learn_kmeans_codebook(bag_training, kmeans_codebook_size)

# second method of bovw calculation: GMM (fisher vector)



m,c,w=estimate_gm(bag_training,kmeans_codebook_size)





# The number of all bovws in dataset                      
num_bovw_all=bovw_id+1
# Number of all files
unique_video_files=list(set(file_counter))
num_videos=len(unique_video_files)





# Bag of frames level
for i in xrange(num_bovw_all):
    current_contained_frames= [ind for ind in range(len(framefeature)) if framefeature[ind].bovw_id == i]
    bovwcodebook[i].contained_frames=current_contained_frames
    middle_frame=current_contained_frames[len(current_contained_frames)//2]
    bovwcodebook[i].middle_frame=middle_frame
    bovwcodebook[i].category=framefeature[middle_frame].category
    bovwcodebook[i].filename=framefeature[middle_frame].filename
    training_list=[]
    for j in current_contained_frames:
        training_list.append(framefeature[j].rawfeature)
    bovwcodebook[i].code=calc_bovw(np.asarray(training_list), kmeans_codebook)

 
 
 
cat_list=[]
sample_ind=0
overall_bovw_ind=[]
X_bovw_code=[]
X_raw_code=[]
X_sample_timestep=[]
# Video file level containing BOVW
for i in xrange(num_videos):
     videofile[i].filename=unique_video_files[i]
     current_contained_bovws= [ind for ind in range(len(bovwcodebook)) if bovwcodebook[ind].filename == unique_video_files[i]]
     videofile[i].contained_bovws=current_contained_bovws
     videofile[i].category= bovwcodebook[current_contained_bovws[len(current_contained_bovws)//2]].category
     # Format the training and testing for TFlearn LSTM model
     chunks_bovws_ind=list(chunks(current_contained_bovws,num_LSTMs))
     if len(chunks_bovws_ind[len(chunks_bovws_ind)-1])<num_LSTMs:
         chunks_bovws_ind=chunks_bovws_ind[0:len(chunks_bovws_ind)-1]
     timestep_ind=0
     for current_bovw_chunk_ind in chunks_bovws_ind:
         cat_list.append(dirs.index(videofile[i].category))
         for timestep in xrange(num_LSTMs):
             overall_bovw_ind.append(current_bovw_chunk_ind[timestep])
             current_bovw_code=bovwcodebook[current_bovw_chunk_ind[timestep]].code
             X_bovw_code.append(current_bovw_code)
             X_raw_code.append(framefeature[bovwcodebook[current_bovw_chunk_ind[timestep]].middle_frame].rawfeature)  
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

  
# Split training and testing sets for frames
all_frames_ind=range(len(cat_list))
train_ind = sample(all_frames_ind,int(0.5*len(cat_list)))
test_ind=np.delete(all_frames_ind,train_ind)


nb_classes=len(dirs)
Y = np_utils.to_categorical(np.asarray(cat_list),nb_classes )
X_test=X[test_ind,:]   
X_train=X[train_ind,:]    
Y_test=Y[test_ind,:]   
Y_train=Y[train_ind,:]  

X_raw_test=X_raw[test_ind,:]   
X_raw_train=X_raw[train_ind,:]    
     
print('Evaluate IRNN...')
model = Sequential()

model.add(SimpleRNN(output_dim=hidden_units,
                    init=lambda shape, name: normal(shape, scale=0.001, name=name),
                    inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
                    activation='relu',
                    input_shape=X_train.shape[1:]))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=nb_epochs,verbose=0)

scores = model.evaluate(X_test, Y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])











print('Evaluate IRNN...')
model = Sequential()


model.add(SimpleRNN(output_dim=hidden_units,
                    init=lambda shape, name: normal(shape, scale=0.001, name=name),
                    inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
                    activation='relu',
                    input_shape=X_raw_train.shape[1:]))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(X_raw_train, Y_train, nb_epoch=nb_epochs,
          verbose=0)

scores = model.evaluate(X_raw_test, Y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])
