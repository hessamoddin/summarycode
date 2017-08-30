#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:48:25 2017

@author: hessam
"""
from __future__ import division, print_function, absolute_import
from __future__ import print_function

from scipy.stats import multivariate_normal
from os.path import isfile, join
from os import path,listdir
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.util import view_as_blocks
from skimage.feature import daisy
from sklearn.cluster import MiniBatchKMeans, KMeans
from glove import Glove
import pickle

import tables as tb
import array
import logging
import sklearn.mixture.gmm as gm
import numpy as np
import os
import imageio
import os.path
import scipy.sparse as sp



"""       
Parameters
"""
subsampling_rate=2
bovw_size=15
num_LSTMs=8
train_frac=0.5
LSTM_overlap=0.25
longest_allowed_frames=500
batch_size = 10
nb_epochs = 200
hidden_units = 5
learning_rate = 1e-6
clip_norm = 1.0
new_shape,step,radius=(240,360),50,20 # for Daisy feaure
embedding_size=5000
framefeatures_str='framefeatures5.h5'
foledr_str='Tour20-Videos5'
dir_str= "dirs5.p"
file_counter_str="file_counter5.p"
folder_str='Tour20-Videos5'

     
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
                
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=codebook_size, batch_size=100,
                      n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit(X)

                
    return mbk

 
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

	
 

 

 

def Feature_Extractor_Fn(vid,num_frames,frame_no,new_shape=(360,480),step=80, radius=45):
    """Extract Daisy feature for a frame of video """
    if frame_no<num_frames-1: 
        frame = vid.get_data(frame_no)  
        frame_resized=resize(frame, new_shape)
        frame_gray= rgb2gray(frame_resized)
        daisy_desc = daisy(frame_gray,step=step, radius=radius)
        daisy_1D=np.ravel(daisy_desc)
         
        """Extract Daisy feature for a patch from the frame of video """
        N=4
        step_glove=int(step/N)
        radius_glove=int(radius/N)
        patch_shape_x=int(new_shape[0]/N)
        patch_shape_y=int(new_shape[1]/N)

        patchs_arr = view_as_blocks(frame_gray, (patch_shape_x,patch_shape_y))
        patch_num_row=patchs_arr.shape[0]
        patch_num_col=patchs_arr.shape[1]
        final_daisy_length=daisy(patchs_arr[0,0,:,:],step=step_glove, radius=radius_glove).size
        patch_daisy_arr=np.zeros((patch_num_row,patch_num_col,final_daisy_length))
        for i in xrange(patch_num_row):
            for k in xrange(patch_num_col):
                patch=patchs_arr[i,k,:,:]
                patch_daisy_desc = daisy(patch,step=step_glove, radius=radius_glove)
                patch_daisy_1D=np.ravel(patch_daisy_desc)
                patch_daisy_arr[i,k,:]=patch_daisy_1D
                
                
        
       #sift = cv2.xfeatures2d.SIFT_create()
       # (sift_kps, sift_descs) = sift.detectAndCompute(frame, None)
       # print("# kps: {}, descriptors: {}".format(len(sift_kps), sift_descs.shape))
       # surf = cv2.xfeatures2d.SURF_create()
      #  (surf_kps, surf_descs) = surf.detectAndCompute(frame, None)
       # print("# kps: {}, descriptors: {}".format(len(surf_kps), surf_descs.shape))
    else:
        print("Frame number is larger than the length of video")
  #  return (daisy_1D,surf_descs,sift_descs)
    return patch_daisy_arr,daisy_1D





def chunks(l, n):
    """
    To split video into different evenly sized set of frames to feed into LSTMs    
    Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

"""
Definition of objects to facilitate bovw feature construction
"""

 


  
   
subsampling_rate=2
bovw_size=15
new_shape,step,radius=(360,480),50,20 # for Daisy feaure
N=4


class framefeature_hdf(tb.IsDescription):
    filename        = tb.StringCol(200, pos=1) 
    category        = tb.StringCol(10,pos=2)        
    rawfeature      = tb.Float32Col(4000, pos=3) 
    bovw_id         = tb.IntCol(pos=4) 
    frame_id        = tb.IntCol(pos=5)  
    griddedfeature    = tb.Float32Col(shape=(N,N,4000), pos=6) 



framefeature_fileh = tb.open_file(framefeatures_str, mode='w')
framefeature_table = framefeature_fileh.create_table(framefeature_fileh.root, 'table', framefeature_hdf,"A table") 


 

""" ************************
****************************
Main body of the code
***************************
************************""" 

# current working directory for the code
cwd = os.getcwd()
# The folder inside which the video files are located in separate folders
parent_dir = os.path.split(cwd)[0] 
# Find the data folders
datasetpath=join(parent_dir,'Tour20',foledr_str)
# Dir the folders; each representing a category of action
dirs = os.listdir( datasetpath )


"""
This FOR loop extracts the video frames feature and store them in 
the array of framefeature objects associated with each frame
"""
print("Thus begins feature extraction!")
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
                 k=0
                 print("***")
                 print(current_file)
                 # full name and path for the current video file
                 videopath=path.join(cat_path,current_file)
                 try:
                     # read the current video file
                     vid = imageio.get_reader(videopath,  'ffmpeg')
                     # The number of frames for this video
                     num_frames=vid._meta['nframes']
                     #sampling_rate=num_frames//longest_allowed_frames+1
                     # step_percent is just a variable to monitor the progress
                     step_percent=num_frames//10
                     # trim out the bag of videos frames in a way that each
                     #have bags number equal to multiples of bovw_size
                       # j is the frame index for the bvw processable parts of video
                     effective_frames=min(5*subsampling_rate*bovw_size*num_LSTMs,num_frames)
                     num_effective_frames=len(range(0,effective_frames,subsampling_rate))
                     for j in xrange(0,effective_frames,subsampling_rate):
                         bovw_id=(i)//bovw_size  # every bovw_size block of frames
                         print("%d frames out of %d processed ..." % (k, num_effective_frames) )
                            # Feature extraction
                            # daisy_1D,surf_descs,sift_descs 		
                         # extract dausy features: for the whole frame or grid-wise for each frame
                         current_grid_feature,current_frame_feature=Feature_Extractor_Fn(vid,num_frames,j) 
                         framefeature_table.append([(videopath,cat,current_frame_feature,bovw_id,i,current_grid_feature)])
                        # print(i)
                         #print(bovw_id)
                         #print(j*subsampling_rate)
                         i=i+1
                         k=k+1
                         file_counter.append(videopath)
                         # Track record of which video does this frame belong toin a list
                         file_counter=list(set(file_counter))
                         # update feature objects for each video
                     #pickle.dump(framefeature, open( "raw_features_Class_array.p", "wb" ) )
                     #framefeature_loaded = pickle.load( open( "raw_features_Class_array.p", "rb" ) )
                 except:
                     print("error on video")
                     print(current_file)
                     print("***")
print("Finished raw feature extraction!")
file_counter=list(set(file_counter))
pickle.dump( file_counter, open(file_counter_str, "wb" ) )
pickle.dump( dirs, open(dir_str, "wb" ) )
framefeature_fileh.close()
