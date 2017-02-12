from __future__ import division, print_function, absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
from scipy.stats import multivariate_normal
from os.path import isfile, join
from os import path,listdir
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.util import view_as_blocks
from skimage.feature import daisy
from sklearn.cluster import KMeans
from random import sample
from glove import Glove
 

import array
import math
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
bovw_size=20
num_LSTMs=10
train_frac=0.5
LSTM_overlap=0.25
longest_allowed_frames=500
batch_size = 1
nb_epochs = 200
hidden_units = 50
learning_rate = 1e-6
clip_norm = 1.0
new_shape,step,radius=(120,180),50,20 # for Daisy feaure
embedding_size=100 
     
"""       
Define functions
"""

def embedding_func(gridded_words_overall,embedding_size):
    
    """***************
     GLOVE for Video
     ***************"""
     
    
    glove_bins=np.squeeze(np.asarray(gridded_words_overall),axis=(1,))
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
    cb = KMeans(codebook_size,
                init="k-means++",
                n_init=10,
                max_iter=500,
                random_state=seed)
    cb.fit(X)
    return cb

 
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
        step_glove=int(step/2)
        radius_glove=int(radius/2)
        patch_shape_x=int(new_shape[0]/2)
        patch_shape_y=int(new_shape[1]/2)

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

 


  
  
  
class framefeature(object):
    """class of video features and other methadata"""
    def __init__(self, filename=None, category=None, rawfeature=None, bovw_id=None,frame_id=None,glovefeature=None,griddedfeature=None,gridded_code=None,words=None,glove_words=None):
        self.filename = filename
        self.category = category
        self.rawfeature = rawfeature
        self.bovw_id=bovw_id
        self.frame_id=frame_id
        self.griddedfeature=griddedfeature
        self.gridded_code=gridded_code
        self.words=words
        self.glove_words=glove_words

        
class bovwcodebook(object):
    """Class of Bag of Video Feature object"""
    def __init__(self, middle_frame=None, category=None, bovw_id=None,contained_frames=None,filename=None,code=None,gridded_code=None,words=None,glove_words=None):
        self.contained_frames = contained_frames
        self.category=category
        self.middle_frame=middle_frame
        self.filename=filename
        self.code=code
        self.gridded_code=gridded_code
        self.words=words
        self.glove_words=glove_words
  
 
class videofile(object):
    """Class of Video file object"""
    def __init__(self, contained_bovws=None,category=None,filename=None):
        self.contained_bovws = contained_bovws
        self.category=category
        self.filename = filename

framefeature  = [ framefeature() for i in range(1000000)]
bovwcodebook=[ bovwcodebook() for i in range(1000000)]
videofile=[ videofile() for i in range(1000000)]



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
datasetpath=join(parent_dir,'Tour20/Tour20-Videos4/')
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
                     bovw_processable_len=bovw_size*(num_frames//(bovw_size))
                     bovw_processable_len=subsampling_rate*(bovw_processable_len//(subsampling_rate))
                      # j is the frame index for the bvw processable parts of video
                     for j in xrange(bovw_processable_len):
                         bovw_id=(i)//bovw_size  # every bovw_size block of frames
                        # print("** frame no %d **" % j)	
                         print("%d %%" % (1+100*subsampling_rate*j//num_frames))	
                        
                            # Feature extraction
                            # daisy_1D,surf_descs,sift_descs 		
                         # extract dausy features: for the whole frame or grid-wise for each frame
                         current_grid_feature,current_frame_feature=Feature_Extractor_Fn(vid,num_frames,j*subsampling_rate) 
                         framefeature[i].filename=videopath # take the name&path ofj the video containing the fraame
                         framefeature[i].category=cat # take the category of the current video 
                         framefeature[i].rawfeature=current_frame_feature #daisy feature for the whole video
                         framefeature[i].bovw_id=bovw_id	#bag number in the video for this frame
                         framefeature[i].frame_id=i # frame number in the video 
                         framefeature[i].griddedfeature=current_grid_feature # gridded Daisy feature for this frame
                        # print(i)
                         #print(bovw_id)
                         #print(j*subsampling_rate)
                         i=i+1
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


# The number of all (subsampled) frames in dataset                      
number_frames_all=i  



"""
Define the codebooks from traditional holistic (gridded) Daisy features for the whole
training frames and then making codebooks without (with) Glove
"""
# Split training and testing sets for frames for Bovw generation
all_frames_ind_1=range(number_frames_all)
train_ind_1 = sample(all_frames_ind_1,int(train_frac*number_frames_all))
test_ind_1=np.delete(all_frames_ind_1,train_ind_1)
    

    

# Construct training and testing features for codeboook generation
overall_holisitc_training=[]
testing_list=[]
glove_training_list=[]
glove_testing_list=[]
for i in train_ind_1:
    overall_holisitc_training.append(framefeature[i].rawfeature)
    glove_training_list.append(framefeature[i].griddedfeature)

for i in test_ind_1:
    testing_list.append(framefeature[i].rawfeature)
    glove_testing_list.append(framefeature[i].griddedfeature)



print("overall_gridded_training") 
# Finalizing the NLP kmeans training set for gridden Daisy feature and 
# transforming it through Glove
overall_gridded_training=[]

num_samples=len(glove_training_list)
num_row=glove_training_list[0].shape[0]
num_col=glove_training_list[0].shape[1]
 

for sample_id in xrange(num_samples):
    for row_id in xrange(num_row):
        for col_id in xrange(num_col):
            current_gridded_feature=glove_training_list[sample_id]
            # Accumulate all the gridded Daisy features from all frames
            # in training set; same training set for holistic Daisy used
            # for regular Kmeans codebook generation
            overall_gridded_training.append(current_gridded_feature[row_id,col_id,:])



# Ok, this is the summary of what happened so far:
# gridded Daisy training data: overall_gridded_training
# holistic Daisy training data: overall_holisitc_training



print("overall_holisitc_training")
# Finalizing the kmeans training set 
overall_holisitc_training=np.asarray(overall_holisitc_training)
overall_gridded_training=np.asarray(overall_gridded_training)

# first method of bovw calculation: kmeans
kmeans_codebook_size_gridded=int(math.sqrt(math.floor(len(overall_gridded_training))))
kmeans_codebook_size_holistic=int(math.sqrt(math.floor(len(overall_holisitc_training))))


print("learn_kmeans_codebook")
# Final codebook created by Kmeans
kmeans_codebook_holistic=learn_kmeans_codebook(overall_holisitc_training, kmeans_codebook_size_holistic)
kmeans_codebook_gridded=learn_kmeans_codebook(overall_gridded_training, kmeans_codebook_size_gridded)

# second method of bovw calculation: GMM (fisher vector) ... to be finished later
m,c,w=estimate_gm(overall_holisitc_training,kmeans_codebook_size_holistic)

# The number of all bovws in dataset                      
num_bovw_all=bovw_id+1
# Number of all files
unique_video_files=list(set(file_counter))
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


num_frames_overall=0
num_bags_overall=0
gridded_words_overall=[]  # All
for i in xrange(num_bovw_all):
    # which frames does the current Bovw contain
    current_contained_frames= [ind for ind in range(len(framefeature)) if framefeature[ind].bovw_id == i]
    
    
    if len(current_contained_frames)==bovw_size:
         num_bags_overall=num_bags_overall+1
         num_frames_overall=num_frames_overall+len(current_contained_frames)
        
     
         bovwcodebook[i].contained_frames=current_contained_frames

        	# take the middle frame of the bag of visual words as its examplar
         middle_frame=current_contained_frames[len(current_contained_frames)//2]
         bovwcodebook[i].middle_frame=middle_frame
         # categotry of the current bag = category of its middle frame= 
         #category of all frames in the bag= category of the video containing the bag
         bovwcodebook[i].category=framefeature[middle_frame].category
         bovwcodebook[i].filename=framefeature[middle_frame].filename
         training_list_holistic=[]
         for j in current_contained_frames:
             training_list_holistic.append(framefeature[j].rawfeature)
     
         bovwcodebook[i].code=calc_bovw(np.asarray(training_list_holistic), kmeans_codebook_holistic)

         training_gridded_intrabag=[]
         gridded_words_intrabag=[]
         for j in current_contained_frames:
             
             current_gridded_frame_feature=framefeature[j].griddedfeature
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
                     
             framefeature[j].gridded_code=calc_bovw(np.squeeze(np.asarray(training_gridded_intraframe),axis=(1,)), kmeans_codebook_gridded)  #saves gridded Bovw for the whole frame
             framefeature[j].words=np.reshape(gridded_words_intraframe, (np.product(gridded_words_intraframe.shape),))   

             gridded_words_intrabag.append(np.reshape(gridded_words_intraframe, (np.product(gridded_words_intraframe.shape),))) # each row contains words for each containing frame
             gridded_words_overall.append(np.reshape(gridded_words_intraframe, (np.product(gridded_words_intraframe.shape),)))
        
    bovwcodebook[i].gridded_code=calc_bovw(np.squeeze(np.asarray(training_gridded_intrabag),axis=(1,)), kmeans_codebook_gridded)  #saves gridded Bovw for the whole bag     
    bovwcodebook[i].words=np.asarray(gridded_words_intrabag) # all words across all frames wihin the bag i           
    
print(len(gridded_words_overall))
new_word_representation,dictionary=embedding_func(gridded_words_overall,embedding_size)

for i in xrange(num_bags_overall):
    current_bag_words=np.ravel(bovwcodebook[i].words)
    
    bag_new_rep=[]
    for j in xrange(current_bag_words.size):
        bag_new_rep.append(np.ravel(new_word_representation[dictionary[current_bag_words[j]]]))
    bovwcodebook[i].glove_words=np.mean(np.transpose(bag_new_rep),axis=1)
    
    
for i in xrange(num_frames_overall):
    current_word=framefeature[i].words
    frame_new_rep=[]
    for j in xrange(current_word.size):
        frame_new_rep.append(np.ravel(new_word_representation[dictionary[current_word[0,j]]]))
    framefeature[i].glove_words=np.mean(np.transpose(frame_new_rep),axis=1)
    

# dic_keys=old gridded Words
# dic_values= position of the word in dictionary
query=11
target=7
sim=np.dot(new_word_representation[dictionary[query]],new_word_representation[dictionary[target]])
print(sim)


cat_list=[]
sample_ind=0
overall_bovw_ind=[]
X_bovw_code=[]
X_raw_code=[]
X_sample_timestep=[]
"""
Video file level containing BOVW
"""

print("Video file level containing BOVW")
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
     
 

 

print('Evaluate IRNN...')
model = Sequential()

model.add(LSTM(output_dim=hidden_units,
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


model.add(LSTM(output_dim=hidden_units,
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
 
 
 
 
 
 
 
 
 
 
 
 
 
