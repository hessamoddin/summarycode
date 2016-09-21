
###  Summarization algorithm by Hessamoddin Shafeian
############ Load Libraries ##############
from __future__ import division, print_function, absolute_import
import sklearn.mixture.gmm as gm
from tempfile import TemporaryFile
from sklearn import mixture
import numpy as np
import scipy.io as matreader
from scipy.stats import multivariate_normal
import pprint
import logging
#import cv2
from os import listdir
from os.path import isfile, join
from os import path
import tflearn
import pylab
import imageio
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import daisy
import matplotlib.cbook as cbook
from sklearn.cluster import KMeans
import math

 # # Reference for GMM and Kmeans with a bit of modification: https://github.com/rkwitt/pyfsa/blob/master/core/fsa.py
# Choose default codebook size and keep it the same for both GMM and K-means method of encoding
def learn_codebook(X, codebook_size=1000, seed=None):
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
def bow(X, cb):
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
    if d != X.shape[1]:
        raise Exception("Dimensionality mismatch!")
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

	

############ Extract frame features ##############
def Feature_Extractor_Fn(vid,num_frames,frame_no,new_shape=(120,180),step=50, radius=20):
    if frame_no<num_frames: 
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







def Video_Feature_Extractor_Daisy(videofilename,step=1,num_LSTMs=10):
	vid = imageio.get_reader(videofilename,  'ffmpeg')
# number of frames in video
	num_frames=vid._meta['nframes']

############ Extract frame features ##############
#Subsample the video
	starting_frame=1
	ending_frame=num_frames
	num_LSTMs=10  #number of LSTMs per video
	sampled_frame_id=np.arange(starting_frame,ending_frame,step) 
	video_sequence_frameid=list(chunks(sampled_frame_id[0:int(len(sampled_frame_id)/num_LSTMs)*num_LSTMs], num_LSTMs))  #batch of video sequence of frame ids
	batch_size=len(video_sequence_frameid)  # batch size: number of rows of sequential data to be fed to LSTMs
 
#Should be zero-padded for same length 
 
	daisy_list=[]
 
# Feature extraction
	for i in xrange(batch_size):
		print(100*i/batch_size)
		for j in xrange(num_LSTMs):
	 		current_frame_id=video_sequence_frameid[i][j]
	 		if len(video_sequence_frameid[i])==num_LSTMs:
        #    daisy_1D,surf_descs,sift_descs=current_feature=Feature_Extractor_Fn(vid,current_frame_id)
		 		daisy_1D=current_feature=Feature_Extractor_Fn(vid,num_frames,current_frame_id)
       				daisy_list.append(daisy_1D)
 
	daisy_arr=np.asarray(daisy_list)
	return daisy_arr
############ Bovw Construction ##############

# Training videos only are used 
# Training videos should be splitted as the size of datasets grows
# For now only daisy features are used 
	
 
        
############################################
########### END of Functions ###############
############################################
############################################

############ Load Video ##############
datasetpath='/home/hessam/code/Tour20/Tour20-Videos/AW/'
onlyfiles = [f for f in listdir(datasetpath) if isfile(join(datasetpath, f))]
daisy_list_total=[]

for videofilename in onlyfiles:
	print(videofilename)
	videofilename=path.join(datasetpath,videofilename)
	daisy_arr=Video_Feature_Extractor_Daisy(videofilename,step=1,num_LSTMs=10)
	daisy_bovw_training=daisy_arr

# first method of bovw calculation: kmeans
	codebook_size=int(math.floor(math.sqrt((daisy_bovw_training.shape[0]))))
	codebook=learn_codebook(daisy_bovw_training, codebook_size)
	kmeans_bovw=bow(daisy_arr, codebook)


# first method of bovw calculation: GMM (fisher vector)
	m,c,w=estimate_gm(daisy_bovw_training,codebook_size)


	np.savetxt(path.splitext(videofilename)[0]+'.csv', kmeans_bovw, delimiter=",")
	daisy_list_total.append(daisy_list)
	print(path.splitext(videofilename)[0]+'.csv')
	print(len(daisy_list_total))
############ Load Summary File ##############
data_path='/home/hessam/code/data/GT'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
num_videos=len(onlyfiles)
current_file=onlyfiles[0]
full_path=join(data_path,onlyfiles[0])
loaded_summary=matreader.loadmat(full_path) # change the filename
nFrames=loaded_summary['gt_score'].shape[0]
summary_score=loaded_summary['gt_score']



# LSTM input size: Batch size (num sequesnces/rows) [if None, it can be changed] X 
#   Sequence length X Dimension of each member of sequence

net = tflearn.input_data(shape=[None, num_LSTMs, codebook_size])
net = tflearn.lstm(net, 32, return_seq=True)
net = tflearn.lstm(net, 32)
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                         loss='categorical_crossentropy', name="output1")
model = tflearn.DNN(net, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1, validation_set=0.1, show_metric=True,
          snapshot_step=100)


 
