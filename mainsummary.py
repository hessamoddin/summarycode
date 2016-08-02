
###  Summarization algorithm by Hessamoddin Shafeian
############ Load Libraries ##############
from __future__ import division, print_function, absolute_import
import sklearn.mixture.gmm as gm
import numpy as np
import scipy.io as matreader
import pprint
from os import listdir
from os.path import isfile, join
import cv2
import tflearn
import pylab
import imageio
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import daisy
import matplotlib.cbook as cbook
from sklearn.cluster import KMeans

# Reference: https://github.com/rkwitt/pyfsa/blob/master/core/fsa.py

def estimate_gm(X,components=3,seed=None):
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
    return gm_obj


def learn_codebook(X, codebook_size=200, seed=None):
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
def Feature_Extractor_Fn(vid,frame_no,new_shape=(120,180),step=50, radius=20):
    if frame_no<num_frames: 
        frame = vid.get_data(frame_no)  
        frame_resized=resize(frame, new_shape)
        frame_gray= rgb2gray(frame_resized)
        daisy_desc = daisy(frame_gray,step=step, radius=radius)
        daisy_1D=np.ravel(daisy_desc)
        sift = cv2.xfeatures2d.SIFT_create()
        (sift_kps, sift_descs) = sift.detectAndCompute(frame, None)
        print("# kps: {}, descriptors: {}".format(len(sift_kps), sift_descs.shape))
        surf = cv2.xfeatures2d.SURF_create()
        (surf_kps, surf_descs) = surf.detectAndCompute(frame, None)
        print("# kps: {}, descriptors: {}".format(len(surf_kps), surf_descs.shape))
    else:
        print("Frame number is larger than the length of video")
    return (daisy_1D,surf_descs,sift_descs)
    



# To split video into different evenly sized set of frames to feed into LSTMs
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
        
############ Load Video ##############
videofilename='/home/hessam/code/data/videos/Air_Force_One.mp4'
vid = imageio.get_reader(videofilename,  'ffmpeg')
# number of frames in video
num_frames=vid._meta['nframes']

############ Extract frame features ##############
#Subsample the video
starting_frame=1
ending_frame=num_frames
step=80  #sampling step
num_LSTMs=10  #number of LSTMs per video
sampling_id=np.arange(starting_frame,ending_frame,step) 
video_sequence_frameid=list(chunks(sampling_id, num_LSTMs)) #batch of video sequence of frame ids
batch_size=len(video_sequence_frameid)  # batch size: number of rows of sequential data to be fed to LSTMs
 
 
daisy_list=[]
 
# Feature extraction
for i in xrange(batch_size):
    for j in xrange(num_LSTMs):
        print(j)
        current_frame_id=video_sequence_frameid[i][j]
        if len(video_sequence_frameid[i])==num_LSTMs:
            daisy_1D,surf_descs,sift_descs=current_feature=Feature_Extractor_Fn(vid,current_frame_id)
            daisy_list.append(daisy_1D)
 
daisy_arr=np.asarray(daisy_list)
############ Bovw Construction ##############

# Training videos only are used 
# Training videos should be splitted as the size of datasets grows
# For now only daisy features are used 
daisy_arr_bovw_training=daisy_arr

N_kmeans=7
bovw_kmeans=[]
kmeans = KMeans(n_clusters=N_kmeans, random_state=0).fit(daisy_arr_bovw_training)
kmeans.fit(daisy_arr)
bovw_cookbooks = kmeans.cluster_centers_.squeeze()
kmeans_labels = kmeans.predict(daisy_arr)



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

 
