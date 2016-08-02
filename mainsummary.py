
###  Summarization algorithm by Hessamoddin Shafeian
############ Load Libraries ##############
from __future__ import division, print_function, absolute_import
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

 
