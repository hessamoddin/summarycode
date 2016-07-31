from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.io as matreader
import pprint
from os import listdir
from os.path import isfile, join
import tflearn
import pylab
import imageio
from skimage.transform import resize
from skimage.color import rgb2gray


# To split video into different evenly sized set of frames to feed into LSTMs
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
        
# Load this video file
videofilename='/home/hessam/code/data/videos/Air_Force_One.mp4'
vid = imageio.get_reader(videofilename,  'ffmpeg')
# number of frames in video
num_frames=vid._meta['nframes']

######## Extract frame features ##############

#Subsample the video
starting_frame=1
ending_frame=num_frames
step=80  #sampling step
num_LSTMs=10  #number of LSTMs per video
sampling_id=np.arange(starting_frame,ending_frame,step) 
video_sequence_frameid=list(chunks(sampling_id, num_LSTMs)) #batch of video sequence of frame ids
batch_size=len(video_sequence_frameid)  # batch size: number of rows of sequential data to be fed to LSTMs


######## Load Video Clip  Raw Data ##############

#10th frame--should be changed
frame = vid.get_data(10) 
#resize frame
frame_resized=resize(frame, (100, 100))
#convert the color frame to gray-scale
frame_gray= rgb2gray(frame_resized)






# Read the summary file
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



# Iterate over keyframes of all segments of videos
for i in range(0,num_splitted_video):
  print(i)
