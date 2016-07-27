import numpy as np
import scipy.io as matreader
import pprint
from os import listdir
from os.path import isfile, join


# To split video into different evenly sized set of frames to feed into LSTMs
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


# Read the summary file
data_path='/home/hessam/code/data/GT'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
num_videos=len(onlyfiles)
current_file=onlyfiles[0]
full_path=join(data_path,onlyfiles[0])
loaded_summary=matreader.loadmat(full_path) # change the filename
nFrames=loaded_summary['gt_score'].shape[0]
summary_score=loaded_summary['gt_score']

#Subsample the video
starting_frame=1
ending_frame=nFrames
step=80  #sampling step
num_LSTMs=10  #number of LSTMs per video
sampling_id=np.arange(starting_frame,ending_frame,step)
splitted_subsampled_summary_id=list(chunks(sampling_id, num_LSTMs))
num_splitted_video=len(splitted_subsampled_summary_id)

# Iterate over keyframes of all segments of videos
for i in range(0,num_splitted_video):
  print(i)
