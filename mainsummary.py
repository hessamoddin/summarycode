import numpy as np
import scipy.io as matreader
import pprint

# To split video into different evenly sized set of frames to feed into LSTMs
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


# Read the summary file
loaded_summary=matreader.loadmat('/home/hessam/code/data/GT/Air_Force_One.mat')
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

for i in range(0,num_splitted_video):
  print(i)
