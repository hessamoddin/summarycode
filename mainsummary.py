import numpy as np
import scipy.io as matreader

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
selected_keyframes=summary_score[sampling_id]
