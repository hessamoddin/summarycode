import numpy as np
import scipy.io as matreader

# Read the summary file
loaded_summary=matreader.loadmat('/home/hessam/code/data/GT/Air_Force_One.mat')
nFrames=loaded_summary['gt_score'].shape[0]
summary_score=nFrames=loaded_summary['gt_score']

#Subsample the video
