from __future__ import division, print_function, absolute_import
from __future__ import print_function

from os.path import isfile, join
from os import path,listdir
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.util import view_as_blocks
from skimage.feature import daisy
import numpy as np
import os
import os.path
import imageio
import pickle
import tables as tb
 

"""       
Parameters
"""
subsampling_rate=3
bovw_size=30
new_shape,step,radius=(240,360),50,20 # for Daisy feaure
N=5


 


"""       
Define functions
"""

  

 

def Feature_Extractor_Fn(vid,num_frames,frame_no,N,new_shape=(360,480),step=60, radius=40):
    N=5
    """Extract Daisy feature for a frame of video """
    if frame_no<num_frames-1: 
        frame = vid.get_data(frame_no)  
        frame_resized=resize(frame, new_shape)
        frame_gray= rgb2gray(frame_resized)
        daisy_desc = daisy(frame_gray,step=step, radius=radius)
        daisy_1D=np.ravel(daisy_desc)
         
        """Extract Daisy feature for a patch from the frame of video """
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

 
     
 
 
     

class framefeature_hdf(tb.IsDescription):
    filename        = tb.StringCol(200, pos=1) 
    category        = tb.StringCol(10,pos=2)        
    rawfeature      = tb.Float32Col(shape=(1,7000), pos=3) 
    bovw_id         = tb.IntCol(pos=4) 
    frame_id        = tb.IntCol(pos=5)  
    grid_feature    = tb.Float32Col(shape=(N,N,7000), pos=6) 

fileh = tb.open_file('videofeatures4.h5', mode='w')
table = fileh.create_table(fileh.root, 'table', framefeature_hdf,"A table") 

 

""" ************************
****************************
Main body of the code
***************************
************************""" 

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
                      # trim out the bag of videos frames in a way that each
                     #have bags number equal to multiples of bovw_size
                       # j is the frame index for the bvw processable parts of video
                       # 
                     for j in xrange(0,min(num_frames,4000),subsampling_rate):
                         bovw_id=(i)//bovw_size  # every bovw_size block of frames
                          
                         
                            # Feature extraction
                            # daisy_1D,surf_descs,sift_descs 		
                         # extract dausy features: for the whole frame or grid-wise for each frame
                         current_grid_feature,current_frame_feature=Feature_Extractor_Fn(vid,num_frames,j,N) 
                         table.append([(videopath,cat,current_frame_feature,bovw_id,i,current_grid_feature)]) #filename,category,rawfeature,bovw_id,frame_id,griddedfeature
                         

                        # print(i)
                         #print(bovw_id)
                         #print(j*subsampling_rate)
                         i=i+1
                         file_counter.append(videopath)
                         # Track record of which video does this frame belong toin a list
                         
                         # update feature objects for each video
                     #pickle.dump(framefeature, open( "raw_features_Class_array.p", "wb" ) )
                     #framefeature_loaded = pickle.load( open( "raw_features_Class_array.p", "rb" ) )
                 except:
                     print("error on video")
                     print(current_file)
                     print("***")
print("Finished raw feature extraction!")
file_counter=list(set(file_counter))
pickle.dump( file_counter, open( "file_counter.p", "wb" ) )
pickle.dump( dirs, open( "dirs.p", "wb" ) )


 
  

 


fileh.close()

 

  
 
   
fileh = tb.open_file('videofeatures4.h5', mode='r')
table_root=fileh.root.table
current_row=table_root[0]
print(current_row)
rf=current_row["rawfeature"]
print(rf)
fileh.close()


 
