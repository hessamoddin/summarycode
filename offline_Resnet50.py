# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 17:04:04 2018

@author: hessam
"""
import os ,glob
from keras.applications.resnet50 import ResNet50
import numpy as np
import imageio
import cv2
import tables as tb

 
 
database_path="/home/hessam/Activity_recognition/HMDB" 

vid_ext=".avi"
max_num_cats=20
max_num_files_per_cat=20
subsample_rate=5
Num_LSTMs=8
max_vid_len=Num_LSTMs* subsample_rate # 8 sample frames per videoss

model = ResNet50(weights='imagenet',include_top=False)


"""
def Resnet_Extractor():
    img_path = 'diamond_head.JPG'
    img = image.load_img(img_path, target_size=(224, 224))
    img_expanded = np.expand_dims(image.img_to_array(img), axis=0)
    img_processed = preprocess_input(img_expanded)
    resnet50_feature = model.predict(img_processed)[0][0]
    return resnet50_feature
"""


def Vid_Proc(videopath,rawfeaturetable,current_category):
    vid = imageio.get_reader(videopath,  'ffmpeg')
    num_frames=vid._meta['nframes']
    print(num_frames)
    if num_frames>max_vid_len-1:
        for i in xrange(0,max_vid_len,subsample_rate):
            frame_data = vid.get_data(i) 
            frame_resized = cv2.resize(frame_data,(224, 224)) 
            frame_resized_expand = np.expand_dims(frame_resized, axis=0)
            resnet50_frame_feature = model.predict(frame_resized_expand)[0][0]
            resnet50_frame_feature=resnet50_frame_feature.reshape(1,2048)
        
            rawfeaturetable.append([(current_category,videopath,i,resnet50_frame_feature)])
            rawfeaturefileh.flush()
        rawfeaturefileh.close()
    else:
        rawfeaturefileh.close()
        
        
         
     
    
    


class framefeature_hdf(tb.IsDescription):
    category        = tb.StringCol(20,pos=1)   
    filepath        = tb.StringCol(250, pos=2) 
    frame_no         = tb.Int32Col((1,2048), pos=3) 
    rawfeature      = tb.Float32Col((1,2048), pos=4) 



#




cat_folders=os.listdir(database_path)
cat_folders=sorted(cat_folders)
num_cats=len(cat_folders)






for i in xrange(max_num_cats):
    
    current_cat=cat_folders[i]
    print("*********")
    print("Processing "+current_cat+" videos ...")
    current_cat_path=os.path.join(database_path,current_cat)
    files_in_cat=glob.glob(current_cat_path+'/*'+vid_ext) 
    
    

    for j in xrange(max_num_files_per_cat):

        current_file=files_in_cat[j]
        current_file_path=os.path.join(current_cat_path,current_file)
        rawfeatures_path=(os.path.splitext(current_file_path)[0]+'.h5')
        rawfeaturefileh = tb.open_file(rawfeatures_path, mode='w')
        rawfeaturetable = rawfeaturefileh.create_table(rawfeaturefileh.root, 'table', framefeature_hdf,"A table")
        print(current_file_path)
        Vid_Proc(current_file_path,rawfeaturetable,current_cat)
 
        
         
        
                
