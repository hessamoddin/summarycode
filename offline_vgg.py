from __future__ import division, print_function, absolute_import
from __future__ import print_function
from keras.applications.vgg19 import VGG19
from os.path import isfile, join
from skimage.transform import resize
from skimage.util import view_as_blocks
import numpy as np
import os
import os.path
import imageio
import pickle
import warnings
import cv2
import glob


from keras.layers import Input
from keras import layers
from os import path,listdir
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
import tables as tb



WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

dir_var= "dirs8.p"
file_counter_str="file_counter8.p"
framefeatures='framefeatures8.h5'
folder='Tour20-Videos8'

N=4

class framefeature_hdf(tb.IsDescription):
    filename        = tb.StringCol(200, pos=1) 
    category        = tb.StringCol(10,pos=2)        
    rawfeature      = tb.Float32Col(1000, pos=3) 
    bovw_id         = tb.IntCol(pos=4) 
    frame_id        = tb.IntCol(pos=5)  
    griddedfeature    = tb.Float32Col(shape=(N*N,1000), pos=6) 


fileh = tb.open_file(framefeatures, mode='w')
table = fileh.create_table(fileh.root, 'table', framefeature_hdf,"A table") 

"""       
Parameters
"""
subsampling_rate=2
bovw_size=15
 

"""
Define HDF database for frame features
"""

 
"""       
Define functions
"""

  

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


 

def VGG_Feature_Extractor_Fn(model,vid,num_frames,frame_no,N):


    N=4
    """Extract Daisy feature for a frame of video """
    if frame_no<num_frames-1: 
        frame = vid.get_data(frame_no).astype('float32')
        frame_resized = cv2.resize(frame,(224, 224)) 
       # frame_resized = np.swapaxes(np.swapaxes(frame_resized, 1, 2), 0, 1)
        frame_resized_expand = np.expand_dims(frame_resized, axis=0)
        processed_img = preprocess_input(frame_resized_expand)
        vgg19_holistic = np.ravel(model.predict(processed_img))

    
     
        """Extract Daisy feature for a patch from the frame of video """

        
        frame_resized = cv2.resize(frame,(N*224, N*224)) 
         
       
       
       
        frame_resized_expand = np.expand_dims(frame_resized, axis=0)
        processed_to_patch = preprocess_input(frame_resized_expand)
        processed_to_patch = preprocess_input(processed_to_patch)
        
        
        vgg19_patchy=np.zeros((N*N,1000))
        patch_ind=0


        for i in xrange(N):
            for k in xrange(N):
                patch=processed_to_patch[:,i*224:(i+1)*224,k*224:(k+1)*224,:]
                vgg19_patchy[patch_ind,:] = np.ravel(model.predict(patch))
                patch_ind=patch_ind+1
                 
 
    else:
        print("Frame number is larger than the length of video")
  #  return (daisy_1D,surf_descs,sift_descs)
    return vgg19_patchy,vgg19_holistic



 
def Resnet_Feature_Extractor_Fn(vid,num_frames,frame_no,N,new_shape=(224,224),step=60, radius=40):
    new_shape=(224,224)
    model = ResNet50(include_top=True, weights='imagenet')
    frame = vid.get_data(frame_no)  
    frame_resized=resize(frame, (224,224))
    frame_resized_float=frame_resized.astype('float64')
 
    
    
    
    
   # img_path = 'cat.jpg'
    #img = image.load_img(img_path, target_size=(224, 224),grayscale=False)
    #x = image.img_to_array(img)




    frame_resized_expand = np.expand_dims(frame_resized_float, axis=0)
    x = preprocess_input(frame_resized_expand)
    print('Input image shape:', x.shape)
    preds = model.predict(x)
 
         
       #sift = cv2.xfeatures2d.SIFT_create()
       # (sift_kps, sift_descs) = sift.detectAndCompute(frame, None)
       # print("# kps: {}, descriptors: {}".format(len(sift_kps), sift_descs.shape))
       # surf = cv2.xfeatures2d.SURF_create()
      #  (surf_kps, surf_descs) = surf.detectAndCompute(frame, None)
       # print("# kps: {}, descriptors: {}".format(len(surf_kps), surf_descs.shape))
       #  return (daisy_1D,surf_descs,sift_descs)
    patch_shape_x=int(new_shape[0]/N)
    patch_shape_y=int(new_shape[1]/N)

    R=frame_resized_float[:,:,0]
    G=frame_resized_float[:,:,1]
    B=frame_resized_float[:,:,2]


    patchs_arr_R = view_as_blocks(R, (patch_shape_x,patch_shape_y))
    patchs_arr_G = view_as_blocks(G, (patch_shape_x,patch_shape_y))
    patchs_arr_B = view_as_blocks(B, (patch_shape_x,patch_shape_y))

    patch_num_row=patchs_arr_R.shape[0]
    patch_num_col=patchs_arr_R.shape[1]
    
    
 
        
    
    final_resnet_length=1000
    
    np.zeros((2,2),dtype='float64')
    patch_resnet_arr=np.zeros((patch_num_row,patch_num_col,final_resnet_length))
    
    
    patch_R=np.zeros((224,224),dtype='float64')
    patch_G=np.zeros((224,224),dtype='float64')
    patch_B=np.zeros((224,224),dtype='float64')
    
    ind=0
    
    for i in xrange(patch_num_row):
            for k in xrange(patch_num_col):
                patch_R[0:patchs_arr_R.shape[2],0:patchs_arr_R.shape[3]]=patchs_arr_R[i,k,:,:]
                patch_G[0:patchs_arr_G.shape[2],0:patchs_arr_G.shape[3]]=patchs_arr_G[i,k,:,:]
                patch_B[0:patchs_arr_B.shape[2],0:patchs_arr_B.shape[3]]=patchs_arr_B[i,k,:,:]
    
                patch=np.dstack((patch_R,patch_G,patch_B))
                frame_resized_expand = np.expand_dims(patch, axis=0)
                x = preprocess_input(frame_resized_expand)
                patch_resnet_arr[i,k,:]=model.predict(x)
                ind=ind+1
  
           
                
        
       #sift = cv2.xfeatures2d.SIFT_create()
       # (sift_kps, sift_descs) = sift.detectAndCompute(frame, None)
       # print("# kps: {}, descriptors: {}".format(len(sift_kps), sift_descs.shape))
       # surf = cv2.xfeatures2d.SURF_create()
      #  (surf_kps, surf_descs) = surf.detectAndCompute(frame, None)
       # print("# kps: {}, descriptors: {}".format(len(surf_kps), surf_descs.shape))
   #  return (daisy_1D,surf_descs,sift_descs)
    return patch_resnet_arr,preds



       
       
       
       
       
       
       
       
       
       
       
       
    return preds
     
 
 
     


 

""" ************************
****************************
Main body of the code
***************************
************************""" 








cwd = os.getcwd()
# The folder inside which the video files are located in separate folders
parent_dir = os.path.split(cwd)[0] 
# Find the data folders
datasetpath=join(parent_dir,'Tour20/',folder)
# Dir the folders; each representing a category of action
dirs = os.listdir( datasetpath )


"""
ssh -i ubuntu@ec2-54-157-195-178.compute-1.amazonaws.com
sudo pip install glove_python
sudo pip install imageio
sudo pip install scikit-image

cat=dirs[0]
cat_path=join(datasetpath,cat)
onlyfiles = [f for f in listdir(cat_path) if isfile(join(cat_path, f))]
current_file = onlyfiles[0]
videopath=path.join(cat_path,current_file)
vid = imageio.get_reader(videopath,  'ffmpeg')
cat=dirs[0]
cat_path=join(datasetpath,cat)
onlyfiles = [f for f in listdir(cat_path) if isfile(join(cat_path, f))]
current_file = onlyfiles[0]
videopath=path.join(cat_path,current_file)
vid = imageio.get_reader(videopath,  'ffmpeg')

num_frames=vid._meta['nframes']
bovw_id=0
j=0
i=0
vgg_model = VGG19(weights='imagenet')
vgg19_patchy,vgg19_holistic=VGG_Feature_Extractor_Fn(vgg_model,vid,num_frames,j,N) 
"""
 
print("Thus begins feature extraction!")
i=0

file_counter=[]
csv_file_counter=[]
vgg_model = VGG19(weights='imagenet')

# cat: categort of actions, also the name of the folder containing the action videos
for cat in dirs:
    print("Processing  %s Videos...." % (cat))    
    print(cat)
    if "." not in cat:
        cat_path=join(datasetpath,cat)
        onlyfiles = [f for f in listdir(cat_path) if isfile(join(cat_path, f))]
        for current_file in onlyfiles:
            fileh.close()
            fileh = tb.open_file(framefeatures, mode='a')
            table_root=fileh.root.table


		  # This dataset contains only mp4 video clips
            if current_file.endswith('.mp4'):
                 print("***")
                 print(current_file)
                 filename_no_ext, ext = os.path.splitext(current_file)
                 # full name and path for the current video file
                 videopath=join(cat_path,current_file)
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
                     for j in xrange(0,min(5*subsampling_rate*bovw_size*8,num_frames),subsampling_rate):
                         bovw_id=(i)//bovw_size  # every bovw_size block of frames
                          
                         print(j)
                            # Feature extraction
                            # daisy_1D,surf_descs,sift_descs 		
                         # extract dausy features: for the whole frame or grid-wise for each frame
                       
                         #current_grid_feature,current_frame_feature=VGG_Feature_Extractor_Fn(vid,num_frames,j,N) 
                         vgg19_patchy,vgg19_holistic=VGG_Feature_Extractor_Fn(vgg_model,vid,num_frames,j,N) 
                          
 
                         table_root.append([(videopath,cat,vgg19_holistic,bovw_id,i,vgg19_patchy)]) #filename,category,rawfeature,bovw_id,frame_id,griddedfeature
                         

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
pickle.dump( file_counter, open(file_counter_str, "wb" ) )
pickle.dump( dirs, open(dir_var, "wb" ) )


 
  

 


fileh.close()

 

  
 
   
fileh = tb.open_file(framefeatures, mode='a')
table_root=fileh.root.table
current_row=table_root[0]
print(current_row)
fileh.close()


     
