from __future__ import print_function
from __future__ import division, print_function, absolute_import

from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.cluster import MiniBatchKMeans, KMeans
from random import sample
import random
from glove import Glove
from sklearn.cross_validation import train_test_split

import array
import logging
import sklearn.mixture.gmm as gm
import numpy as np
import pickle
import scipy.sparse as sp
import tables as tb
import warnings
import os,glob
import os.path
import math
import pandas as pd
from collections import OrderedDict,defaultdict

from numpy import dot
from numpy.linalg import norm

from keras.layers import Input
from keras import layers
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
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


 
"""
vgg_feature_file_list=[]
for vgg_feature_file in files:
    if os.stat(vgg_feature_file).st_size>100000:
        vgg_feature_file_list.append(vgg_feature_file)
        
        
for vgg_feature_file in vgg_feature_file_list:      
        
"""        
        
        
        
        
"""       
Parameters
"""
Num_samples_per_video=5
bovw_size=20
num_LSTMs=8
train_frac=0.5
longest_allowed_frames=500
batch_size = 1
nb_epochs = 200
hidden_units = 6
learning_rate = 1e-6
clip_norm = 1.0
embedding_size=100
N=4


filecounter_str="file_counter8.p"
framefeatures='framefeatures8_vgg.h5'
gridfeatures='gridfeatures8.h5'
gridded_bovwfeatures='gridded_bovwfeatures8.h5'
glovefeatures='glovefeatures8.h5'
glovefeatures_test='glovefeatures8_test.h5'

bovwfeatures='bovwfeatures8.h5'
dir_str="dirs8.p"
bovwfeatures="bovwfeature.h5"

  
def hist_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection
  
 
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

 
class videofile(object):
    """Class of Video file object"""
    def __init__(self, contained_bovws=None,category=None,filename=None):
        self.contained_bovws = contained_bovws
        self.category=category
        self.filename = filename
 


"""       
Define functions
"""

def embedding_func(gridded_words_overall,embedding_size):
    
    """***************
     GLOVE for Video
     ***************"""
     
    
    glove_bins=np.asarray(gridded_words_overall)
    print(glove_bins)
    glove_shape=glove_bins.shape
    glove_weights=np.ones((glove_shape))
    #bovw_shape=(3,5)
    #bovw_bins = np.random.randint(9,13, size=bovw_shape)
    #bovw_weights = np.random.randint(2, size=bovw_shape)
    
    
    
    
    #print('Bovw bins')
    #print(bovw_bins)
    #print('Bovw weights')
    #print(bovw_weights)
     
    
    
    
    
    dictionary = {}
    rows = []
    cols = []
    data = array.array('f')
     
    k=0 
    #print(bovw_bins)
    
    for frame in glove_bins:
            for i, first_word in enumerate(frame):
                first_word_idx = dictionary.setdefault(first_word,
                                                       len(dictionary))
                w1=glove_weights[k,i]                                    
                for j, second_word in enumerate(frame):
                    second_word_idx = dictionary.setdefault(second_word,
                                                            len(dictionary))
                    w2=glove_weights[k,j]            
                    distance = 1
                    w=w1*w2
    
                    if first_word_idx == second_word_idx:
                        pass
                    elif first_word_idx < second_word_idx:
                        rows.append(first_word_idx)
    
                        cols.append(second_word_idx)
                        data.append(np.double(w*np.double(1.0) / distance))
                    else:
                        rows.append(second_word_idx)
                        cols.append(first_word_idx)
                        data.append(np.double(w*np.double(1.0) / distance))
            k=k+1
         
                            
     
    
    x=sp.coo_matrix((data, (rows, cols)),
                             shape=(len(dictionary),
                                    len(dictionary)),
                             dtype=np.double).tocsr().tocoo()      
    print(dictionary)     
           
 
    
                  
    xarr=x.toarray()                         
    xarr/=np.amax(xarr)
    print("coocurance matrix")
    print(xarr)
    xsparse=sp.coo_matrix(xarr)   
    
    glove_model = Glove(no_components=embedding_size, learning_rate=0.05)
    glove_model.fit(xsparse,
                        epochs=500,
                        no_threads=2)
    
    
    new_word_representation=glove_model.word_vectors


    return new_word_representation,dictionary
    

def learn_kmeans_codebook(X, codebook_size=1000, seed=None):
    """ Learn a codebook.
    source: https://github.com/KitwareMedical/TubeTK/blob/master/Base/Python/pyfsa/core/fsa.py
    """
    logger = logging.getLogger()
    logger.info("Learning codebook with %d words ..." % codebook_size)
    # Run vector-quantization
                
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=codebook_size, batch_size=3000,
                      n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit(X)

                
    return mbk

def chunks(l, n):
    """
    To split video into different evenly sized set of frames to feed into LSTMs    
    Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

 
def calc_bovw(X, cb):
    """Compute a (normalized) BoW histogram.
   source: https://github.com/rkwitt/pyfsa/blob/master/core/fsa.py
    """
    # Get nr. codewords
    n,d = cb.cluster_centers_.shape
    # Compute closest cluster centers
    assignments = cb.predict(X)
    # Compute (normalized) BoW histogram
    B = range(0,n+1)
    return np.histogram(assignments,bins=B,density=True)[0]

 

def estimate_gm(X,components=1000,seed=None):
    """Estimate a Gaussian mixture model.
    source: https://github.com/rkwitt/pyfsa/blob/master/core/fsa.py
    """
    logger = logging.getLogger()
    n, d = X.shape
    logger.info("Estimating %d-comp. GMM from (%d x %d) ..." %
                (components, n, d))
    gm_obj = gm.GMM (n_components=components,
                     covariance_type='diag',
                     random_state=seed)
    gm_obj.fit(X)   
    np.random.seed(1)
    return  np.float32(gm_obj.means_), np.float32(gm_obj.covars_), np.float32(gm_obj.weights_)




"""
Fisher Vector
https://github.com/jacobgil/pyfishervector/blob/master/fisher.py
"""
     
def likelihood_moment(x, ytk, moment):    
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk
    
def likelihood_statistics(samples, means, covs, weights):
    gaussians, s0, s1,s2 = {}, {}, {}, {}
#    samples = zip(range(0, len(samples)), samples)
    
    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights))]
    for index, x in samples:
        for k in range(0, len(weights)):
            gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

    return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples, means, covs, w):
    s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    return fv


videofile=[ videofile() for i in range(1000000)]

    
 
framefileh = tb.open_file(framefeatures, mode='r')
frametable=framefileh.root.table

num_rows=frametable[0]['griddedfeature'].shape[0]
num_cols=frametable[0]['griddedfeature'].shape[1]




class glovefeature_hdf(tb.IsDescription):
    category        = tb.StringCol(10,pos=1) 
    filename        = tb.StringCol(200,pos=2) 
    bovw_id         = tb.Int32Col(pos=3) 
    word            = tb.Int32Col(pos=4) 
    grid_word            = tb.Int32Col(num_rows,pos=5) 
                            
                      


gloveh = tb.open_file(glovefeatures, mode='w')
gloveh_test = tb.open_file(glovefeatures_test, mode='w')


glovetable = gloveh.create_table(gloveh.root, 'table', glovefeature_hdf,"A table") 
glovetable_test = gloveh.create_table(gloveh_test.root, 'table', glovefeature_hdf,"A table") 

   

unique_filenames,file_indices=np.unique(frametable[:]['filename'],return_index=True)
num_videos=len(unique_filenames)
unique_filenames=unique_filenames[file_indices.argsort()] 
file_indices=np.sort(file_indices)
file_indices_appended=np.append(file_indices,frametable.nrows)





dirs = pd.unique(frametable[:]['category'])
unique_categories,cat_indices=np.unique(frametable[:]['category'],return_index=True)
num_cats=len(unique_categories)
unique_categories=unique_categories[cat_indices.argsort()] 
cat_indices=np.sort(cat_indices)
cat_indices_appended=np.append(cat_indices,frametable.nrows)


cat_intervals= zip(cat_indices_appended, cat_indices_appended[1:]-1)
file_intervals=zip(file_indices_appended,file_indices_appended[1:]-1) 


train_ind=[]
test_ind=[]

cat_per_file=[]
for i in xrange(num_videos):
    #file_indices_appended[i]
    cat_per_file.append(frametable[int(np.mean(file_intervals[i]))]['category'])
     
    
# 
tuple_cat_inds=zip(cat_per_file,file_indices_appended,file_indices_appended[1:]-1)

cat_interval_dict={}
 
for current_tuple in tuple_cat_inds:
    current_cat=current_tuple[0]
    frame_interval=(current_tuple[1],current_tuple[2])
    if current_cat in cat_interval_dict:
        cat_interval_dict[current_cat].append(frame_interval)
    else:
        cat_interval_dict[current_cat]=[frame_interval]

        
        
train_dict={}
test_dict={}        
for cat in cat_interval_dict:
    intervals=cat_interval_dict[cat]
    num_intervals=len(intervals)
    num_train=int(train_frac*num_intervals)
    num_test=num_intervals-num_train
    ind_train=np.random.choice(num_intervals, num_train, replace=False)
    ind_test=np.delete(range(num_intervals),ind_train)
    train_intervals=[]
    test_intervals=[]
    for k in ind_train:
        train_intervals.append(intervals[k])
    for k in ind_test:
        test_intervals.append(intervals[k])
        
    train_dict[cat]=train_intervals
    test_dict[cat]=test_intervals

        
    
       
num_train_samples=0
train_ind_kmeans=[]
for cat in train_dict:
    for interval_id in xrange(len(train_dict[cat])):
        current_interval=train_dict[cat][interval_id]
        k=0
        for j in xrange(current_interval[0],current_interval[1]+1):
            train_ind_kmeans.append(j)
            k=k+1
        num_train_samples=num_train_samples+abs(current_interval[1]-current_interval[0])+1
        
    

kmeans_codebook_size=int(math.sqrt(math.floor(num_train_samples)))
 
kmeans_codebook=learn_kmeans_codebook(frametable[train_ind_kmeans]['rawfeature'], kmeans_codebook_size)
    
 






print("Starting converting raw features to visual words ... ")
gridded_words_overall=[]  # All
bovw_id=-1
categorized_frames_dict={}
categorized_frames_id=[]
num_frames_overall=0
frame_id=0
for cat in train_dict:
    categorized_frames=[]
    
    for interval_id in xrange(len(train_dict[cat])):
        current_interval=train_dict[cat][interval_id]
        file_ind_before=current_interval[0]
        file_ind_after=current_interval[1]
            
        num_frames_overall=num_frames_overall+abs(current_interval[1]-current_interval[0])+1
        
        current_bovw_indices= range(file_ind_before,file_ind_after+2,bovw_size)

        for bovw_ind_before, bovw_ind_after in zip(current_bovw_indices, current_bovw_indices[1:]):
             bovw_id=bovw_id+1
             glovetable.flush()

             current_contained_frames=range(bovw_ind_before,bovw_ind_after)
            
        
             num_frames_overall=num_frames_overall+len(current_contained_frames)
         
            # take the middle frame of the bag of visual words as its examplar
             middle_frame=current_contained_frames[len(current_contained_frames)//2]
         
             current_category=frametable[middle_frame]['category']
             current_filename=frametable[middle_frame]['filename']
             

             for j in current_contained_frames:
                 categorized_frames.append(j)
                 frame_id=frame_id+1
                 current_frame_feature=frametable[j]['rawfeature']
                 current_frame_feature=current_frame_feature.reshape(1,-1)
                 current_frame_word = kmeans_codebook.predict(current_frame_feature)[0]
                 
                 
                 current_gridded_frame_feature=frametable[j]['griddedfeature']
                 grid_word=[]
                 for ind in xrange(num_rows):
                     current_grid_feature=current_gridded_frame_feature[ind,:]
                     current_grid_feature=current_grid_feature.reshape(1,-1)
                     current_grid_word = kmeans_codebook.predict(current_grid_feature)
                     grid_word.append(current_grid_word[0])
                     # Map the gridded daisy feature to a word
                     
         
                 glovetable.append([(current_category,current_filename,np.int32(bovw_id),np.int32(current_frame_word),np.array(grid_word))])
    categorized_frames_dict[cat]=categorized_frames
    
    
gridded_words_overall=[]  # All
bovw_id=-1
categorized_frames_dict={}
categorized_frames_id=[]
num_frames_overall=0
frame_id=0
for cat in test_dict:
    categorized_frames=[]
    
    for interval_id in xrange(len(test_dict[cat])):
        current_interval=test_dict[cat][interval_id]
        file_ind_before=current_interval[0]
        file_ind_after=current_interval[1]
            
        num_frames_overall=num_frames_overall+abs(current_interval[1]-current_interval[0])+1
        
        current_bovw_indices= range(file_ind_before,file_ind_after+2,bovw_size)

        for bovw_ind_before, bovw_ind_after in zip(current_bovw_indices, current_bovw_indices[1:]):
             bovw_id=bovw_id+1
             glovetable_test.flush()

             current_contained_frames=range(bovw_ind_before,bovw_ind_after)
            
        
             num_frames_overall=num_frames_overall+len(current_contained_frames)
         
            # take the middle frame of the bag of visual words as its examplar
             middle_frame=current_contained_frames[len(current_contained_frames)//2]
         
             current_category=frametable[middle_frame]['category']
             current_filename=frametable[middle_frame]['filename']
             

             for j in current_contained_frames:
                 categorized_frames.append(j)
                 frame_id=frame_id+1
                 current_frame_feature=frametable[j]['rawfeature']
                 current_frame_feature=current_frame_feature.reshape(1,-1)
                 current_frame_word = kmeans_codebook.predict(current_frame_feature)[0]
                 
                 
                 current_gridded_frame_feature=frametable[j]['griddedfeature']
                 grid_word=[]
                 for ind in xrange(num_rows):
                     current_grid_feature=current_gridded_frame_feature[ind,:]
                     current_grid_feature=current_grid_feature.reshape(1,-1)
                     current_grid_word = kmeans_codebook.predict(current_grid_feature)
                     grid_word.append(current_grid_word[0])
                     # Map the gridded daisy feature to a word
                     
         
                 glovetable_test.append([(current_category,current_filename,np.int32(bovw_id),np.int32(current_frame_word),np.array(grid_word))])
                 
    categorized_frames_dict[cat]=categorized_frames    

print("Start visual words embedding ..")  
new_gridword_representation,grid_dictionary=embedding_func(glovetable[:]['grid_word'],embedding_size)
new_gridword_representation_test,grid_dictionary_test=embedding_func(glovetable_test[:]['grid_word'],embedding_size)


max_bovw=np.max(glovetable[:]['grid_word']  )
max_bovw_test=np.max(glovetable_test[:]['grid_word']  )

bovw_bins=(max_bovw+1)
bovw_bins_test=(max_bovw_test+1)

hist_bovw_per_cat=np.zeros((num_cats,bovw_bins))
hist_bovw_per_cat_test=np.zeros((num_cats,bovw_bins_test))

hist_glove_per_cat_test=np.zeros((num_cats,embedding_size))
hist_glove_per_cat=np.zeros((num_cats,embedding_size))




  
print("Construct the profile for each category ..")  





# Construct BovW histogram  


i=0
cat_list=[]
for cat in train_dict:
    cat_list.append(cat)
    cat_frames=np.where(glovetable[:]['category']==cat)[0]
    words_within_bag=np.ravel(glovetable[cat_frames]['grid_word'])
        
    sum_embedded_gridded=[]
    for j in words_within_bag:
        sum_embedded_gridded.append(new_gridword_representation[grid_dictionary[j]])

    embedded_gridded=np.mean(np.array(sum_embedded_gridded),axis=0)    
    


    current_hist_bovw=np.histogram(words_within_bag,bins=bovw_bins,density=True)[0]
 
    
    current_hist_bovw_list=current_hist_bovw.tolist()
    hist_bovw_per_cat[i,:]=current_hist_bovw
    hist_glove_per_cat[i,:]=embedded_gridded
    i=i+1
     

i=0
for cat in test_dict:
    cat_frames=np.where(glovetable_test[:]['category']==cat)[0]
    words_within_bag=np.ravel(glovetable_test[cat_frames]['grid_word'])
        
    sum_embedded_gridded=[]
    for j in words_within_bag:
        sum_embedded_gridded.append(new_gridword_representation_test[grid_dictionary_test[j]])

    embedded_gridded=np.mean(np.array(sum_embedded_gridded),axis=0)    
    


    current_hist_bovw=np.histogram(words_within_bag,bins=bovw_bins_test,density=True)[0]
 
    
    current_hist_bovw_list=current_hist_bovw.tolist()
    hist_bovw_per_cat_test[i,:]=current_hist_bovw
    hist_glove_per_cat_test[i,:]=embedded_gridded
    i=i+1

  
  
  
print(cat_list)

unique_filenames,file_indices=np.unique(glovetable_test[:]['filename'],return_index=True)
top1_glove=[]
top2_glove=[]
top3_glove=[]

top1_bovw=[]
top2_bovw=[]
top3_bovw=[]

tested_videos=[]

for current_test_video in  unique_filenames:
    tested_videos.append(current_test_video)
    current_test_video_frames=np.ravel(np.where(glovetable_test[:]['filename']==current_test_video)[0])
    start_frame=min(current_test_video_frames)
    end_frame=max(current_test_video_frames)
    vide_frame_id=range(start_frame,end_frame)
    video_grid_words=np.ravel(glovetable_test[vide_frame_id]['grid_word'])
    middle_frame=int(0.5*(start_frame+end_frame))
    bovw_sim=[] 
    glove_sim=[]

    for i in xrange(hist_bovw_per_cat_test.shape[0]):
        
        
        hist_video_words=np.histogram(video_grid_words,bins=bovw_bins,density=True)[0]
    
        cat_score=hist_intersection(hist_bovw_per_cat[i,:],hist_video_words)
        bovw_sim.append(cat_score)
        
    for i in xrange(hist_bovw_per_cat_test.shape[0]):

        sum_embedded_gridded=[]
        for j in video_grid_words:
            try:
                sum_embedded_gridded.append(new_gridword_representation[grid_dictionary[j]])
            except:
                pass
        embedded_gridded=np.mean(np.array(sum_embedded_gridded),axis=0)    
        a=hist_glove_per_cat[i,:]
        b=embedded_gridded
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        glove_sim.append(cos_sim)
    cat_array=np.asarray(cat_list)
    bovw_sim=np.asarray(bovw_sim)
    glove_sim=np.asarray(glove_sim)
    bovw_sim_ranked=bovw_sim[bovw_sim.argsort()[::-1]] 
    bovw_category_predicted=cat_array[bovw_sim.argsort()[::-1]]
    glove_sim_ranked=glove_sim[glove_sim.argsort()[::-1]] 
    glove_category_predicted=cat_array[glove_sim.argsort()[::-1]]
    true_cat=glovetable_test[middle_frame]['category']
    if true_cat in glove_category_predicted[0]:
        top1_glove.append(current_test_video)
    if true_cat in glove_category_predicted[0:1]:
        top2_glove.append(current_test_video)
    if true_cat in glove_category_predicted[0:2]:
        top3_glove.append(current_test_video)

    if true_cat in bovw_category_predicted[0]:
        top1_bovw.append(current_test_video)
    if true_cat in bovw_category_predicted[0:1]:
        top2_bovw.append(current_test_video)
    if true_cat in bovw_category_predicted[0:2]:
        top3_bovw.append(current_test_video)
print("Bovw top-1 accuracy")
print(len(top1_bovw)/len(tested_videos))

print("Glove top-1 accuracy")
print(len(top1_glove)/len(tested_videos))

    


 


    
 




# Histogram intersection for classification of bovws




# Histogram intersection for classification of gloves


    

 
























 
unique_bovws,bovw_indices=np.unique(glovetable[:]['bovw_id'],return_index=True)
num_bovws=len(unique_bovws)
unique_bovws=unique_bovws[bovw_indices.argsort()] 
bovw_indices=np.sort(bovw_indices)  



class bovwfeature_hdf(tb.IsDescription):
    category        = tb.StringCol(10,pos=1) 
    filename        = tb.StringCol(200,pos=2) 
    current_bovw       = tb.Int32Col(pos=3) 
    embedded_holistic            = tb.Float32Col(embedding_size,pos=4) 
    embedded_gridded           = tb.Float32Col(embedding_size,pos=5) 

                      


bovwh = tb.open_file(bovwfeatures, mode='w')
bovwtable = bovwh.create_table(bovwh.root, 'table', bovwfeature_hdf,"A table") 




bovw_indices=np.append(bovw_indices,len(glovetable)-1)
bovw_indices=list(OrderedDict.fromkeys(bovw_indices))

for bovw_ind_before, bocw_ind_after in zip(bovw_indices, bovw_indices[1:]):
    current_bovw=glovetable[bovw_ind_before]['bovw_id']
    filename=glovetable[bovw_ind_before]['filename']
    category=glovetable[bovw_ind_before]['category']
    words_within_bag=np.ravel(glovetable[bovw_ind_before:bocw_ind_after-1]['grid_word'])
    
    whole_word_for_bag=glovetable[bovw_ind_before:bocw_ind_after-1]['word']
    sum_embedded_gridded=[]
    for i in words_within_bag:
        sum_embedded_gridded.append(new_gridword_representation[grid_dictionary[i]])
        
    embedded_gridded=np.mean(np.array(sum_embedded_gridded),axis=0)
    
    
    
    sum_embedded_holistic=[]   
    """
    for i in whole_word_for_bag:
        sum_embedded_holistic.append(new_word_representation[dictionary[i]])
    np.mean(np.array(sum_embedded_holistic),axis=0)
    
    embedded_holisitc=np.mean(np.array(sum_embedded_holistic),axis=0)

    """
 
    bovwtable.append([(category,filename,current_bovw,embedded_gridded,embedded_gridded)])


 




gloveh.close()
glovefileh = tb.open_file(glovefeatures, mode='r')
glovetable = glovefileh.root.table















# dic_keys=old gridded Words
# dic_values= position of the word in dictionary
 
cat_list=[]
sample_ind=0
overall_bovw_ind=[]
X_bovw_code=[]
X_raw_code=[]
X_glove_code=[]
X_sample_timestep=[]
"""
Video file level containing BOVW
"""

print("Video file level containing BOVW")
for i in xrange(num_videos):
     videofile[i].filename=unique_filenames[i]
     current_contained_bovws= [ind for ind in range(bovwtable.nrows) if bovwtable[ind]['filename'] == unique_filenames[i]]
     videofile[i].contained_bovws=current_contained_bovws
     videofile[i].category= bovwtable[current_contained_bovws[len(current_contained_bovws)//2]]['category']
     # Format the training and testing for TFlearn LSTM model
     chunks_bovws_ind=list(chunks(current_contained_bovws,num_LSTMs))
     if len(chunks_bovws_ind[len(chunks_bovws_ind)-1])<num_LSTMs:
         chunks_bovws_ind=chunks_bovws_ind[0:len(chunks_bovws_ind)-1]
     timestep_ind=0
     for current_bovw_chunk_ind in chunks_bovws_ind:
         cat_list.append(dirs.tolist().index(videofile[i].category))
         for timestep in xrange(num_LSTMs):
             overall_bovw_ind.append(current_bovw_chunk_ind[timestep])
             current_glove_words=bovwtable[current_bovw_chunk_ind[timestep]]['embedded_gridded']
             X_glove_code.append(current_glove_words)
             X_sample_timestep.append((sample_ind,timestep))
         sample_ind=sample_ind+1
          
         
 
# Training samples to LSTM (num samples X num timesteps aka LSTMS X feature dim)
 
X_glove=np.zeros((sample_ind,num_LSTMs,len(X_glove_code[0])))

for i in xrange(len(overall_bovw_ind)):
    ind1=X_sample_timestep[i][0]
    ind2=X_sample_timestep[i][1]
    X_glove[ind1,ind2,:]=X_glove_code[i]
    
  
# Split training and testing sets for frames
all_frames_ind_2=range(len(cat_list))
train_ind_2= sample(all_frames_ind_2,int(0.5*len(cat_list)))
test_ind_2=np.delete(all_frames_ind_2,train_ind_2)


nb_classes=len(dirs)
Y = np_utils.to_categorical(np.asarray(cat_list),nb_classes )


  
X_glove_test=X_glove[test_ind_2,:]   
X_glove_train=X_glove[train_ind_2,:]
Y_test=Y[test_ind_2,:]   
Y_train=Y[train_ind_2,:]  
 
 
 
  
print('Evaluate IRNN with Glove...')
model = Sequential()

model.add(LSTM(output_dim=hidden_units,activation='relu',input_shape=X_glove_train.shape[1:]))
 
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(X_glove_train, Y_train, nb_epoch=nb_epochs,verbose=0)

scores = model.evaluate(X_glove_test, Y_test, verbose=0)
#print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])
bovwh.close()
gloveh.close() 
  
