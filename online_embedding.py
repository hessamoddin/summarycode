from __future__ import print_function
from __future__ import division, print_function, absolute_import
from sklearn.cluster import MiniBatchKMeans


from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from keras.optimizers import RMSprop

from keras.utils import np_utils
from keras.layers import Dropout,Embedding
import logging
import numpy as np
import tables as tb
import math


 

  
        
        
"""       
Parameters
"""
bovw_size=15
train_frac=0.90

batch_size = 1
nb_epochs = 100
learning_rate = 1e-6


N=4


framefeatures="framefeatures8_vgg.h5"
glovefeatures_train="glove_train.h5"
glovefeatures_test="glove_test.h5"

 


"""       
Define functions
"""

def LSTM_Func(hidden_units,vgg_matrix_train,num_cats,learning_rate,y_train,nb_epochs,vgg_matrix_test, y_test):

    model = Sequential()

 
 

    model.add(LSTM(output_dim=hidden_units,activation='relu',input_shape=vgg_matrix_train.shape[1:]))
    model.add(Dense(num_cats))
    model.add(Dropout(0.2))

    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

    model.fit(vgg_matrix_train, y_train, nb_epoch=nb_epochs,verbose=0)

    scores = model.evaluate(vgg_matrix_test, y_test, verbose=0)    
    print('Accuracy:', scores[1])

    ranked_classes_all=np.argsort(-model.predict(vgg_matrix_test),axis=1)

    AP=0
    sum_ind=0
 
    for i in xrange(y_test.shape[0]):
        #predicted_classes=np.argmax(model.predict(vgg_matrix_test),axis=1)
        expected_class=y_test.shape[1]
        predicted_classes=ranked_classes_all[i,:]
        x=(predicted_classes==expected_class)
        r=x*1
        ind=np.where(predicted_classes==expected_class)[0][0]
        sum_ind=sum_ind+ind
        #AP=AP+1./(1+ind)
        AP=AP+sum([sum(r[:z + 1]) / (z + 1.)  for z, y in enumerate(r) if y])
    
    mAP=AP/y_test.shape[0]

    print('Mean Average Precision:',mAP)
    return scores,mAP


def train_test_date(glovetable,num_cat,num_LSTMs):
    unique_filenames,file_indices=np.unique(glovetable[:]['filename'],return_index=True)
    num_videos=len(unique_filenames)
    unique_filenames=unique_filenames[file_indices.argsort()] 
    file_indices=np.sort(file_indices)
    file_indices_appended=np.append(file_indices,glovetable.shape[0])


    # Index for files 
    file_intervals=zip(file_indices_appended,file_indices_appended[1:]-1) 
    y_label=[]
    X=[]

    for interval in file_intervals:
        mid_frame_ind=int(np.mean(interval))
        current_cat_enumerated=unique_categories.index(glovetable[mid_frame_ind]['category'])
        current_words=glovetable[interval[0]:interval[1]]['word']
        current_words_chuncked=map(None, *([iter(current_words)] * num_LSTMs))
        if (interval[1]-interval[0])%num_LSTMs>0:
            current_words_chuncked=current_words_chuncked[:-1]
        for words_chuncked in current_words_chuncked:
            X.append(words_chuncked)
            y_label.append(current_cat_enumerated)
            
    y=np.array(np_utils.to_categorical(y_label,num_cat))
 
    return np.array(X),np.array(y)




def learn_kmeans_codebook(X, codebook_size=1000, seed=None):
    """ Learn a codebook.
    source: https://github.com/KitwareMedical/TubeTK/blob/master/Base/Python/pyfsa/core/fsa.py
    """
    logger = logging.getLogger()
    logger.info("Learning codebook with %d words ..." % codebook_size)
    # Run vector-quantization
                
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=codebook_size, batch_size=30,
                      n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit(X)

                
    return mbk
 

 
   








    
 
framefileh = tb.open_file(framefeatures, mode='r')
frametable=framefileh.root.table 
  
 
num_rows=frametable[0]['griddedfeature'].shape[0]
num_cols=frametable[0]['griddedfeature'].shape[1]




class glovefeature_hdf(tb.IsDescription):
    category        = tb.StringCol(30,pos=1) 
    filename        = tb.StringCol(200,pos=2) 
    bovw_id         = tb.Int32Col(pos=3) 
    word            = tb.Int32Col(pos=4) 
    grid_word            = tb.Int32Col(num_rows,pos=5) 
    frame_feature =tb.Float32Col(1000,pos=6)
    

    

        

 
gloveh_train = tb.open_file(glovefeatures_train, mode='w')
glovetable_train = gloveh_train.create_table(gloveh_train.root, 'table', glovefeature_hdf,"A table") 

gloveh_test = tb.open_file(glovefeatures_test, mode='w')
glovetable_test = gloveh_test.create_table(gloveh_test.root, 'table', glovefeature_hdf,"A table") 
                       



unique_filenames,file_indices=np.unique(frametable[:]['filename'],return_index=True)
num_videos=len(unique_filenames)
unique_filenames=unique_filenames[file_indices.argsort()] 
file_indices=np.sort(file_indices)
file_indices_appended=np.append(file_indices,frametable.shape[0])





unique_categories,cat_indices=np.unique(frametable[:]['category'],return_index=True)
num_cats=len(unique_categories)
unique_categories=unique_categories[cat_indices.argsort()] 
cat_indices=np.sort(cat_indices)
cat_indices_appended=np.append(cat_indices,frametable.shape[0])


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
print("For training")
gridded_words_overall=[]  # All
bovw_id=-1
categorized_frames_dict={}
categorized_frames_id=[]
num_frames_overall=0
frame_id=0
for cat in train_dict:
    print(cat)
    categorized_frames=[]
    
    for interval_id in xrange(len(train_dict[cat])):
        current_interval=train_dict[cat][interval_id]
        file_ind_before=current_interval[0]
        file_ind_after=current_interval[1]
            
        num_frames_overall=num_frames_overall+abs(current_interval[1]-current_interval[0])+1
        
        current_bovw_indices= range(file_ind_before,file_ind_after+2,bovw_size)

        for bovw_ind_before, bovw_ind_after in zip(current_bovw_indices, current_bovw_indices[1:]):
             bovw_id=bovw_id+1
             glovetable_train.flush()

             current_contained_frames=range(bovw_ind_before,bovw_ind_after)
            
        
            # num_frames_overall=num_frames_overall+len(current_contained_frames)
         
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
                     
         
                 glovetable_train.append([(current_category,current_filename,np.int32(bovw_id),np.int32(current_frame_word),np.array(grid_word),current_frame_feature)])
    categorized_frames_dict[cat]=categorized_frames


print("For testing")

gridded_words_overall=[]  # All
categorized_frames_dict={}
categorized_frames_id=[]
frame_id=0
for cat in test_dict:
    print(cat)
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
            
        
            # num_frames_overall=num_frames_overall+len(current_contained_frames)
         
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
                     
         
                 glovetable_test.append([(current_category,current_filename,np.int32(bovw_id),np.int32(current_frame_word),np.array(grid_word),current_frame_feature)])
    categorized_frames_dict[cat]=categorized_frames
 
    
 


unique_categories,cat_indices=np.unique(glovetable_train[:]['category'],return_index=True)
unique_categories=list(unique_categories)

num_cat=len(unique_categories)
unique_filenames,file_indices=np.unique(glovetable_train[:]['filename'],return_index=True)
num_train_videos=len(unique_filenames)
unique_filenames=unique_filenames[file_indices.argsort()] 
file_indices=np.sort(file_indices)
file_indices_appended=np.append(file_indices,glovetable_train.shape[0])




embedding_size=[512]
hidden_units = [256]
num_LSTMs=[15]
parametrized_performance=[]

for size in embedding_size:
    for units in hidden_units:
        for num in num_LSTMs:
            
            X_train,y_train=train_test_date(glovetable_train,num_cat,num)
            X_test,y_test=train_test_date(glovetable_test,num_cat,num)

            model = Sequential()
            model.add(Embedding(size, units, input_shape=X_train.shape[1:]))
            model.add(Dropout(0.1))

            #model.add(LSTM(hidden_units, return_sequences=True))
            model.add(LSTM(units))
            model.add(Dropout(0.3))
            model.add(Dense(num_cat))
            model.add(Activation('softmax'))
            optimizer = RMSprop(lr=0.01)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            #print(model.summary())



            model.fit(X_train, y_train, nb_epoch=50,verbose=0)

            scores = model.evaluate(X_test, y_test, verbose=0)
            current_performance=[size,units,num,scores[1]]
            parametrized_performance.append(current_performance)
            print('--size',size,'--units',units,'--num',num,'=>Accuracy:',scores[1])
            


 
ranked_classes_all=np.argsort(-model.predict(X_test),axis=1)

AP=0
sum_ind=0
 
for i in xrange(len(y_test)):
        #predicted_classes=np.argmax(model.predict(vgg_matrix_test),axis=1)
    expected_class=int(list(y_test[-1]).index(1))
    predicted_classes=ranked_classes_all[i,:]
    x=(predicted_classes==expected_class)
    r=x*1
    ind=np.where(predicted_classes==expected_class)[0][0]
    sum_ind=sum_ind+ind
    #AP=AP+1./(1+ind)
    AP=AP+sum([sum(r[:z + 1]) / (z + 1.)  for z, y in enumerate(r) if y])
    
mAP=AP/len(y_test)
#print('Mean Average Precision:',mAP)
     
 
