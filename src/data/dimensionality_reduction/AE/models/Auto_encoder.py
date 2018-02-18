import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
import os
from tqdm import tqdm
import shutil
import tensorflow as tf
import h5py


from keras.layers import merge, Input,MaxPooling2D,Flatten,UpSampling2D
from keras.models import Sequential, Model
from keras.layers.core import Reshape,RepeatVector,Dense, Dropout, Reshape,Activation
from keras.callbacks import TensorBoard,History
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam

class Conv_AE():

    def __init__(self,dict_p):
        
        ################## Parameters  ##########################
        #########################################################
        #########################################################

        self.model = Sequential()
        self.encoded = Sequential()
        self.decoded = Sequential()
        
        
        self.img_width = dict_p['img_width']
        self.img_height = dict_p['img_height']

        self.train_data_dir = dict_p['train_data_dir']
        self.validation_data_dir = dict_p['validation_data_dir']
        self.tensorboard_dir = dict_p['tensorboard_dir']
        self.model_path = dict_p['model_path']
        
        self.nb_train_samples = dict_p['nb_train_samples']
        self.nb_validation_samples = dict_p['nb_validation_samples']
        self.epochs = dict_p['epochs']
        self.batch_size = dict_p['batch_size']
        
        self.lr = dict_p['lr']
        self.dropout = dict_p['dropout']
        
        self.filters = dict_p['filters']
        self.upsampling = dict_p['upsampling']
        self.output = dict_p['output']
        
        self.optimizer = self.conf_optimizer(dict_p['optimizer'])
        self.loss = dict_p['loss']
        
        
        self.colormode =dict_p['colormode']
        self.rescale = dict_p['rescale']
        self.shear_range = dict_p['shear_range']
        self.zoom_range = dict_p['zoom_range']
        self.horizontal_flip = dict_p['horizontal_flip']
        
        self.tensorboard = TensorBoard(log_dir= self.tensorboard_dir+str(len(os.listdir(self.tensorboard_dir))),
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images = True,
                                        embeddings_freq=0, 
                                        write_grads = False,
                                        embeddings_layer_names=True, 
                                        embeddings_metadata=True)
        self.history = History()
               
        ###################### model ################################
        #############################################################
        input_shape = (self.img_width,self.img_height,1)
        
        for i,filter_ in enumerate(self.filters):     
            
            if(i==0):
                conv_layer =  Conv2D(filter_[0], (filter_[1], filter_[2]), activation='relu', padding='same',input_shape=input_shape)
                max_pool = MaxPooling2D((filter_[3], filter_[3]), padding='same')
            else:
                conv_layer =  Conv2D(filter_[0], (filter_[1], filter_[2]), activation='relu', padding='same')
                max_pool = MaxPooling2D((filter_[3], filter_[3]), padding='same')
            
            self.model.add(conv_layer)
            self.model.add(max_pool)
        
        for up_sample in self.upsampling:
        
            conv_layer = Conv2D(up_sample[0], (up_sample[1], up_sample[2]), activation='relu', padding='same')
            upsample = UpSampling2D((up_sample[3], up_sample[3]))
            self.model.add(conv_layer)
            self.model.add(upsample)

            
        conv_layer = Conv2D(self.output[0], (self.output[1], self.output[2]), activation='sigmoid', padding='same')
        self.model.add(conv_layer)

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        assert(self.model.output_shape==self.model.input_shape)

    ############################## Public Functions ##################
    ##################################################################
             
    def fit_generator(self):
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
                rescale=self.rescale,
                shear_range=self.shear_range,
                zoom_range=self.shear_range,
                horizontal_flip=self.shear_range)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=self.rescale)


        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size= self.batch_size,
            color_mode= self.colormode)


        validation_generator = test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            color_mode= self.colormode)

        history=self.model.fit_generator(
                self.fixed_generator(train_generator),
                steps_per_epoch=self.nb_train_samples//self.batch_size,
                epochs=self.epochs,
                validation_data=self.fixed_generator(validation_generator),
                validation_steps=self.nb_validation_samples//self.batch_size,
                callbacks = [self.tensorboard,self.history]
                )
        return history.history
    
    def evaluate_df(self,df_data):
        return df_data.apply(self.apply_error_eval,axis = 1)

        
    def retrieve_all_features(self,batch):
        dict_features = {}
        for layer_nr,layer in enumerate(self.model.layers):
            dict_features[layer_nr] = self.retrieve_features_layer(batch,layer_nr)

        return dict_features


    def evaluate_picture(self,batch,batch_p):
        evaluations = []
        for i in range(len(batch)):
            X = batch[i].reshape((1,batch[i].shape[0],batch[i].shape[1],batch[i].shape[2]))
            y = batch_p[i].reshape((1,batch_p[i].shape[0],batch_p[i].shape[1],batch_p[i].shape[2]))
            eval_ = self.model.evaluate(X,y, verbose=0)
            evaluations.append(eval_)

        return evaluations       

       
    def give_tensorboard_command(self):
        path = 'Dropbox/thesis/Conv/'+self.tensorboard_dir
        print('tensorboard --logdir='+path)
        
    def model_save(self):
        self.model.save(self.model_path)
  
    def return_model(self):
        return self.model
        
    def model_load(self):
        self.model = load_model(self.model_path)
        return self.model

    def predict_picture(self,picture):
        frame_R= cv2.resize(picture,(self.img_width,self.img_height))*self.rescale
        frame_RR = frame_R.reshape((1,self.img_width,self.img_height,1))       
        prediction = self.model.predict(frame_RR,verbose = 0)
        evaluation = self.model.evaluate(frame_RR,prediction,verbose = 0)
        prediction = prediction.reshape(self.img_width,self.img_height)/self.rescale
        return prediction,evaluation
    
    ###### Private functions ########################################################
    #################################################################################
    
    def apply_error_eval(self,row):
        movie = self.load_in_movie(row)
        prediction = self.model.predict(movie)
        evaluations = self.evaluate_picture(movie,prediction)
        return  evaluations

        
    
    def load_in_movie(self,row):
        movie = []
        for frame in row['frames']:
            if(row['label']==True):
                path = 'data/BGS/pos_images/class1/'+frame
            elif(row['label']==False):
                path = 'data/BGS/neg_images/class1/'+frame
            else:
                path = None
            
            if(path != None):
                frame_ = cv2.imread(path,-1)
                frame_R= cv2.resize(frame_,(self.img_width,self.img_height))*self.rescale
                frame_RR = frame_R.reshape((self.img_width,self.img_height,1))
                movie.append(frame_RR)
                
        return np.array(movie)

        
    def fixed_generator(self,generator):
        for batch in generator:
            yield (batch[0], batch[0])

    def conf_optimizer(self,optimizer):
        if optimizer == 'adam':
            return Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        
    def retrieve_features_layer(self,batch,layer_nr):

        features = []
        get_3rd_layer_output = K.function([self.model.layers[0].input],[self.model.layers[layer_nr].output])

        for sample in batch:
                layer_output = get_3rd_layer_output([sample])[0]
                features.append(layer_output.reshape(layer_output.shape[0],-1))

        shape_layer = layer_output[0].shape[0:]
        return [shape_layer,features]


        
