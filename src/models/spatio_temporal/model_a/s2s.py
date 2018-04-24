
from src.models.spatio_temporal.data_manager_ST import data_manager_ST
from src.models.spatio_temporal.conf_ST import return_dict

from src.dst.keras_model.model import model
from tqdm import tqdm
import shutil
from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Sequential
import numpy as np
class spatio_temporal(model, data_manager_ST):

    def __init__(self, dict_c=None, path=None):

        model.__init__(self, dict_c=dict_c,path=path)
        data_manager_ST.__init__(self,dict_c)



        self.dimension        = None
        self.time_dim         = None
        self.batch_size       = None
        self.input_shape      = None
        self.output_shape     = None
        self.output_dim       = None


        self.copy_experiment()



        self.model            = self._create_model()

        self.print_summary()
        self.print_TB_command()

    def fit(self):

        history = self.model.fit_generator(
            self.generator_train(),
            steps_per_epoch       =self.dict_c['steps_per_epoch'],
            epochs                =1.,
            validation_data       =self.generator_val(),
            validation_steps      =self.dict_c['steps_per_epoch_val'],
            verbose               =self.dict_c['verbose']
        )


        return history.history['loss'],history.history['val_loss']

    def predict(self):
        df_t_train     = self.df_t_train.apply(self._predict, axis=1)
        df_t_val       = self.df_t_val.apply(self._predict, axis=1)
        df_t_test      = self.df_t_test.apply(self._predict, axis=1)

        df_f_train     = self.df_f_train.apply(self._predict, axis=1)
        df_f_val       = self.df_f_val.apply(self._predict, axis=1)
        df_f_test      = self.df_f_test.apply(self._predict, axis=1)

        dict_data    = {
                        'path_o'      : self.path_gen,

                        'df_t_train'  : df_t_train,
                        'df_t_val'    : df_t_val,
                        'df_t_test'   : df_t_test,

                        'df_f_train'  : df_f_train,
                        'df_f_val'    : df_f_val,
                        'df_f_test'   : df_f_test,

                        'dict_c'      : self.dict_c,
                        'batch_size'  : self.dict_c['batch_size']
        }

        return dict_data



    def _create_model(self):

        input_shape = (self.dict_c['time_dim'], self.dict_c['width'], self.dict_c['heigth'],1)
        model = Sequential()

        model.add(TimeDistributed(Conv2D(self.dict_c['conv_encoder'][0][0],
                                         kernel_size =  (self.dict_c['conv_encoder'][0][1], self.dict_c['conv_encoder'][0][2]),
                                         padding     =  'same',
                                         strides     =  (self.dict_c['conv_encoder'][0][2], self.dict_c['conv_encoder'][0][2]),
                                         name        = 'conv1'),input_shape = input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))

        for i in range(1,len(self.dict_c['conv_encoder'])):
            model.add(TimeDistributed(Conv2D(self.dict_c['conv_encoder'][i][0],
                                             kernel_size    = ( self.dict_c['conv_encoder'][i][1], self.dict_c['conv_encoder'][i][2]),
                                             padding        = 'same',
                                             strides        = (self.dict_c['conv_encoder'][i][2], self.dict_c['conv_encoder'][i][2]))
                                             ,input_shape  = input_shape))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))



        for i in range(0,len(self.dict_c['conv_LSTM'])):

            model.add(ConvLSTM2D(self.dict_c['conv_LSTM'][i][0],
                                 kernel_size      =(self.dict_c['conv_LSTM'][i][1],self.dict_c['conv_LSTM'][i][1]),
                                 padding          ='same',
                                 return_sequences =True))

        model.add(ConvLSTM2D(self.dict_c['middel_LSTM'][0],
                             kernel_size=(self.dict_c['middel_LSTM'][1], self.dict_c['middel_LSTM'][1]),
                             padding='same',
                             return_sequences=True))

        for i in range(len(self.dict_c['conv_LSTM'])-1,-1,-1):

            model.add(ConvLSTM2D(self.dict_c['conv_LSTM'][i][0],
                                 kernel_size      =(self.dict_c['conv_LSTM'][i][1],self.dict_c['conv_LSTM'][i][1]),
                                 padding          ='same',
                                 return_sequences =True))
        for i in range(len(self.dict_c['conv_encoder'])-1,-1,-1):
            if(i != 0):
                model.add(TimeDistributed(Conv2DTranspose(self.dict_c['conv_encoder'][i][0],
                                                 kernel_size=(
                                                 self.dict_c['conv_encoder'][i][1], self.dict_c['conv_encoder'][i][2]),
                                                 padding='same',
                                                 strides=(
                                                 self.dict_c['conv_encoder'][i][2], self.dict_c['conv_encoder'][i][2]))
                                          , input_shape=input_shape))
            else:
                model.add(TimeDistributed(Conv2DTranspose(1,
                                                 kernel_size=(
                                                 self.dict_c['conv_encoder'][i][1], self.dict_c['conv_encoder'][i][2]),
                                                 padding='same',
                                                 strides=(
                                                 self.dict_c['conv_encoder'][i][2], self.dict_c['conv_encoder'][i][2]))
                                          , input_shape=input_shape))
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))



        optimizer = self.conf_optimizer()
        model.compile(optimizer, 'mse')

        return model


    def _predict(self, row):

        X      = (self._configure_movie(row['images'])-self.mean_)/self.std

        y_pred = self.model.predict(X, batch_size=self.dict_c['batch_size'],
                                    verbose=0)

        y_pred = np.mean(y_pred, axis =(1,2,3,4))


        row['error_e'] = y_pred
        row['error_m'] = np.max(y_pred)
        return row


    def copy_experiment(self):
        src = 'src'
        print(self.path_gen)
        dst = self.path_gen+'/src'

        shutil.copytree(src,dst)

if __name__ == '__main__':

    d   = return_dict()
    spatio_temporal(d).fit()
