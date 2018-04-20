
from src.models.LSTM.data_manager_LSTM import data_manager
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

class LSTM_(model, data_manager):

    def __init__(self, dict_c=None, path=None):

        model.__init__(self, dict_c=dict_c,path=path)
        data_manager.__init__(self,dict_c)



        self.dimension        = None
        self.time_dim         = None
        self.batch_size       = None
        self.input_shape      = None
        self.output_shape     = None
        self.output_dim       = None


        self.copy_experiment()



        self._configure_model()
        self.model            = self._create_model()

        self.print_summary()
        self.print_TB_command()

    def fit(self):


        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch       =self.dict_c['train_steps'],
            epochs                =self.dict_c['epochs'],
            validation_data       =validation_generator,
            validation_steps      =self.dict_c['val_steps'],
            verbose               =self.dict_c['verbose']
        )

        count = self.model.count_params()

        return history.history, count
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

        input_tensor = Input(shape=(t, 224, 224, 1))

        conv1 = TimeDistributed(Conv2D(128, kernel_size=(11, 11), padding='same', strides=(4, 4), name='conv1'),
                                input_shape=(t, 224, 224, 1))(input_tensor)
        conv1 = TimeDistributed(BatchNormalization())(conv1)
        conv1 = TimeDistributed(Activation('relu'))(conv1)

        conv2 = TimeDistributed(Conv2D(64, kernel_size=(5, 5), padding='same', strides=(2, 2), name='conv2'))(conv1)
        conv2 = TimeDistributed(BatchNormalization())(conv2)
        conv2 = TimeDistributed(Activation('relu'))(conv2)

        convlstm1 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm1')(conv2)
        convlstm2 = ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm2')(
            convlstm1)
        convlstm3 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm3')(
            convlstm2)

        deconv1 = TimeDistributed(
            Conv2DTranspose(128, kernel_size=(5, 5), padding='same', strides=(2, 2), name='deconv1'))(convlstm3)
        deconv1 = TimeDistributed(BatchNormalization())(deconv1)
        deconv1 = TimeDistributed(Activation('relu'))(deconv1)

        decoded = TimeDistributed(
            Conv2DTranspose(1, kernel_size=(11, 11), padding='same', strides=(4, 4), name='deconv2'))(
            deconv1)

        return Model(inputs=input_tensor, outputs=decoded)




    def _configure_model(self):
        self.sample_size = self.dict_c['batch_size']
        self.time_dim    = self.dict_c['time_dim']
        self.dimension   = self.df_f_train.iloc[0]['data_X'].shape[2]





    def _predict(self, row):

        X = row['data_X']
        y_pred = self.model.predict(X, batch_size=self.dict_c['batch_size'],
                                    verbose=0)
        self.model.reset_states()
        row['data_y_p'] = y_pred

        return row


    def copy_experiment(self):
        src = 'src'
        print(self.path_gen)
        dst = self.path_gen+'/src'

        shutil.copytree(src,dst)

