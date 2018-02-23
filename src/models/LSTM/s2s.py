from keras.layers.recurrent import LSTM
from keras.layers import Dense,RepeatVector
from keras.models import Sequential

from src.models.LSTM.data_manager import data_manager
from src.dst.keras_model.model import model
from src.dst.outputhandler.pickle import pickle_load
from tqdm import tqdm
import numpy as np

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
        self.X_train          = None
        self.y_train          = None
        self.X_val            = None
        self.y_val            = None
        self._configure_model()

        self.model            = self.create_model()

        self.print_summary()
        self.print_TB_command()

    def create_model(self):

        if (self.dict_c['stateful'] == True):
            model = self._create_model_stateful()

        else:
            model = self._create_model_unstateful()

        return model

    def fit(self):

        if(self.dict_c['stateful'] == True):
            hist = self._fit_stateful()

        else:
            hist = self._fit_unstateful()

        return hist

    def predict(self):
        df_true      = self.df_t.apply(self._predict, axis=1)
        df_false     = self.df_f_train.sample(800).apply(self._predict, axis=1)
        df_f_val     = self.df_f_val.apply(self._predict, axis=1)

        dict_data    = {
                        'path_o'      : self.path_gen,
                        'model'       : self.model,
                        'df_true'     : df_true,
                        'df_false'    : df_false,
                        'df_false_val': df_f_val,
                        'dict_c'      : self.dict_c,
                        'batch_size'  : self.dict_c['batch_size']
        }

        return dict_data,self.model

    def _create_model_unstateful(self):
        model = Sequential()

        hidden1 = 300
        hidden2 = 350
        hidden3 = 400

        model.add(LSTM(hidden1,
               input_shape       = self.input_shape,
               return_sequences  = True,
               stateful          = False
                   ))


        model.add(LSTM(hidden2,
               return_sequences  = False,
               stateful          = False
                   ))

        model.add(Dense(hidden3))


        model.add(RepeatVector(self.input_shape[0]))



        model.add(LSTM(hidden2,
                    return_sequences   = True,
                    stateful           = False))


        model.add(LSTM(self.dimension,
                    return_sequences   = True,
                    stateful           = False))

        # model.add(TimeDistributed(Dense(self.output_dim)))

        optimizer = self.conf_optimizer()
        model.compile(optimizer, 'mse')

        return model

    def _create_model_stateful(self):

        hidden1 = 300
        hidden2 = 350
        hidden3 = 400
        model = Sequential()
        ##Encoder
        model.add(LSTM(hidden1,
               batch_input_shape = self.input_shape,
               return_sequences  = True,
               stateful          = True
                   ))


        input_l2 = (1,self.input_shape[1],hidden1)

        model.add(LSTM(hidden2,
               batch_input_shape = input_l2,
               return_sequences  = False,
               stateful          = True
                   ))

        model.add(Dense(hidden3))


        model.add(RepeatVector(self.input_shape[1]))



        input_d1 = (1,self.input_shape[1],hidden3)

        model.add(LSTM(hidden2,
                    batch_input_shape  = input_d1,
                    return_sequences   = True,
                    stateful           = True))

        input_d2 = (1,self.input_shape[1],hidden2)

        model.add(LSTM(self.dimension,
                    batch_input_shape  = input_d2,
                    return_sequences   = True,
                    stateful           = True))

        optimizer  = self.conf_optimizer()
        model.compile(optimizer, 'mse')

        return model





    def _fit_unstateful(self):

        callbacks = self.create_callbacks()

        hist = self.model.fit(self.X_train, self.y_train,
                              batch_size       = self.dict_c['batch_size'],
                              epochs           = 1,
                              verbose          = self.dict_c['verbose'],
                              callbacks        = callbacks,
                              validation_data  = (self.X_val,self.y_val),
                              )

        return hist.history['loss']

    def _fit_stateful(self):


        loss   = []
        boolean= True

        for i in tqdm(range(len(self.df_f_train))):


            X   = self.df_f_train.iloc[i]['data_X']
            y   = self.df_f_train.iloc[i]['data_y']


            hist    = self.model.fit(X,y,
                           batch_size       = 1,
                           epochs           = 1,
                           verbose          = 0,
                           shuffle          = False
                           )
            loss.extend(hist.history['loss'])
            self.model.reset_states()

            if(i%20 == 0):
                print(np.mean(loss))


        return loss

    def _configure_model(self):
        self.batch_size = self.dict_c['batch_size']
        self.time_dim   = self.dict_c['time_dim']
        self.dimension  = self.df_f_train.iloc[0]['data_X'].shape[2]

        if(self.dict_c['stateful']==True):
            self.input_shape = (1,self.time_dim,self.dimension)

            if(self.dict_c['pred_seq'] == True):
                     self.output_shape = (1,self.time_dim,self.dimension)
            else:
                     self.output_shape = (1, self.dimension)


        else:
            self.input_shape  = (self.time_dim,self.dimension)

            if (self.dict_c['pred_seq'] == True):
                self.output_shape = (self.batch_size, self.time_dim, self.dimension)
            else:
                self.output_shape = (self.batch_size, self.dimension)

        if(self.dict_c['stateful'] == False):
            _, path_tr, path_v = self._return_path_dict_data(self.dict_c)
            self.X_train,self.y_train  = self.main_data_conf_stateless('train')
            self.X_val,self.y_val      = self.main_data_conf_stateless('val')




    def _predict(self, row):

        X = row['data_X']
        y_pred = self.model.predict(X, batch_size=self.dict_c['batch_size'],
                                    verbose=0)
        self.model.reset_states()
        row['data_y_p'] = y_pred

        return row
