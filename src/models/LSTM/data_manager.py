from src.dst.outputhandler.pickle import pickle_load
from src.dst.helper.apply_mp import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.utils import shuffle
import os

from src.data.dimensionality_reduction.HCF.main_HCR import pipe_line_data



class data_manager(pipe_line_data):

    def __init__(self,dict_c):

        self.dict_c       = dict_c
        pipe_line_data.__init__(self,dict_c)

        self.len_df       = None
        self.count_t      = None
        self.scaler_p     = None
        self.scaler_v     = None

        path,_,_          = self._return_path_dict_data(dict_c)

        self.Series_data  = pickle_load(path,self.main_data_conf, ())

        self.df_f_train   = self.Series_data[self.dict_c['train']]
        self.df_f_val     = self.Series_data[self.dict_c['val']]
        self.df_t         = self.Series_data[self.dict_c['anomaly']]


        print(len(self.df_f_train),len(self.df_f_val),len(self.df_t ))

    def main_data_conf(self,*args):
        path_df,path_sc_p,path_sc_v = self.return_path_pd(self.dict_c)

        self.df           = pickle_load(path_df,self.peak_derivation, ())
        self.df           = self.df[self.df['countFrames']>5]



        self.len_df       = len(self.df)

        self.scaler_p     = pickle_load(path_sc_p,self._train_scaler, self.df['data_p'])
        self.scaler_v     = pickle_load(path_sc_v,self._train_scaler, self.df['data_v'])



        self.df =  apply_by_multiprocessing(self.df, self._configure_data_movie, axis=1, workers=6        )
        self.df =  self.df[self.df['data_X'] != '']
        self.df =  self.df[self.df['data_y'] != '']
        self.count_t = len(self.df)

        self.df =  self.df[['name','label','frames','data_X','data_y']]
        self._print_data()

        df_t = shuffle(self.df[self.df['label'] == True])
        df_f = shuffle(self.df[self.df['label'] == False])

        val_samples    = int(len(df_f)*self.dict_c['val_split'])
        df_f_val  = df_f.iloc[0:val_samples]
        df_f_tr   = df_f.iloc[val_samples:len(df_f)]


        dict_ = {
                 'df_f_tr' : df_f_tr,
                 'df_f_val': df_f_val,
                 'df_t'    : df_t

        }
        Series = pd.Series(dict_)

        return Series

    def main_data_conf_stateless(self,mode):
        mode = mode[0]

        if (mode == 'val'):
            df = self.df_f_val
        else:
            df = self.df_f_train



        X = np.concatenate(np.array(df['data_X']), axis = 0)
        y = np.concatenate(np.array(df['data_y']), axis = 0)




        return X,y

    def _configure_data_movie(self,row):

        data = self._choose_features(row)

        X_m,y_m        = self._configure_movie(data)
        try:
            tmp = X_m.shape
            row['data_X'] = X_m
            row['data_y'] = y_m

        except Exception as e:
            row['data_X'] = ''
            row['data_y'] = ''


        return row

    def _choose_features(self,row):
        bool_ = False
        data  = None
        for mode in self.dict_c['mode_data']:

            if(mode == 'p'):
                if(bool_ == False):
                    data  = self._transform_scaler(row['data_p'],mode)
                    bool_ = True
                else:
                    tmp   = self._transform_scaler(row['data_p'],mode)
                    data  = np.concatenate((data,tmp), axis = 0)


            elif(mode=='v'):
                if (bool_ == False):
                    data  = self._transform_scaler(row['data_v'],mode)
                    bool_ = True
                else:
                    tmp   = self._transform_scaler(row['data_v'],mode)
                    data = np.concatenate((data,tmp), axis=0)

        return data

    def _configure_movie(self,movie):

        time_dim  = self.dict_c['time_dim']
        window    = self.dict_c['window']
        pred_seq  = self.dict_c['pred_seq']
        dimension = movie.shape[1]


        samples = len(movie) - window + 1

        if (samples > 0):
            X = np.zeros((samples, time_dim, dimension))

            if (pred_seq == True):
                y = np.zeros((samples, time_dim, dimension))
            else:
                y = np.zeros((samples, 1, dimension))

            for i in range(1, samples):
                if (i < time_dim):
                    X[i][-i:][:] = movie[:i]
                elif (i >= time_dim):
                    X[i][:][:] = movie[i - time_dim:i]

                if (pred_seq == True):
                    window_i = i + window
                    if (window_i < time_dim):
                        y[i][-window_i:][:] = movie[:window_i]
                    elif (window_i >= time_dim):
                        y[i][:][:] = movie[window_i - time_dim:window_i]

                else:
                    window_i = i + window - 1
                    y[i][0][:] = movie[window_i]




            if(self.dict_c['pred_seq'] == False):
                y = y.reshape(y.shape[0], y.shape[2])

            return X,y
        else:
            return None,None

    def _transform_scaler(self,data,mode):
        if(mode == 'p'):
            data = self.scaler_p.transform(data)
        elif(mode =='v'):
            data = self.scaler_v.transform(data)

        return data

    def _train_scaler(self,Series):
        Series  = Series[0]
        scaler  = MinMaxScaler(feature_range=(0, 1))
        array   = None

        for i,x in enumerate(Series):

            if(i == 0):
                array = Series.iloc[i]

            else:
                array = np.concatenate((array,Series.iloc[i]), axis = 0)

        scaler.fit(array)


        return scaler

    def _print_data(self):

        if (self.dict_c['print_nr_mov'] == True):
            print(
                'Ratio of movies parced : ' + str(self.count_t) + '/' + str(self.len_df) + '. with seq2seq is: ' + str(
                    self.dict_c['pred_seq']) + '. Statefull: ' + str(self.dict_c['stateful']))

    def _return_path_dict_data(self,dict_c):

        df = 'df'

        TH = '_T_' + str(dict_c['threshold'])
        A = '_A_' + str(dict_c['area'])
        C = '_C_' + str(dict_c['nr_contours'])
        R = '_R_' + str(dict_c['resolution'])

        df = df + TH + A + C + R
        path = './data/processed/df/df_r/'

        path_dir = path+df
        if (os.path.exists(path_dir)==False):
            os.mkdir(path_dir)


        string = ''
        for mode in dict_c['mode_data']:
            string += mode

        window   = str(dict_c['window'])

        if(dict_c['pred_seq'] == True):
            pred_seq = 'T'
        else:
            pred_seq = 'F'
        time_dim = str(dict_c['time_dim'])

        val_split = str(dict_c['val_split'])

        path = path_dir  + '/W_'+window + '_T_'+ time_dim + '_PS_' + pred_seq +'_V_'+val_split+ '_M_'+string

        if (os.path.exists(path)==False):
            os.mkdir(path)

        path_df    = path + '/df.p'
        path_ut_tr = path + '/train.p'
        path_ut_va = path + '/val.p'

        return path_df,path_ut_tr,path_ut_va

