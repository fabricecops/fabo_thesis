from src.dst.outputhandler.pickle import pickle_load
from src.dst.helper.apply_mp import *
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


        self.df_f_val   = None
        self.df_f_test  = None
        self.df_f_train = None

        self.df_t_train = None
        self.df_t_val   = None
        self.df_t_test  = None

        self.configure_shuffle()


        print(len(self.df_f_train),len(self.df_f_val),len(self.df_f_test))
        print(len(self.df_t_train),len(self.df_t_val),len(self.df_t_test))

    def main_data_conf(self,*args):
        path_df,path_sc_p,path_sc_v,_ = self.return_path_pd(self.dict_c)
        self.df           = pickle_load(path_df,self.peak_derivation, ())
        self.len_df       = len(self.df)

        self.df =  apply_by_multiprocessing(self.df, self._configure_data_movie, axis=1, workers=12)
        self.df =  self.df[self.df['data_X'] != '']
        self.df =  self.df[self.df['data_y'] != '']
        self.count_t = len(self.df)
        self.df =  self.df[['name','label','frames','data_X','data_y','segmentation','location']]
        self._print_data()

        df_f     = self.df[self.df['label'] == False]
        df_t     = self.df[self.df['label'] == True]


        dict_ = {
                 'df_f' :    df_f,
                 'df_t'    : df_t

        }
        Series = pd.Series(dict_)

        return Series

    def main_data_conf_stateless(self,mode):

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
        data  = None
        for i,mode in enumerate(self.dict_c['mode_data']):

            if(mode == 'p'):
                if(i == 0):
                    data  = row['data_p']
                else:
                    data  = np.concatenate((data,row['data_p']), axis = 1)


            elif(mode=='v'):
                if(i == 0):
                    data  = row['data_v']
                else:
                    data  = np.concatenate((data,row['data_v']), axis = 1)

            elif(mode == 'PCA'):
                if(i == 0):
                    data  = row['PCA']
                else:
                    data  = np.concatenate((data,row['PCA']), axis = 1)




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
        PCA = '_PCA_'+str(dict_c['PCA_components'])

        df = df + TH + A + C + R +PCA
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

        val_split = str(dict_c['val_split_t'])

        path = path_dir  + '/W_'+window + '_T_'+ time_dim + '_PS_' + pred_seq +'_V_'+val_split+ '_M_'+string

        if (os.path.exists(path)==False):
            os.mkdir(path)

        path_df    = path + '/df.p'
        path_ut_tr = path + '/train.p'
        path_ut_va = path + '/val.p'

        return path_df,path_ut_tr,path_ut_va

    def return_df(self):
        return self.df_f,self.df_t

    def return_split(self):


        return self.df_f_train,self.df_t_train,self.df_f_val,self.df_t_val,self.df_f_test,self.df_t_test

    def configure_shuffle(self):

        path,_,_          = self._return_path_dict_data(self.dict_c)

        self.Series_data  = self.main_data_conf()

        self.df_f = self.Series_data['df_f']
        self.df_t = self.Series_data['df_t']

        self.df_t = shuffle(self.df_t,random_state = self.dict_c['random_state'])
        self.df_f = shuffle(self.df_f,random_state = self.dict_c['random_state'])


        val_samples_f  = int(len(self.df_f) * self.dict_c['val_split_f'])
        test_samples_f = int(len(self.df_f) * self.dict_c['test_split_f'])
        val_samples_t  = int(len(self.df_t) * self.dict_c['val_split_t'])
        test_samples_t = int(len(self.df_t) * self.dict_c['test_split_f'])

        self.df_f_val = self.df_f.iloc[0:val_samples_f]
        self.df_f_test = self.df_f.iloc[val_samples_f:val_samples_f + test_samples_f]
        self.df_f_train = self.df_f.iloc[val_samples_f + test_samples_f:len(self.df_f)]


        if(self.dict_c['shuffle_style'] == 'testing'):

            self.df_f_val = self.df_f.iloc[0:val_samples_f].iloc[0:5]
            self.df_f_test = self.df_f.iloc[val_samples_f:val_samples_f + test_samples_f].iloc[0:5]
            self.df_f_train = self.df_f.iloc[val_samples_f + test_samples_f:len(self.df_f)].iloc[0:5]

            self.df_t_train = self.df_t.iloc[val_samples_t + test_samples_t:len(self.df_t)].iloc[0:5]
            self.df_t_val = self.df_t.iloc[0:val_samples_t].iloc[0:5]
            self.df_t_test = self.df_t.iloc[val_samples_t:val_samples_t + test_samples_t].iloc[0:5]

        elif(self.dict_c['shuffle_style'] == 'random'):
            print('random!!!!')

            self.df_t_train = self.df_t.iloc[val_samples_t + test_samples_t:len(self.df_t)]
            self.df_t_val = self.df_t.iloc[0:val_samples_t]
            self.df_t_test = self.df_t.iloc[val_samples_t:val_samples_t + test_samples_t]

        elif (self.dict_c['shuffle_style'] == 'segmentated'):
            print('segmentated!!!!')



            self.df_t_train = pd.DataFrame()
            self.df_t_val   = pd.DataFrame()
            self.df_t_test  = pd.DataFrame()

            state_var       = True
            for i,group in enumerate(self.df_t.groupby(['segmentation','location'])):


                val_samples_t  = round(len(group[1]) * self.dict_c['val_split_t'])
                test_samples_t = round(len(group[1]) * self.dict_c['test_split_t'])

                if(val_samples_t == 0 and test_samples_t == 0 and group[0][1] == 'bnp_1' ):
                    if(state_var == True):
                        val_samples_t = 1
                        state_var     = False
                    else:
                        test_samples_t = 1
                        state_var      = True



                if(i==0):
                    self.df_t_train = group[1].iloc[val_samples_t + test_samples_t:len(group[1])]
                    self.df_t_val   = group[1].iloc[0:val_samples_t]
                    self.df_t_test  = group[1].iloc[val_samples_t:val_samples_t + test_samples_t]

                else:
                    self.df_t_train = self.df_t_train.append(group[1].iloc[val_samples_t + test_samples_t:len(group[1])],ignore_index=True)
                    self.df_t_val   = self.df_t_val.append(group[1].iloc[0:val_samples_t],ignore_index=True)
                    self.df_t_test  = self.df_t_test.append(group[1].iloc[val_samples_t:val_samples_t + test_samples_t],ignore_index=True)





        self.df_f_train = self.df_f_train.reset_index()
        self.df_f_val   = self.df_f_val.reset_index()
        self.df_f_test  = self.df_f_test.reset_index()

        self.df_t_train = self.df_t_train.reset_index()
        self.df_t_val   = self.df_t_val.reset_index()
        self.df_t_test  = self.df_t_test.reset_index()