from src.dst.outputhandler.pickle import pickle_load
from src.dst.helper.apply_mp import *
import pandas as pd
from sklearn.utils import shuffle
import os

import src.data.dimensionality_reduction.HCF.preprocessing as pp
from src.models.spatio_temporal.conf_ST import return_dict_bounds
import cv2

class data_manager():

    def __init__(self,dict_c):

        self.dict_c       = dict_c
        self.len_df       = None
        self.count_t      = None

        self.Series     = pickle_load('./data/processed/ST/df_img.p',self.main_data_conf)

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
        self.path       = './data/raw/configured_raw/'
        self.list_names = os.listdir(self.path)


        GD = pp.get_df(self.path)
        path_o = './data/processed/df/df_1.p'

        df = pickle_load(path_o, GD.get_df_data, *self.list_names)

        df_f = df[df['label'] == False]
        df_t = df[df['label'] == True]
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

    def _configure_frames(self,row):


        path_ = './data/interim/BGS/bit_8/'

        imgs  = np.zeros((len(row['frames']),240,320))


        for i,frame in enumerate(row['frames']):
            path_image = path_+frame
            imgs[i]    = cv2.imread(path_image,-1)

        imgs_TS = self._configure_movie(imgs)
        print(imgs_TS.shape)

        return row

    def _configure_movie(self,movie):

        time_dim  = self.dict_c['time_dim']
        window    = self.dict_c['window']

        width     = movie.shape(1)
        heigth    = movie.shape(2)


        samples = len(movie) - window + 1

        if (samples > 0):

            X = np.zeros((samples, time_dim, width, heigth))

            for i in range(1, samples):
                if (i < time_dim):
                    X[i][-i:][:] = movie[:i]
                elif (i >= time_dim):
                    X[i][:][:] = movie[i - time_dim:i]



            return X
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

if __name__ =='__main__':
    d,b = return_dict_bounds('DEEP1')
    data_manager(d).main_data_conf()