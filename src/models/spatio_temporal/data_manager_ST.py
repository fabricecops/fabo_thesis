from src.dst.outputhandler.pickle import pickle_load
from src.dst.helper.apply_mp import *
import pandas as pd
from sklearn.utils import shuffle
import os

import src.data.dimensionality_reduction.HCF.preprocessing as pp
from src.models.spatio_temporal.conf_ST import return_dict
import cv2
import time


class data_manager_ST():

    def __init__(self,dict_c):

        self.dict_c       = dict_c
        self.len_df       = None
        self.count_t      = None

        self.min_final  = 1000.
        self.max_final  = 0.
        self.Series     = pickle_load('./data/processed/ST/df_img.p',self.main_data_conf)

        self.mean_      = self.Series['mean']
        self.std        = self.Series['std']

        self.df_f_val   = None
        self.df_f_test  = None
        self.df_f_train = None

        self.df_t_train = None
        self.df_t_val   = None
        self.df_t_test  = None

        self.configure_shuffle()


        print(len(self.df_f_train),len(self.df_f_val),len(self.df_f_test))
        print(len(self.df_t_train),len(self.df_t_val),len(self.df_t_test))

    def generator_train(self):
        self.df_f_train = shuffle(self.df_f_train)
        len_df_f        = len(self.df_f_train)


        batch_size      = len_df_f//self.dict_c['steps_per_epoch']

        for i in range(self.dict_c['steps_per_epoch']):
            array = []
            for j in range(batch_size):

                imgs_TS = (self._configure_movie(self.df_f_train['images'].iloc[i*batch_size+j])-self.mean_)/self.std

                array.append(imgs_TS)

            data = np.vstack(array)
            data = shuffle(data)[:self.dict_c['max_batch_size']]

            yield (data, data)

    def generator_val(self):
        self.df_f_val = shuffle(self.df_f_val)
        len_df_f = len(self.df_f_val)
        batch_size = len_df_f // self.dict_c['steps_per_epoch_val']

        for i in range(self.dict_c['steps_per_epoch_val']):
            array = []
            for j in range(batch_size):
                imgs_TS = (self._configure_movie(self.df_f_val['images'].iloc[i * batch_size + j])-self.mean_)/self.std
                array.append(imgs_TS)

            data = np.vstack(array)
            yield (data, data)

    def main_data_conf(self,*args):
        self.path       = './data/raw/configured_raw/'
        self.list_names = os.listdir(self.path)


        GD = pp.get_df(self.path)
        path_o = './data/processed/df/df_1.p'

        df                       = pickle_load(path_o, GD.get_df_data, *self.list_names)
        df['images']             = list(map(self.add_frames_df,df['frames']))

        tmp                      = np.vstack(df['images'])


        df_f = df[df['label'] == False]
        df_t = df[df['label'] == True]
        dict_ = {
                 'df_f' :    df_f,
                 'df_t'    : df_t,
                 'max'  : np.max(tmp),
                 'min'  : np.max(tmp),
                 'mean' : np.mean(tmp),
                 'std'  : np.std(tmp)

        }


        Series = pd.Series(dict_)

        return Series

    def add_frames_df(self,row):

        path_ = './data/interim/BGS/bit_8/'

        imgs = np.zeros((len(row),self.dict_c['width'],self.dict_c['heigth'],1))

        min_final = 100
        max_final = 0
        for i, frame in enumerate(row):
            path_image = path_ + frame
            img        = cv2.imread(path_image)

            img         = cv2.resize(img, (self.dict_c['heigth'],self.dict_c['width']))
            img         = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(self.dict_c['width'],self.dict_c['heigth'],1)

            # cv2.imshow('frame', img)
            # time.sleep(0.05)
            # if (np.nan in img):
            #     print('lol')
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     break

            imgs[i]    = img

            df_max = np.max(img)
            df_min = np.min(img)

            if(df_max > max_final):
                self.max_final = df_max
            if(df_min < min_final):
                self.min_final = df_min



        return imgs

    def _configure_movie(self,movie):

        time_dim  = self.dict_c['time_dim']
        window    = self.dict_c['window']

        width     = movie.shape[1]
        heigth    = movie.shape[2]


        samples = len(movie) - window + 1

        if (samples > 0):

            X = np.zeros((samples, time_dim, width, heigth,1))

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

    def return_df(self):
        return self.df_f,self.df_t

    def return_split(self):


        return self.df_f_train,self.df_t_train,self.df_f_val,self.df_t_val,self.df_f_test,self.df_t_test

    def configure_shuffle(self):


        self.df_f = self.Series['df_f']
        self.df_t = self.Series['df_t']

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
    d = return_dict()
    DM = data_manager_ST(d)


    print(next(DM.generator_val())[0].shape)
    print(next(DM.generator_val())[0].shape)
    print(next(DM.generator_val())[0].shape)
    print(next(DM.generator_val())[0].shape)

