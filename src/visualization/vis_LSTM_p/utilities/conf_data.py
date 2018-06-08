from src.dst.outputhandler.pickle import pickle_load
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class conf_data():

    def __init__(self, dict_c):
        self.path           = dict_c['path']
        self.dict_c         = pickle_load(dict_c['path_dict'],None)
        self.dict_c['mode'] = dict_c['mode']
        self.dict_c['plot_mode'] = dict_c['plot_mode']



        self.AUC_max     = None
        self.x           = None
        self.df          = None

        self._configure_data()


    def conf_pred_feat(self,i,feature_index):


        output = np.zeros(len(self.df.iloc[i]['frames']))
        pred   = np.zeros(len(self.df.iloc[i]['frames']))
        error  = np.zeros(len(self.df.iloc[i]['frames']))


        for j in range(len(output)-self.dict_c['window']+1):
            output[j+self.dict_c['window']-1] = self.df.iloc[i]['data_y'][j][-1][feature_index]
            pred[j+self.dict_c['window']-1]   = self.df.iloc[i]['data_y_p'][j][-1][feature_index]


            error[j+self.dict_c['window']-1]  = np.square(output[j+self.dict_c['window']-1]-pred[j+self.dict_c['window']-1])




        return output, pred, error

    def _configure_data(self):
        # path_o          = self.path+'output.p'
        # path_y          = self.path + '/pred.p'
        # path_csv        = self.path + 'hist.csv'
        #
        # dict            = pickle_load(path_o,None)
        # df_o_t          = dict['df_o_t']
        # df_o_f          = dict['df_o_f']
        #
        # dict            = pickle_load(path_y,None)
        # df_y_t          = dict['df_y_t']
        # df_y_f          = dict['df_y_f']
        #
        # df_stats        = pd.read_csv(path_csv)
        #
        #
        #
        #
        # self.AUC        = max(np.array(df_stats['AUC_v']))
        # self.df_true    = df_o_t.join(df_y_t).sort_values(by = 'error_tm')
        #
        #
        #
        # self.df_false   = df_o_f.join(df_y_f, how = 'inner').sort_values(by = 'error_tm')
        #
        #
        # self.x          = np.zeros(100)
        # self.df         = self.df_true
        #
        # self.resolution = self.dict_c['resolution']
        # self.height     =  np.arange(self.dict_c['min_h'], self.dict_c['max_h'], self.dict_c['resolution'])[-1]
        #
        # array_t = zip(list(df_o_t['data_y']), list(df_y_t['data_y_p']))
        # df_o_t['error_m'] = list(map(self._get_error_m, array_t))

        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXx')
        # print(np.mean(list(map(np.mean,df_o_t['error_m']))))
        #
        #
        # print(len(self.df_false))
        # print(len(self.df_true))
        #
        # print(np.array(self.df_false['error_tm']))


        if(self.dict_c['plot_mode'] == 'error'):
            path = self.path + 'best/data_best.p'
            data = pickle_load(path, None)['mode']

        else:
            path = self.path + 'best/df_val.p'
            data = pickle_load(path,None)



        data = data
        self.df = data




    def _get_error_m(self, row):
        # y   = row['data_y']
        # y_p = row['data_y_p']
        y     = row[0]
        y_p   = row[1]

        e_f = np.mean(np.square(np.power((y - y_p),2)),axis=1)
        # row['error_m'] = np.mean(e_f,axis = 1)

        return e_f

