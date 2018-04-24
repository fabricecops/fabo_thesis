import os
from src.dst.outputhandler.pickle import pickle_save_,pickle_load
import pandas as pd
import numpy as np

class data_manager():


    def __init__(self,dict_c):

        self.dict_c = dict_c

        path        = './data/processed/ensemble/df_'+str(self.dict_c['clusters'])



    def main_DM(self):

        if(self.dict_c['random'] == False):
            data = self.load_data(self.df_groups)
        else:
            data = self.load_data(self.df_random)




        return data








    def load_data(self,df):

        dict_ = {

        }



        array_data   = []
        keys         = [ 'df_t_train', 'df_t_val', 'df_t_test', 'df_f_train', 'df_f_val', 'df_f_test']
        for i in range(len(df)):
            path_best = df['path'].iloc[i]+'best/data_best.p'
            data = pickle_load(path_best,None)
            array_data.append(data)


        for key in keys:
            array_dicts = []
            for data_point in range(len(array_data[0][key])):
                dict_DP = {}
                for model_id in range(len(array_data)):
                    if (model_id == 0):
                        data = array_data[model_id][key]['error_v'].iloc[data_point]
                    else:
                        data = np.vstack((data,  array_data[model_id][key]['error_v'].iloc[data_point]))

                data          = data.reshape(-1,self.dict_c['clusters'])
                data_mean     = np.mean(data,axis = 1)
                data_mean_max = np.max(data_mean)

                dict_DP['error_e']     = data
                dict_DP['error_v']     = data_mean
                dict_DP['error_m']     = data_mean_max


                array_dicts.append(dict_DP)
            dict_[key] = pd.DataFrame(array_dicts,columns=['error_e','error_mean','error_m'])


        return dict_










if __name__ == '__main__':
    dict_c = {
                'path_save': './models/ensemble/',
                'mode'     : 'no_cma.p',
                'path_a'   : ['./models/bayes_opt/DEEP2/'],
                'clusters' : 10,
                'KM_n_init': 10,

                'random'   : False,
    }

    MS = data_manager(dict_c).main()












