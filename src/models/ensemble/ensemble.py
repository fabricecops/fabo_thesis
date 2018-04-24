from src.models.ensemble.data_manager import data_manager

from src.models.ensemble.CMA_ES import CMA_ES

import os
from src.dst.outputhandler.pickle import pickle_load
from src.models.ensemble.model_selection import model_selection
import numpy as np
import pandas as pd
class ensemble(data_manager):

    def __init__(self,dict_c):
        self.dict_c = dict_c
        self._configure_dir(self.dict_c['path_save'])



        data_manager.__init__(self,dict_c)

        path        = './data/processed/df_ensemble/df_'+str(self.dict_c['clusters'])+'.p'
        self.df     = pickle_load(path,model_selection(dict_c).main)


    def main(self):



        array_df = self.configure_data()
        for i in range(len(array_df)):
            pass






    def _configure_dir(self,path):
        path = path+'/best'
        string_a = path.split('/')
        path = ''

        for string in string_a:
            if string != '':
                path += string+'/'

                if (os.path.exists(path) == False):
                    os.mkdir(path)



    def run_CMA_ES(self,dict_c,data):
        cma  = CMA_ES(dict_c)
        data = cma.main(data)

        return data


    def configure_data(self):
        dict_ = {}
        for clusters, df in self.df.groupby('clusters'):
            dict_[clusters] = len(df)

        array = []
        for i in range(self.dict_c['iterations']):
            dict_comb = {}
            for key in dict_.keys():
                int_rand = np.random.randint(0, dict_[key])
                dict_comb[key] = int_rand

            array.append(dict_comb)

        dict_df = {}
        for cluster, df in self.df.groupby('clusters'):
            dict_df[cluster] = df

        array_df = []
        for combination in array:
            tmp = pd.DataFrame()

            for key in combination.keys():
                index = combination[key]
                tmp = tmp.append(dict_df[key].iloc[index])




        return array_df

if __name__ == '__main__':
    def return_dict():

        dict_c = {
            'path_save'     : './models/ensemble/test/',
            'path_a'        : ['./models/bayes_opt/DEEP2/'],

            'resolution_AUC': 1000,
            'mode'          : 'no_cma.p',
            'clusters'      : 2,
            'KM_n_init'     : 10,
            'threshold'     : 0.6,


            #### fit ############
            'iterations'    : 10,


            ###### CMA_ES    ######
            'CMA_ES'          : True,
            'verbose_CMA'     : 1,
            'verbose_CMA_log' : 0,
            'evals'           : 21*1,
            'bounds'          : [-100., 100.],
            'sigma'           : 0.4222222222222225,
            'progress_ST'     : 0.3,
            'popsize'         : 21,

            'epoch': 0

        }
        return dict_c

    dict_c = return_dict()


    dict_c['path_a']    = ['./models/bayes_opt/DEEP2/']
    dict_c['path_save'] = './models/ensemble/test/df_'+str(dict_c['clusters'])+'.p'
    ensemble(dict_c).main()


