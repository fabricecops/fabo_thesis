import os
from src.dst.outputhandler.pickle import pickle_save_,pickle_load
import pandas as pd
import numpy as np

class data_manager():


    def __init__(self,dict_c):

        self.dict_c = dict_c

    def configure_data(self,dict_c):
        data = self.load_data(dict_c['path_i'])
        data = self.pick_best(data)


        return data



    def pick_best(self,data):

        max_AUC = 0.
        for model in data:
            if(model['AUC_v']> max_AUC):
                    best_model = model



        return best_model



    def load_data(self,path):
        list_names = os.listdir(path)
        array_data =[]

        for name in list_names:
            try:
                path_best = path+name+'/best/data_best.p'
                data = pickle_load(path_best,None)
                array_data.append(data)
            except Exception as e:
                print(e)
        return array_data



if __name__ == '__main__':
    def return_dict():
        dict_c = {
            'path_i': './models/bayes_opt/DEEP1/',
            'path_o': './models/ensemble/ensemble/',

            'resolution_AUC': 1000,

            ###### CMA_ES    ######
            'CMA_ES': True,
            'verbose_CMA': 1,
            'verbose_CMA_log': 0,
            'evals': 10000,
            'bounds': [-100, 100.],
            'sigma': 0.4222222222222225,
            'progress_ST': 0.3,

            'epoch': 0

        }

        return dict_c

    dict_c = return_dict()
    data_manager(dict_c).configure_data(dict_c)