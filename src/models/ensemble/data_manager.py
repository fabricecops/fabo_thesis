import os
from src.dst.outputhandler.pickle import pickle_save_,pickle_load


class data_manager():


    def __init__(self,dict_c):

        self.dict_c = dict_c
        self.data   = self.load_data()


    def load_data(self):
        path       = self.dict_c['path_i']
        list_names = os.listdir(path)
        array_data =[]

        for name in list_names:

            path_best = path+name+'/best/best_error.p'
            data = pickle_load(path_best,None)
            array_data.append(data)

        return data











