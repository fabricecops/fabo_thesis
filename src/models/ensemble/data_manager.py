import os
from src.dst.outputhandler.pickle import pickle_save_,pickle_load
import pandas as pd
import numpy as np
class data_manager():


    def __init__(self,dict_c):

        self.dict_c = dict_c

    def configure_data(self,dict_c):

        data = self.load_in_all_data(dict_c)
        data = self.pick_best(data)
        data = self.convert_data_frames(data)
        data = self.combine_models(data)


        return data



    def combine_models(self,data):

        dict_ = {}
        for key in data['DEEP1'].keys():
            if (key != 'AUC_v'):

                error_D1_a = np.array(data['DEEP1'][key]['error'])
                error_D2_a = np.array(data['DEEP2'][key]['error'])
                error_D3_a = np.array(data['DEEP3'][key]['error'])

                array = []

                for error_D1,error_D2,error_D3 in zip(error_D1_a,error_D2_a,error_D3_a):
                    error_D1 = error_D1.reshape(-1,1)
                    error_D2 = error_D2.reshape(-1,1)
                    error_D3 = error_D3.reshape(-1,1)
                    array.append(np.concatenate([error_D1,error_D2,error_D3],axis = 1))

                df_combined = data['DEEP1'][key]
                df_combined['error'] = array


                dict_[key] = df_combined

        return dict_






    def convert_data_frames(self,data):
        for key in data.keys():
            for key2 in data[key]:
                if(key2 != 'AUC_v'):
                    try:
                        data[key][key2]['error'] = data[key][key2]['error_e']
                    except Exception as e:
                        print(e)
        return data

    def pick_best(self,data):

        dict_ = {}

        for key in data.keys():
            max_AUC = 0.
            for model in data[key]:
                if(model['AUC_v']> max_AUC):
                    best_model = model

            dict_[key] =  best_model


        return dict_





    def load_in_all_data(self, dict_c):
        dict_ = {}

        for key in dict_c['path_ensemble'].keys():
            try:
                dict_[key] = self.load_data(dict_c['path_ensemble'][key])
            except:
                pass

        return dict_


    def load_data(self,path):
        list_names = os.listdir(path)
        array_data =[]

        for name in list_names:

            path_best = path+name+'/best/best_error.p'
            data = pickle_load(path_best,None)
            array_data.append(data)

        return array_data











