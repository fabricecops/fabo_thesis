import os
from src.dst.outputhandler.pickle import pickle_save_,pickle_load
import pandas as pd
import numpy as np
from src.dst.metrics.AUC import AUC
import matplotlib.pyplot as plt
class data_manager(AUC):


    def __init__(self,dict_c):

        self.dict_c = dict_c
        AUC.__init__(self,dict_c)


    ###### model selection
    def configure_data(self):

        array = []
        for path in self.dict_c['path_a']:

            list_names = os.listdir(path)[:20]

            for name in list_names:
                try:
                    path_best = path + name + '/best/data_best.p'
                    data = pickle_load(path_best, None)

                    error_ = list(data['df_f_train']['error_m'])
                    error_.extend(list(data['df_t_train']['error_m']))
                    error_ = np.array(error_)
                    dict_ = {
                        'error_m': error_,
                        'path': path + name + '/',
                        'AUC_v': data['AUC_v']
                    }
                    array.append(dict_)
                except Exception as e:
                    pass
        df = pd.DataFrame(array, columns=['error_m', 'path', 'AUC_v'])
        return df

    def _configure_dir(self,path):
        path = path
        string_a = path.split('/')
        path = ''

        for string in string_a:
            if string != '':
                path += string+'/'

                if (os.path.exists(path) == False):
                    os.mkdir(path)



    ###### ensemble data collection #######
    def load_data(self,df,nr):

        dict_ = {

        }



        array_data   = []
        keys         = [ 'df_t_train', 'df_t_val', 'df_t_test', 'df_f_train', 'df_f_val', 'df_f_test']
        print('x'*50)
        print(len(df))
        print('x'*50)
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

                data          = data.reshape(-1,nr)


                dict_DP['error_e']     = data



                array_dicts.append(dict_DP)
            dict_[key] = pd.DataFrame(array_dicts,columns=['error_e','location', 'segmentation','frames','label'])


        return dict_

    def configure_data_ensemble(self):
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

    ##### outputhandeler          #######
    def get_data_dict(self,dict_,epoch):
        df_t_train = dict_['df_t_train']
        df_t_val = dict_['df_t_val']
        df_t_test = dict_['df_t_test']

        df_f_train = dict_['df_f_train']
        df_f_val = dict_['df_f_val']
        df_f_test = dict_['df_f_test']

        AUC, FPR, TPR       = self.get_AUC_score( df_t_train['error_m'],  df_f_train['error_m'])
        AUC_v, FPR_v, TPR_v = self.get_AUC_score( df_t_val['error_m'],    df_f_val['error_m'])
        AUC_t, FPR_t, TPR_t = self.get_AUC_score( df_t_test['error_m'],   df_f_test['error_m'])


        dict_data   =  {
                        'AUC'            : AUC,
                        'FPR'            : FPR,
                        'TPR'            : TPR,

                        'AUC_v'          : AUC_v,
                        'TPR_v'          : TPR_v,
                        'FPR_v'          : FPR_v,


                        'AUC_t'          : AUC_t,
                        'TPR_t'          : TPR_t,
                        'FPR_t'          : FPR_t,

                        'df_t_train'     : df_t_train,
                        'df_t_val'       : df_t_val,
                        'df_t_test'      : df_t_test,

                        'df_f_train'     : df_f_train,
                        'df_f_val'       : df_f_val,
                        'df_f_test'      : df_f_test,

                        'epoch'          : epoch


                        }

        path = self.dict_c['path_save'] + 'hist.p'
        df = pd.DataFrame([dict_data])[['AUC','AUC_v','AUC_t']]


        if('hist.p' not in os.listdir( self.dict_c['path_save'])):
            pickle_save_(path,df)
            df_saved = df
        else:

            df_saved = pickle_load(path,None)
            df_saved = df_saved.append(df,ignore_index=False)


            pickle_save_(path,df_saved)
            if(epoch % 10 == 0):
                self.plot(df_saved)

        path_b = self.dict_c['path_save'] +'best/'

        if (os.path.exists(path_b) == False):
            os.mkdir(path_b)


        if (dict_data['AUC_v'] >= max(list(df_saved['AUC_v']))):
            dict_data['df_t_train'] = dict_data['df_t_train'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]
            dict_data['df_t_val'] = dict_data['df_t_val'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]
            dict_data['df_t_test'] = dict_data['df_t_test'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]

            dict_data['df_f_train'] = dict_data['df_f_train'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]
            dict_data['df_f_val'] = dict_data['df_f_val'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]
            dict_data['df_f_test'] = dict_data['df_f_test'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]

            pickle_save_(path_b+'data_best.p',dict_data)


        return AUC_v


    def plot(self,df):
        fig = plt.figure(figsize=(16, 4))
        plt.plot(df['AUC'])
        plt.plot(df['AUC_v'])
        plt.plot(df['AUC_t'])
        plt.savefig(self.dict_c['path_save']+'plots.png')





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












