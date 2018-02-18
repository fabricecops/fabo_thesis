from src.dst.outputhandler.pickle import pickle_load

import matplotlib.pyplot as plt
import numpy as np
class conf_data():

    def __init__(self, path,epoch):
        self.path        = path
        self.epoch       = epoch
        self.dict_c      = pickle_load(path + 'dict.p',None)



        self.df_true     = None
        self.df_false    = None
        self.AUC_max     = None
        self.x           = None
        self.df          = None

        self._configure_data()


    def conf_pred_feat(self,i,feature_index):


        output = np.zeros(len(self.df.iloc[i]['frames']))
        pred   = np.zeros(len(self.df.iloc[i]['frames']))
        error  = np.zeros(len(self.df.iloc[i]['frames']))

        print(self.dict_c['window'])

        for j in range(len(output)-self.dict_c['window']+1):
            output[j+self.dict_c['window']-1] = self.df.iloc[i]['data_y'][j][-1][feature_index]
            pred[j+self.dict_c['window']-1]   = self.df.iloc[i]['data_y_p'][j][-1][feature_index]
            error[j+self.dict_c['window']-1]  = self.df.iloc[i]['error_f'][j][-1][feature_index]




        return output, pred, error

    def _configure_data(self):
        path_o          = self.path+'output.p'
        path_y          = self.path + 'predictions/epoch_' + str(self.epoch) + '/pred.p'
        path_sts        = self.path + 'predictions/epoch_' + str(self.epoch) + '/stats.p'

        dict            = pickle_load(path_o,None)
        df_o_t          = dict['df_o_t']
        df_o_f          = dict['df_o_f']

        dict            = pickle_load(path_y,None)
        df_y_t          = dict['df_y_t']
        df_y_f          = dict['df_y_f']

        dict            = pickle_load(path_sts,None)
        self.AUC        = dict['AUC_max']
        self.df_true    = df_o_t.join(df_y_t)
        self.df_false   = df_o_f.join(df_y_f)
        self.x          = dict['x']
        self.df         = self.df_true

        self.resolution = self.dict_c['resolution']
        self.height     = self.dict_c['min_h']
