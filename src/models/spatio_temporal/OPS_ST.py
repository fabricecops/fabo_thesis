from src.dst.outputhandler.pickle import pickle_save
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from src.dst.outputhandler.pickle import pickle_save_,pickle_load
from src.dst.plots.plots import plotting_tool


class OPS_LSTM(plotting_tool):

    def __init__(self,dict_c):
        self.dict_c = dict_c
        plotting_tool.__init__(self,dict_c)

    def main(self,dict_data):

        dict_data2           = self._get_data_no_cma(dict_data)


        return dict_data2

    def _get_data_no_cma(self,dict_data):
        i          = dict_data['epoch']
        df_t_train = dict_data['df_t_train']
        df_t_val   = dict_data['df_t_val']
        df_t_test  = dict_data['df_t_test']

        df_f_train = dict_data['df_f_train']
        df_f_val   = dict_data['df_f_val']
        df_f_test  = dict_data['df_f_test']

        dimension = df_t_train.iloc[0]['data_X'].shape[2]


        loss_f_tr       = dict_data['loss_f_tr']
        loss_f_v        = dict_data['loss_f_v']


        self.AUC_max = 0
        self.AUC_min = 100


        array_t                        = zip(list(df_t_train['data_y']),list(df_t_train['data_y_p']))
        df_t_train['error_e']          = list(map(self._get_error_cma,array_t))
        df_t_train['error_v']          = list(map(self.get_error_vis,df_t_train['error_e']))
        df_t_train['error_m']          = list(map(self._get_error_m,df_t_train['error_v']))

        array_t                        = zip(list(df_t_val['data_y']),list(df_t_val['data_y_p']))
        df_t_val['error_e']            = list(map(self._get_error_cma,array_t))
        df_t_val['error_v']            = list(map(self.get_error_vis,df_t_val['error_e']))
        df_t_val['error_m']            = list(map(self._get_error_m,df_t_val['error_v']))

        array_t                        = zip(list(df_t_test['data_y']),list(df_t_test['data_y_p']))
        df_t_test['error_e']           = list(map(self._get_error_cma,array_t))
        df_t_test['error_v']           = list(map(self.get_error_vis,df_t_test['error_e']))
        df_t_test['error_m']           = list(map(self._get_error_m,df_t_test['error_v']))

        array_f                        = zip(list(df_f_train['data_y']),list(df_f_train['data_y_p']))
        df_f_train['error_e']          = list(map(self._get_error_cma,array_f))
        df_f_train['error_v']          = list(map(self.get_error_vis,df_f_train['error_e']))
        df_f_train['error_m']          = list(map(self._get_error_m,df_f_train['error_v']))

        array_f                        = zip(list(df_f_val['data_y']),list(df_f_val['data_y_p']))
        df_f_val['error_e']            = list(map(self._get_error_cma,array_f))
        df_f_val['error_v']          = list(map(self.get_error_vis,df_f_val['error_e']))
        df_f_val['error_m']            = list(map(self._get_error_m, df_f_val['error_v']))

        array_f                        = zip(list(df_f_test['data_y']), list(df_f_test['data_y_p']))
        df_f_test['error_e']           = list(map(self._get_error_cma, array_f))
        df_f_test['error_v']          = list(map(self.get_error_vis,df_f_test['error_e']))
        df_f_test['error_m']           = list(map(self._get_error_m, df_f_test['error_v']))

        df_t_train_val= pd.concat([df_t_train,df_t_val])



        AUC, FPR, TPR       = self.get_AUC_score( df_t_train_val['error_m'],  df_f_train['error_m'])
        AUC_v, FPR_v, TPR_v = self.get_AUC_score( df_t_train_val['error_m'],    df_f_val['error_m'])
        AUC_t, FPR_t, TPR_t = self.get_AUC_score( df_t_test['error_m'],   df_f_test['error_m'])

        loss_t_tr, loss_t_v, loss_t_t, loss_f_t,loss_t_v_tr = self._calc_loss(df_t_train,df_f_train,df_t_val,df_f_val,df_t_test,df_f_test)
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

                        'train_f'        : loss_f_tr,
                        'val_f'          : loss_f_v,
                        'test_f'         : loss_f_t,

                        'train_t'        : loss_t_tr,
                        'val_t'          : loss_t_v,
                        'test_t'         : loss_t_t,

                        'train_val_t'    : loss_t_v_tr,

                        'path_o'         : dict_data['path_o'],
                        'epoch'          : dict_data['epoch'],

                        'df_t_train'     : df_t_train,
                        'df_t_val'       : df_t_val,
                        'df_t_test'      : df_t_test,
                        'df_t_val_train' : df_t_train_val,

                        'df_f_train'     : df_f_train,
                        'df_f_val'       : df_f_val,
                        'df_f_test'      : df_f_test,


                        }

        path = dict_data['path_o'] + 'hist.p'
        df = pd.DataFrame([dict_data])[['AUC','AUC_v','AUC_t',
                                        'train_f','train_t','val_f','val_t','test_t','test_f','train_val_t']]




        if('hist.p' not in os.listdir(dict_data['path_o'])):
            pickle_save(path,df)
            df_saved = df
        else:

            df_saved = pickle_load(path,None)
            df_saved = df_saved.append(df,ignore_index=False)


            pickle_save(path,df_saved)

        path_b = dict_data['path_o'] +'best/'

        if (os.path.exists(path_b) == False):
            os.mkdir(path_b)


        if (dict_data['AUC_v'] >= max(list(df_saved['AUC_v']))):
            dict_data['df_t_train'] = dict_data['df_t_train'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]
            dict_data['df_t_val'] = dict_data['df_t_val'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]
            dict_data['df_t_test'] = dict_data['df_t_test'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]
            dict_data['df_t_val_train'] = dict_data['df_t_val_train'][['error_e', 'error_m', 'error_v','location', 'segmentation','frames','label']]

            dict_data['df_f_train'] = dict_data['df_f_train'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]
            dict_data['df_f_val'] = dict_data['df_f_val'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]
            dict_data['df_f_test'] = dict_data['df_f_test'][['error_e', 'error_m','error_v', 'location', 'segmentation','frames','label']]

            pickle_save_(path_b+'data_best.p',dict_data)


        return dict_data

    def get_error_vis(self,row):
        return np.mean(row, axis = 1)

    def _get_error_m(self,row):



        e_f = np.max(row)
        return e_f

    def _get_error_cma(self, row):

        y     = row[0]
        y_p   = row[1]
        e_f = np.mean(np.power((y - y_p),2),axis=1)


        return e_f

    def _calc_loss(self,df_t_train,df_f_train,df_t_val,df_f_val,df_t_test,df_f_test):
        t_tr_y    = np.concatenate(list(df_t_train['data_y']))
        t_tr_yp   = np.concatenate(list(df_t_train['data_y_p']))
        t_e_tr    = np.mean(np.power((t_tr_y-t_tr_yp),2),axis = (1,2))
        loss_t_tr = np.mean(t_e_tr)
        del t_tr_y,t_tr_yp

        t_v_y    = np.concatenate(list(df_t_val['data_y']))
        t_v_yp   = np.concatenate(list(df_t_val['data_y_p']))
        t_e_v    = np.mean(np.power((t_v_y-t_v_yp),2),axis = (1,2))

        t_e_v_tr    = np.hstack((t_e_tr,t_e_v))



        loss_t_v    = np.mean(t_e_v)
        loss_t_v_tr = np.mean(t_e_v_tr)
        del t_v_y,t_v_yp,t_e_v,t_e_tr

        t_t_y    = np.concatenate(list(df_t_test['data_y']))
        t_t_yp   = np.concatenate(list(df_t_test['data_y_p']))
        t_e_t    = np.mean(np.power((t_t_y-t_t_yp),2),axis = (1,2))
        loss_t_t = np.mean(t_e_t)
        del t_t_y,t_t_yp,t_e_t


        f_t_y     = np.concatenate(list(df_f_test['data_y']))
        f_t_yp    = np.concatenate(list(df_f_test['data_y_p']))
        f_e_t     = np.mean(np.power((f_t_y-f_t_yp),2),axis = (1,2))
        loss_f_t  = np.mean(f_e_t)
        del f_t_y,f_t_yp,f_e_t

        return loss_t_tr,loss_t_v,loss_t_t,loss_f_t,loss_t_v_tr


    def save_output(self,dict_data,i):
        if(i==0):



            df_t_train = dict_data['df_t_train'][['frames', 'name', 'label', 'data_X', 'data_y']]
            df_t_val   = dict_data['df_t_val'][['frames', 'name', 'label', 'data_X', 'data_y']]

            df_f_train = dict_data['df_f_train'][['frames', 'name', 'label', 'data_X', 'data_y']]
            df_f_val   = dict_data['df_f_val'][['frames', 'name', 'label', 'data_X', 'data_y']]


            dict_o = {
                'df_t_train': df_t_train,
                'df_t_val'  : df_t_val,

                'df_f_train': df_f_train,
                'df_f_val'  : df_f_val
            }


            path_o = dict_data['path_o'] + 'output.p'
            pickle_save(path_o, dict_o)
