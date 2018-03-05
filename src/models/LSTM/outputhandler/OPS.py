from src.dst.outputhandler.pickle import pickle_save
import os
import pandas as pd
import matplotlib.pyplot as plt
import functools
plt.style.use('ggplot')
import numpy as np
from src.dst.metrics.AUC import AUC
from src.dst.outputhandler.pickle import pickle_save_

class OPS_LSTM(AUC):

    def __init__(self,dict_c):
        self.dict_c = dict_c
        AUC.__init__(self,dict_c)

    def main(self,dict_data,i):
        dict_data2           = self.get_data(dict_data,i)
        dict_data2['path_o'] = dict_data['path_o']

        self._save_plots(dict_data2)


    def _save_plots(self,dict_data):

        dir = dict_data['path_o'] +'predictions/'
        str_= 'epoch_'+str(len(os.listdir(dir))-1)+'/'

        path_p = dir+str_


        ### validation curve
        df = pd.read_csv(dict_data['path_o']+'hist.csv')




        fig = plt.figure(figsize=(16, 4))

        ax1 = plt.subplot(121)
        ax1.plot(df['val_t'],    color = 'red',   label = 'val_t')
        ax1.fill_between(range(len(df)),df['val_t']-df['val_std_t'],df['val_t']+df['val_std_t'], color = 'red', alpha = 0.3)
        ax1.plot(df['val_f'],    color = 'green', label = 'val_f')
        ax1.fill_between(range(len(df)),df['val_f']-df['val_std_f'],df['val_t']+df['val_std_f'], color = 'green', alpha = 0.3)
        ax1.plot(df['train_f'],  color = 'blue',  label = 'train_f')
        ax1.fill_between(range(len(df)),df['train_f']-df['train_std'],df['train_f']+df['train_std'], color = 'blue', alpha = 0.3)
        plt.legend()
        plt.title('validation/train curve')
        plt.xlabel('epoch nr')
        plt.ylabel('average loss')

        ax2 = plt.subplot(122)
        plt.plot(df['AUC'], color = 'r', label = 'train')
        plt.plot(df['AUC_v'], color = 'g', label = 'val')
        plt.title('train/val AUC')
        plt.xlabel('epoch nr')
        plt.ylabel('AUC')
        plt.savefig(dict_data['path_o'] + '/val_curve.png')

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)

        ax1.plot(dict_data['FPR'],dict_data['TPR'])
        plt.title('ROC curve train  with AUC: '+str(round(dict_data['AUC'],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        ax2 = plt.subplot(122)
        ax2.hist(dict_data['t_m'], label = 'True', color = 'red', alpha = 0.5 , bins = 50, range=(-2,5))
        ax2.hist(dict_data['f_tr_m'],label = 'False',color = 'green', alpha = 0.5,bins = 50, range=(-2,5))
        plt.legend()
        plt.savefig(path_p + 'AUC.png')

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)

        ax1.plot(dict_data['FPR_v'], dict_data['TPR_v'])
        plt.title('ROC curve validation with AUC: '+str(round(dict_data['AUC_v'],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        ax2 = plt.subplot(122)
        ax2.hist(dict_data['t_m'], label='True', color='red', alpha=0.5, bins=50,range=(-2,5))
        ax2.hist(dict_data['f_v_m'], label='False', color='green', alpha=0.5, bins=50,range=(-2,5))
        plt.legend()
        plt.savefig(path_p + 'AUC_val.png')



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


    def get_data(self,dict_data,i):

        self.df_t_train = dict_data['df_t_train']
        self.df_t_val = dict_data['df_t_val']

        self.df_f_train = dict_data['df_f_train']
        self.df_f_val = dict_data['df_f_val']

        self.dimension = self.df_t_train.iloc[0]['data_X'].shape[2]

        train_loss = dict_data['losses']

        self.AUC_max = 0
        self.AUC_min = 100


        array_t                             = zip(list(self.df_t_train['data_y']),list(self.df_t_train['data_y_p']))
        array_t_m                           = list(map(self._get_error_m,array_t))
        array_t                             = zip(list(self.df_t_train['data_y']),list(self.df_t_train['data_y_p']))
        val_t_train                         = list(map(self._get_error_mean,array_t))

        array_t                             = zip(list(self.df_t_val['data_y']),list(self.df_t_val['data_y_p']))
        array_t_m.extend(list(map(self._get_error_m,array_t)))
        array_t                             = zip(list(self.df_t_val['data_y']),list(self.df_t_val['data_y_p']))
        val_t_train.extend(list(map(self._get_error_mean,array_t)))


        array_f                             = zip(list(self.df_f_train['data_y']),list(self.df_f_train['data_y_p']))
        array_f_tr                          = list(map(self._get_error_m,array_f))
        array_f                             = zip(list(self.df_f_train['data_y']),list(self.df_f_train['data_y_p']))
        train_f                             = list(map(self._get_error_mean,array_f))

        array_fv                            = zip(list(self.df_f_val['data_y']),list(self.df_f_val['data_y_p']))
        val_f                               = list(map(self._get_error_mean,array_fv))
        array_fv                            = zip(list(self.df_f_val['data_y']),list(self.df_f_val['data_y_p']))
        array_f_v                           = list(map(self._get_error_m,array_fv))


        AUC, FPR, TPR = self.get_AUC_score(array_t_m, array_f_tr)
        AUC_v, FPR_v, TPR_v = self.get_AUC_score(array_t_m, array_f_v)



        loss_f_t     = np.mean(train_f)
        loss_f_v     = np.mean(val_f)
        loss_t       = np.mean(val_t_train)


        loss_t_std      = np.std(val_t_train)
        loss_f_t_std   = np.std(train_f)
        loss_f_v_std   = np.std(val_f)

        dict_data   =  {
                        'AUC'        : AUC,


                        'FPR'            : FPR,
                        'TPR'            : TPR,
                        'val_f'          : loss_f_v,
                        'val_t'          : loss_t,
                        'train_f'        : loss_f_t,
                        'train_std'      : loss_f_t_std,
                        'val_std_f'      : loss_f_v_std,
                        'val_std_t'      : loss_t_std,


                        'AUC_v'          : AUC_v,
                        'TPR_v'          : TPR_v,
                        'FPR_v'          : FPR_v,

                        'path_o'         : dict_data['path_o'],

                        't_m'            : array_t_m,
                        'f_tr_m'         : array_f_tr,
                        'f_v_m'          : array_f_v
                        }

        path = dict_data['path_o'] + 'hist.csv'
        df = pd.DataFrame([dict_data])[['AUC','train_f','val_f','val_t','val_std_t','val_std_f','train_std','AUC_v','TPR_v','FPR_v','TPR','FPR']]
        if(i == 0):
            df.to_csv(path)
        else:
            df.to_csv(path, mode = 'a', header = False)

        return dict_data

    def _get_error_m(self, row):

        y     = row[0]
        y_p   = row[1]


        e_f = np.max(np.mean(np.power((y - y_p),2),axis=(1,2)))

        return e_f

    def _get_error_mean(self, row):

        y     = row[0]
        y_p   = row[1]

        e_f = np.mean(np.power((y - y_p), 2))



        return e_f


    def save_output_CMA(self,df,dict_,path):
        path_d = path+'data_CMA/'
        if (os.path.exists(path_d) == False):
            os.mkdir(path_d)

        path_n = path_d + '/nr_'+str(len(os.listdir(path_d)))
        if (os.path.exists(path_n) == False):
            os.mkdir(path_n)

        string = 'experiment_'+str(len(os.listdir(path))-1)
        path   = path+string

        pickle_save_(path_n+'/df.p',df)
        pickle_save_(path_n+'/dict.p',dict_)
        fig = plt.figure(figsize=(16, 4))

        ax1 = plt.subplot(131)
        for i in range(len(df)):
            ax1.plot(df['FPR'].iloc[i],df['TPR'].iloc[i], label = 'CV_'+str(i)+'_'+str(round(df['AUC'].iloc[i],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('Train AUC')


        ax2 = plt.subplot(132)
        for i in range(len(df)):
            ax2.plot(df['FPR_v'].iloc[i],df['TPR_v'].iloc[i], label = 'CV_'+str(i)+'_'+str(round(df['AUC_v'].iloc[i],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('val AUC')

        ax3 = plt.subplot(133)
        for i in range(len(df)):
            ax3.plot(df['FPR_t'].iloc[i],df['TPR_t'].iloc[i], label = 'CV_'+str(i)+'_'+str(round(df['AUC_t'].iloc[i],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('test AUC')
        plt.savefig(path_n +'/AUC_curve.png')



        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)
        for i in range(len(df)):
            ax1.plot(df['x'].iloc[i], label = 'x_'+str(i)+'_'+str(round(df['AUC'].iloc[i],3)))
        ax1.plot(dict_['x'], color = 'k', linewidth = 3)
        plt.xlabel('weigths')
        plt.ylabel('value')
        plt.legend()
        plt.title('Weights cma')


        ax2 = plt.subplot(122)
        ax2.plot(dict_['FPR_tr'],dict_['TPR_tr'], label = 'train ROC')
        ax2.plot(dict_['FPR_t'],dict_['TPR_t'] ,label = 'test ROC')

        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('CV_score: '+str(round(-dict_['CV_score'],4))+' Test_score: '+str(round(-dict_['CV_score'],4))+
                             ' Train_score: '+str(round(dict_['AUC_tr'],4)))

        plt.savefig(path_n+'/AUC_T.png')






