from src.dst.outputhandler.pickle import pickle_save
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

class OPS():

    def __init__(self,dict_c):
        self.verbose_AUC = dict_c['verbose_AUC']



    def main_OPS(self,dict_data,epoch):
        dir_,df = self._save_prediction(dict_data,epoch)
        self._save_plots(dict_data,dir_)
        self._verbose(epoch, df)


    def _save_prediction(self,dict_data,epoch):
        string = 'epoch_'+str(epoch)
        dir_   = dict_data['path_o'] + 'predictions/'+string


        df_t_train = dict_data['df_t_train'][['error_tm']]
        df_t_val   = dict_data['df_t_val'][['error_tm']]
        df_f_train = dict_data['df_f_train'][['error_tm']]
        df_f_val   = dict_data['df_f_val'][['error_tm']]

        dict_p = {
                'df_t_train': df_t_train,
                'df_t_val'  : df_t_val,
                'df_f_train': df_f_train,
                'df_f_val'  : df_f_val,
                'x'         : dict_data['x']
            }

        path_p = dir_+ '/pred.p'
        pickle_save(path_p, dict_p)




        path = dict_data['path_o'] + 'hist.csv'
        df = pd.DataFrame([dict_data])[['AUC_min','AUC_max','train_f','val_f','val_t','val_std_t','val_std_f','train_std','AUC_v','TPR_v','FPR_v','TPR','FPR']]
        if(epoch == 0):
            df.to_csv(path)
        else:
            df.to_csv(path, mode = 'a', header = False)

        return dir_,df

    def _save_output(self,dict_data,i):
        if(i==0):
            df_o_t = dict_data['df_true'][['frames', 'name', 'label', 'data_X', 'data_y']]
            df_o_f = dict_data['df_false_T'][['frames', 'name', 'label', 'data_X', 'data_y']]

            dict_o = {
                'df_o_t': df_o_t,
                'df_o_f': df_o_f
            }


            path_o = dict_data['path_o'] + 'output.p'
            pickle_save(path_o, dict_o)



    def _verbose(self,epoch,df):
        if (self.verbose_AUC == 1):
            AUC_max = df.iloc[0]['AUC_max']
            AUC_min = df.iloc[0]['AUC_min']
            train_f = df.iloc[0]['train_f']
            val_f   = df.iloc[0]['val_f']
            val_t   = df.iloc[0]['val_t']


            print()
            print('The AUC for epoch ', epoch, ' is equal to: ',(AUC_min,AUC_max ))
            print('Train f: ',round(train_f,4),' Val_f: ',round(val_f,4),' val_t: ',round(val_t,4))

    def _save_plots(self,dict_data,dir):

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
        plt.plot(df['AUC_v'], color = 'g')
        ax2.fill_between(range(len(df)),df['AUC_min'],df['AUC_max'], color = 'red', alpha = 0.7,label = 'AUC')
        plt.title('MIN/MAX AUC after doing CMA_ES')
        plt.xlabel('epoch nr')
        plt.ylabel('AUC')
        plt.savefig(dict_data['path_o'] + '/val_curve.png')


        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)

        ax1.plot(dict_data['FPR'],dict_data['TPR'])
        plt.title('ROC curve train  with AUC: '+str(round(dict_data['AUC_max'],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        ax2 = plt.subplot(122)
        ax2.hist(dict_data['df_true']['error_tm'], label = 'True', color = 'red', alpha = 0.5 , bins = 50, range=(-2,5))
        ax2.hist(dict_data['df_false']['error_tm'],label = 'False',color = 'green', alpha = 0.5,bins = 50, range=(-2,5))
        plt.legend()
        plt.savefig(dir + '/AUC.png')

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)

        ax1.plot(dict_data['FPR_v'], dict_data['TPR_v'])
        plt.title('ROC curve validation with AUC: '+str(round(dict_data['AUC_v'],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        ax2 = plt.subplot(122)
        ax2.hist(dict_data['df_true']['error_tm'], label='True', color='red', alpha=0.5, bins=50,range=(-2,5))
        ax2.hist(dict_data['df_false_val']['error_tm'], label='False', color='green', alpha=0.5, bins=50,range=(-2,5))
        plt.legend()
        plt.savefig(dir + '/AUC_val.png')


        if(dict_data['AUC_v'] == max(np.array(df['AUC_v']))):
            df_p_t = dict_data['df_t_val'][['error_tm','data_y_p']]
            df_p_f = dict_data['df_f_val'][['error_tm','data_y_p']]


            dict_p = {
                'df_t_val': df_p_t,
                'df_f_val': df_p_f,
                'x'     : dict_data['x']
            }

            path_p = dict_data['path_o'] + '/pred.p'
            pickle_save(path_p, dict_p)










