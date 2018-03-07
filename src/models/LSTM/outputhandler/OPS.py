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
        ax1.plot(df['test_t'],    color = 'k',   label = 'test_t')

        ax1.plot(df['train_f'],  color = 'blue',  label = 'train_f')
        ax1.plot(df['val_f'],    color = 'green', label = 'val_f')
        ax1.plot(df['test_f'],  color = 'yellow',  label = 'test_f')

        plt.legend()
        plt.title('validation/train curve')
        plt.xlabel('epoch nr')
        plt.ylabel('average loss')

        ax2 = plt.subplot(122)
        plt.plot(df['AUC'], color = 'r', label = 'train')
        plt.plot(df['AUC_v'], color = 'g', label = 'val')
        plt.plot(df['AUC_t'], color = 'b', label = 'test')

        plt.title('train/val AUC')
        plt.xlabel('epoch nr')
        plt.ylabel('AUC')
        plt.legend()
        plt.savefig(dict_data['path_o'] + '/val_curve.png')

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)

        ax1.plot(dict_data['FPR'],dict_data['TPR'])
        plt.title('ROC curve train  with AUC: '+str(round(dict_data['AUC'],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        
        ax2 = plt.subplot(122)
        ax2.hist(dict_data['t_m_tr'], label = 'True', color = 'red', alpha = 0.5 , bins = 50)
        ax2.hist(dict_data['f_tr_m'],label = 'False',color = 'green', alpha = 0.5,bins = 50)
        plt.legend()
        plt.savefig(path_p + 'AUC.png')

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)

        ax1.plot(dict_data['FPR_v'], dict_data['TPR_v'])
        plt.title('ROC curve validation with AUC: '+str(round(dict_data['AUC_v'],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        ax2 = plt.subplot(122)
        ax2.hist(dict_data['t_m_tr'], label='True', color='red', alpha=0.5, bins=50)
        ax2.hist(dict_data['f_v_m'], label='False', color='green', alpha=0.5, bins=50)
        plt.legend()
        plt.savefig(path_p + 'AUC_val.png')

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)

        ax1.plot(dict_data['FPR_t'], dict_data['TPR_t'])
        plt.title('ROC curve test with AUC: ' + str(round(dict_data['AUC_t'], 3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        ax2 = plt.subplot(122)
        ax2.hist(dict_data['t_m_t'], label='True', color='red', alpha=0.5, bins=50)
        ax2.hist(dict_data['f_v_m'], label='False', color='green', alpha=0.5, bins=50)
        plt.legend()
        plt.savefig(path_p + 'AUC_test.png')

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
        self.df_t_val   = dict_data['df_t_val']

        self.df_f_train = dict_data['df_f_train']
        self.df_f_val   = dict_data['df_f_val']
        self.df_f_test  = dict_data['df_f_test']

        self.dimension = self.df_t_train.iloc[0]['data_X'].shape[2]


        loss_f_tr       = dict_data['loss_f_tr']
        loss_f_v       = dict_data['loss_f_v']


        self.AUC_max = 0
        self.AUC_min = 100


        array_t                             = zip(list(self.df_t_train['data_y']),list(self.df_t_train['data_y_p']))
        array_t_m_tr                         = list(map(self._get_error_m,array_t))


        array_t                             = zip(list(self.df_t_val['data_y']),list(self.df_t_val['data_y_p']))
        array_t_m_t                         = list(map(self._get_error_m,array_t))

        array_f                             = zip(list(self.df_f_train['data_y']),list(self.df_f_train['data_y_p']))
        array_f_tr                          = list(map(self._get_error_m,array_f))

        array_fv                            = zip(list(self.df_f_val['data_y']),list(self.df_f_val['data_y_p']))
        array_f_v                           = list(map(self._get_error_m,array_fv))

        array_ft = zip(list(self.df_f_test['data_y']), list(self.df_f_test['data_y_p']))
        array_f_te = list(map(self._get_error_m, array_ft))


        AUC, FPR, TPR = self.get_AUC_score(array_t_m_tr, array_f_tr)
        AUC_v, FPR_v, TPR_v = self.get_AUC_score(array_t_m_tr, array_f_v)
        AUC_t, FPR_t, TPR_t = self.get_AUC_score(array_t_m_t, array_f_te)

        loss_f_te, loss_t_v,loss_t_t = self._calc_loss()
        dict_data   =  {
                        'AUC'        : AUC,
                        'FPR'            : FPR,
                        'TPR'            : TPR,

                        'AUC_v'          : AUC_v,
                        'TPR_v'          : TPR_v,
                        'FPR_v'          : FPR_v,


                        'AUC_t'          : AUC_t,
                        'TPR_t'          : TPR_t,
                        'FPR_t'          : FPR_t,

                        'val_f'          : loss_f_v,
                        'train_f'        : loss_f_tr,
                        'test_f'         : loss_f_te,


                        'val_t'          : loss_t_v,
                        'test_t'         : loss_t_t,

                        'path_o'         : dict_data['path_o'],

                        't_m_tr'         : array_t_m_tr,
                        't_m_t'          : array_f_te,

                        'f_tr_m'         : array_f_tr,
                        'f_v_m'          : array_f_v,
                        'f_t_m'          : array_f_te,
                        }

        path = dict_data['path_o'] + 'hist.csv'
        df = pd.DataFrame([dict_data])[['AUC','TPR','FPR','AUC_v','TPR_v','FPR_v','AUC_t','TPR_t','FPR_t',
                                        'train_f','val_f','val_t','test_t','test_f']]
        if(i == 0):
            df.to_csv(path)
        else:
            df.to_csv(path, mode = 'a', header = False)

        return dict_data


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

        string = 'CV_score: '+str(round(-dict_['CV_score'],4))
        print(-dict_['AUC_t'])
        ax2 = plt.subplot(122)
        ax2.plot(dict_['FPR_tr'],dict_['TPR_tr'], label = 'train ROC'+str(round(-dict_['AUC_tr'],4)))
        ax2.plot(dict_['FPR_t'],dict_['TPR_t'] ,label = 'test ROC'+str(round(-dict_['AUC_t'],4)))

        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title(string)

        plt.savefig(path_n+'/AUC_T.png')

    def _get_error_m(self, row):

        y     = row[0]
        y_p   = row[1]


        e_f = np.max(np.mean(np.power((y - y_p),2),axis=(1,2)))

        return e_f

    def _get_error_mean(self, row):

        y     = row[0]
        y_p   = row[1]

        e_f = np.mean(np.power((y - y_p), 2))*y.shape[0]



        return e_f



    def _calc_loss(self):
        t_v_y    = np.concatenate(list(self.df_t_train['data_y']))
        t_t_y     = np.concatenate(list(self.df_t_val['data_y']))

        t_v_yp    = np.concatenate(list(self.df_t_train['data_y_p']))
        t_t_yp     = np.concatenate(list(self.df_t_val['data_y_p']))




        f_t_y     = np.concatenate(list(self.df_f_test['data_y']))
        f_t_yp    = np.concatenate(list(self.df_f_test['data_y_p']))


        e_t_v    = np.mean(np.power((t_v_y-t_v_yp),2),axis = (1,2))
        del t_v_y,t_v_yp

        e_t_t    = np.mean(np.power((t_t_y-t_t_yp),2),axis = (1,2))
        del t_t_y,t_t_yp

        e_f_t = np.mean(np.power((f_t_y-f_t_yp),2),axis = (1,2))
        del f_t_y,f_t_yp



        loss_t_v = np.mean(e_t_v)
        loss_t_t = np.mean(e_t_t)
        loss_f_t = np.mean(e_f_t)


        return loss_f_t,loss_t_v,loss_t_t



