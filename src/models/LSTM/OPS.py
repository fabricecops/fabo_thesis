from src.dst.outputhandler.pickle import pickle_save
import os
import pandas as pd
import matplotlib.pyplot as plt
import functools
plt.style.use('ggplot')
import numpy as np
from src.dst.metrics.AUC import AUC
from src.dst.outputhandler.pickle import pickle_save_,pickle_load

class OPS_LSTM(AUC):

    def __init__(self,dict_c):
        self.dict_c = dict_c
        AUC.__init__(self,dict_c)

    def main(self,dict_data):

        dict_data2           = self._get_data_no_cma(dict_data)
        self.save_plots_no_cma(dict_data2)

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

    def save_plots_no_cma(self,dict_data):


        ### validation curve
        df = pickle_load(dict_data['path_o']+'hist.p', None)

        path_b = dict_data['path_o'] +'best/'
        if (os.path.exists(path_b) == False):
            os.mkdir(path_b)


        try:
            df_CMA = pickle_load(dict_data['path_o']+'AUC_CMA.p', None)
        except:
            df_CMA = {
                'AUC_v': [0.5],
                'AUC_t': [0.5],
                'AUC'  : [0.5]
            }





        fig = plt.figure(figsize=(16, 4))


        ax1 = plt.subplot(131)
        ax1.plot(np.array(df['train_t']),    color = 'm',   label = 'train_t')
        ax1.plot(np.array(df['val_t']),    color = 'red',   label = 'val_t')
        ax1.plot(np.array(df['test_t']),    color = 'k',   label = 'test_t')

        ax1.plot(np.array(df['train_f']),  color = 'blue',  label = 'train_f')
        ax1.plot(np.array(df['val_f']),    color = 'green', label = 'val_f')
        ax1.plot(np.array(df['test_f']),  color = 'yellow',  label = 'test_f')
        ax1.set_ylim(0.0,0.025)

        plt.legend()
        plt.title('validation/train curve')
        plt.xlabel('epoch nr')
        plt.ylabel('average loss')

        ax2 = plt.subplot(132)
        ax2.plot(np.array(df['AUC']), color = 'r', label = 'train')
        ax2.plot(np.array(df['AUC_v']), color = 'g', label = 'val')
        ax2.plot(np.array(df['AUC_t']), color = 'b', label = 'test')

        plt.title('train/val/test AUC no CMA')
        plt.xlabel('epoch nr')
        plt.ylabel('AUC')
        plt.legend()

        ax3 = plt.subplot(133)
        ax3.plot(np.array(df_CMA['AUC']), color = 'r', label = 'train')
        ax3.plot(np.array(df_CMA['AUC_v']), color = 'g', label = 'val')
        ax3.plot(np.array(df_CMA['AUC_t']), color = 'b', label = 'test')

        plt.title('train/val/test AUC  CMA')
        plt.xlabel('epoch nr')
        plt.ylabel('AUC')
        plt.legend()
        plt.savefig(dict_data['path_o'] + '/val_curve.png')


        if (dict_data['AUC_v'] >= max(list(df['AUC_v']))):

            fig = plt.figure(figsize=(16, 4))
            ax1 = plt.subplot(141)

            ax1.plot(dict_data['FPR'],dict_data['TPR'],label= 'train')
            ax1.plot(dict_data['FPR_v'],dict_data['TPR_v'],label= 'val')
            ax1.plot(dict_data['FPR_t'],dict_data['TPR_t'],label= 'test')
            plt.legend()

            plt.title('ROC curve no CMA at epoch: '+str(dict_data['epoch']))
            plt.xlabel('FPR')
            plt.ylabel('TPR')

            ax2 = plt.subplot(142)
            ax2.hist(dict_data['t_tr_m'], label = 'True', color = 'red', alpha = 0.5 , bins = 50)
            ax2.hist(dict_data['f_tr_m'],label = 'False',color = 'green', alpha = 0.5,bins = 50)
            plt.legend()
            plt.title('Train set: '+str(round(dict_data['AUC'],3)))
            plt.xlabel('error')
            plt.ylabel('frequency')

            ax3 = plt.subplot(143)
            ax3.hist(dict_data['t_v_m'], label = 'True', color = 'red', alpha = 0.5 , bins = 50)
            ax3.hist(dict_data['f_v_m'],label = 'False',color = 'green', alpha = 0.5,bins = 50)
            plt.title('Val set: '+str(round(dict_data['AUC_v'],3)))
            plt.xlabel('error')
            plt.ylabel('frequency')
            plt.legend()

            ax4 = plt.subplot(144)
            ax4.hist(dict_data['t_t_m'], label = 'True', color = 'red', alpha = 0.5 , bins = 50)
            ax4.hist(dict_data['f_t_m'],label = 'False',color = 'green', alpha = 0.5,bins = 50)
            plt.legend()
            plt.title('test set: '+str(round(dict_data['AUC_t'],3)))
            plt.xlabel('error')
            plt.ylabel('frequency')

            plt.savefig(path_b+'AUC_best_no_CMA_val.png')



        plt.close('all')

    def save_output_CMA(self,dict_):

        path   = dict_['path_o']


        path_AUC_CMA = path+'AUC_CMA.p'
        df = pd.DataFrame([dict_])[['AUC', 'AUC_v','AUC_t']]
        if (dict_['epoch'] == 0):
            pickle_save(path_AUC_CMA,df)
        else:
            df_saved = pickle_load(path_AUC_CMA,None)
            df_saved = df_saved.append(df,ignore_index=False)
            pickle_save(path_AUC_CMA,df_saved)
            df       = df_saved



        path_b = dict_['path_o']+'best/'
        if (os.path.exists(path_b) == False):
            os.mkdir(path_b)


        if (dict_['AUC_v'] >= max(list(df['AUC_v']))):

            fig = plt.figure(figsize=(16, 4))

            ax1 = plt.subplot(141)
            ax1.plot(dict_['FPR'],dict_['TPR'], label = 'train')
            ax1.plot(dict_['FPR_v'],dict_['TPR_v'], label = 'val')
            ax1.plot(dict_['FPR_t'],dict_['TPR_t'], label = 'test')

            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()
            plt.title('AUC CMA at epoch: '+str(dict_['epoch']))


            ax2 = plt.subplot(142)
            ax2.hist(dict_['df_f_train']['error_m'], label = 'False',color = 'g', alpha = 0.5, bins = 50)
            ax2.hist(dict_['df_t_train']['error_m'], label = 'True', color = 'r', alpha = 0.5, bins = 50)
            plt.xlabel('Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('Train distribution: '+str(round(dict_['AUC'],3))+' at epoch: '+str(dict_['epoch']))


            ax2 = plt.subplot(143)
            ax2.hist(dict_['df_f_val']['error_m'], label = 'False',color = 'g', alpha = 0.5, bins = 50)
            ax2.hist(dict_['df_t_val']['error_m'], label = 'True',color = 'r', alpha = 0.5, bins = 50)
            plt.xlabel('Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('val distribution: '+str(round(dict_['AUC_v'],3)))

            ax3 = plt.subplot(144)
            ax3.hist(dict_['df_f_test']['error_m'], label = 'False',color = 'g', alpha = 0.5, bins = 50)
            ax3.hist(dict_['df_t_test']['error_m'], label = 'True',color = 'r', alpha = 0.5, bins = 50)
            plt.xlabel('Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('test distribution: '+str(round(dict_['AUC_t'],3)))

            plt.savefig(path_b+'AUC_best_cma_dist.png')

            fig = plt.figure(figsize=(16, 4))
            plt.plot(dict_['x'], color = 'k', linewidth = 3)
            plt.xlabel('weigths')
            plt.ylabel('value')
            plt.title('Weights cma')

            plt.savefig(path_b+'best_weights.png')

            plt.close('all')

    def save_ROC_segment(self,dict_data):

        path         = dict_data['path_o']
        path_AUC_CMA = path+'AUC_CMA.p'
        df           = pickle_load(path_AUC_CMA, None)

        path_save    = dict_data['path_o']+'best/'

        if (dict_data['AUC_v'] >= max(list(df['AUC_v']))):
            dict_data = self._get_data_segmented(dict_data)


            fig = plt.figure(figsize=(16, 4))
            ax1 = plt.subplot(141)

            ax1.plot(dict_data['FPR'],dict_data['TPR'],color = 'k',label = 'MAIN')
            for key in dict_data['dict_combined'].keys():
                data =  dict_data['dict_combined'][key]
                ax1.plot(data[1], data[2], label= key+'_'+str(round(data[0],2)))

            plt.legend()
            plt.title('ROC segmentated combined')
            plt.xlabel('FPR')
            plt.ylabel('TPR')

            ax2 = plt.subplot(142)
            ax2.plot(dict_data['FPR'],dict_data['TPR'],color = 'k',label = 'MAIN')
            for key in dict_data['dict_train'].keys():
                data =  dict_data['dict_train'][key]
                ax2.plot(data[1], data[2], label= key+'_'+str(round(data[0],2)))

            plt.legend()
            plt.title('ROC segmentated training')
            plt.xlabel('FPR')
            plt.ylabel('TPR')

            ax3 = plt.subplot(143)
            ax3.plot(dict_data['FPR'], dict_data['TPR'], color='k', label='MAIN')
            for key in dict_data['dict_val'].keys():
                data = dict_data['dict_val'][key]
                ax3.plot(data[1], data[2], label=key + '_' + str(round(data[0], 2)))

            plt.legend()
            plt.title('ROC segmentated val')
            plt.xlabel('FPR')
            plt.ylabel('TPR')

            ax4 = plt.subplot(144)
            ax4.plot(dict_data['FPR'], dict_data['TPR'], color='k', label='MAIN')
            for key in dict_data['dict_test'].keys():
                data = dict_data['dict_test'][key]
                ax4.plot(data[1], data[2], label=key + '_' + str(round(data[0], 2)))

            plt.legend()
            plt.title('ROC segmentated test')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.savefig(path_save+'segmented_ROC.png')


            plt.close('all')
            if(dict_data['epoch'] == 0):
                fig = plt.figure(figsize=(16, 4))

                bottom2 = [x+y for x,y in zip(dict_data['dict_bar']['train'],dict_data['dict_bar']['val'])]

                indexes = range(len(dict_data['dict_bar']['train']))
                plt.bar(indexes,dict_data['dict_bar']['train'], align = 'center',label = 'train')
                plt.bar(indexes,dict_data['dict_bar']['val'], align="center",bottom = dict_data['dict_bar']['train'],label = 'val')
                plt.bar(indexes,dict_data['dict_bar']['test'], align="center",bottom = bottom2,label = 'test')
                plt.xticks(range(len(dict_data['dict_bar']['groups'])), dict_data['dict_bar']['groups'])
                plt.legend()
                plt.xlabel('Groups')
                plt.ylabel('Frequency')
                plt.title('Distribution segmentation')
                plt.savefig(dict_data['path_o']+'dist_segmentation.png')









    def _get_data_segmented(self,dict_data):

        df_t_train = dict_data['df_t_train']
        df_t_val   = dict_data['df_t_val']
        df_t_test  = dict_data['df_t_test']

        df_f_train = dict_data['df_f_train']
        df_f_val   = dict_data['df_f_val']
        df_f_test  = dict_data['df_f_test']



        dict_train = {}
        for group in df_t_train.groupby('segmentation'):


            AUC, FPR, TPR        = self.get_AUC_score(group[1]['error_m'], df_f_train['error_m'])
            dict_train[group[0]] = [AUC,FPR,TPR]

        dict_val   = {}
        for group in df_t_val.groupby('segmentation'):
            AUC, FPR, TPR        = self.get_AUC_score(group[1]['error_m'], df_f_val['error_m'])
            dict_val[group[0]]   = [AUC,FPR,TPR]

        dict_test = {}
        for group in df_t_test.groupby('segmentation'):

            AUC, FPR, TPR        = self.get_AUC_score(group[1]['error_m'],  df_f_test['error_m'])
            dict_test[group[0]] = [AUC,FPR,TPR]


        df_t  = df_t_train.append(df_t_val,ignore_index = True)
        df_t  = df_t.append(df_t_test      ,ignore_index = True)

        dict_combined = {}
        for group in df_t.groupby('segmentation'):

            AUC, FPR, TPR            = self.get_AUC_score(group[1]['error_m'], df_f_val['error_m'])
            dict_combined[group[0]]  =  [AUC,FPR,TPR,len(group[0])]

        groups   = ['gooien','onder','boven','muren','sneaky','object']
        array_tr = [0,0,0,0,0,0]
        array_v  = [0,0,0,0,0,0]
        array_t  = [0,0,0,0,0,0]


        for i,group in enumerate(groups):

            array_tr[i] = len(df_t_train[df_t_train['segmentation'] == group])
            array_v[i]  = len(df_t_val[df_t_val['segmentation'] == group])
            array_t[i]  = len(df_t_test[df_t_test['segmentation'] == group])

        dict_bar  = {
                    'groups': groups,
                    'train' : array_tr,
                    'val'   : array_v,
                    'test'  : array_t
        }



        dict_data['dict_train']     = dict_train
        dict_data['dict_val']       = dict_val
        dict_data['dict_test']      = dict_test
        dict_data['dict_combined']  = dict_combined
        dict_data['dict_bar']       = dict_bar



        return dict_data




    def _get_data_no_cma(self,dict_data):
        i               = dict_data['epoch']
        self.df_t_train = dict_data['df_t_train']
        self.df_t_val   = dict_data['df_t_val']
        self.df_t_test  = dict_data['df_t_test']

        self.df_f_train = dict_data['df_f_train']
        self.df_f_val   = dict_data['df_f_val']
        self.df_f_test  = dict_data['df_f_test']

        self.dimension = self.df_t_train.iloc[0]['data_X'].shape[2]


        loss_f_tr       = dict_data['loss_f_tr']
        loss_f_v        = dict_data['loss_f_v']


        self.AUC_max = 0
        self.AUC_min = 100


        array_t                             = zip(list(self.df_t_train['data_y']),list(self.df_t_train['data_y_p']))
        array_t_m_tr                        = list(map(self._get_error_m,array_t))

        array_t                             = zip(list(self.df_t_val['data_y']),list(self.df_t_val['data_y_p']))
        array_t_m_v                         = list(map(self._get_error_m,array_t))

        array_t                             = zip(list(self.df_t_test['data_y']),list(self.df_t_test['data_y_p']))
        array_t_m_t                         = list(map(self._get_error_m,array_t))

        array_f                             = zip(list(self.df_f_train['data_y']),list(self.df_f_train['data_y_p']))
        array_f_m_tr                        = list(map(self._get_error_m,array_f))

        array_f                             = zip(list(self.df_f_val['data_y']),list(self.df_f_val['data_y_p']))
        array_f_m_v                         = list(map(self._get_error_m,array_f))

        array_f                             = zip(list(self.df_f_test['data_y']), list(self.df_f_test['data_y_p']))
        array_f_m_t                         = list(map(self._get_error_m, array_f))


        AUC, FPR, TPR       = self.get_AUC_score(array_t_m_tr, array_f_m_tr)
        AUC_v, FPR_v, TPR_v = self.get_AUC_score(array_t_m_v, array_f_m_v)
        AUC_t, FPR_t, TPR_t = self.get_AUC_score(array_t_m_t, array_f_m_t)

        loss_t_tr, loss_t_v, loss_t_t, loss_f_t = self._calc_loss()
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

                        'path_o'         : dict_data['path_o'],
                        'epoch'          : dict_data['epoch'],

                        't_tr_m'         : array_t_m_tr,
                        't_v_m'          : array_t_m_v,
                        't_t_m'          : array_t_m_t,

                        'f_tr_m'         : array_f_m_tr,
                        'f_v_m'          : array_f_m_v,
                        'f_t_m'          : array_f_m_t,
                        }

        path = dict_data['path_o'] + 'hist.p'
        df = pd.DataFrame([dict_data])[['AUC','AUC_v','AUC_t',
                                        'train_f','train_t','val_f','val_t','test_t','test_f']]



        if(i == 0):
            pickle_save(path,df)
        else:

            df_saved = pickle_load(path,None)
            df_saved = df_saved.append(df,ignore_index=False)


            pickle_save(path,df_saved)

        return dict_data

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
        t_tr_y    = np.concatenate(list(self.df_t_train['data_y']))
        t_tr_yp   = np.concatenate(list(self.df_t_train['data_y_p']))
        t_e_tr    = np.mean(np.power((t_tr_y-t_tr_yp),2),axis = (1,2))
        loss_t_tr = np.mean(t_e_tr)
        del t_tr_y,t_tr_yp,t_e_tr

        t_v_y    = np.concatenate(list(self.df_t_val['data_y']))
        t_v_yp   = np.concatenate(list(self.df_t_val['data_y_p']))
        t_e_v    = np.mean(np.power((t_v_y-t_v_yp),2),axis = (1,2))
        loss_t_v = np.mean(t_e_v)
        del t_v_y,t_v_yp,t_e_v

        t_t_y    = np.concatenate(list(self.df_t_test['data_y']))
        t_t_yp   = np.concatenate(list(self.df_t_test['data_y_p']))
        t_e_t    = np.mean(np.power((t_t_y-t_t_yp),2),axis = (1,2))
        loss_t_t = np.mean(t_e_t)
        del t_t_y,t_t_yp,t_e_t


        f_t_y     = np.concatenate(list(self.df_f_test['data_y']))
        f_t_yp    = np.concatenate(list(self.df_f_test['data_y_p']))
        f_e_t     = np.mean(np.power((f_t_y-f_t_yp),2),axis = (1,2))
        loss_f_t  = np.mean(f_e_t)
        del f_t_y,f_t_yp,f_e_t

        return loss_t_tr,loss_t_v,loss_t_t,loss_f_t
