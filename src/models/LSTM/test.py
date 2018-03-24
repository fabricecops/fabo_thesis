from src.models.LSTM.configure import return_dict_bounds
from src.models.LSTM.main_LSTM import model_mng
from src.dst.outputhandler.pickle import tic,toc,pickle_save_,pickle_load
import numpy as np
import matplotlib.pyplot as plt
import os
from memory_profiler import profile
import multiprocessing as mp

class model_tests():


    def __init__(self,dict_c):
        self.dict_c    = dict_c
        self.iteration = 20
        self.path      = './models/variance/'

        self.testing   = False


    def variance_calculation(self):
        dict_ = pickle_load('./models/variance/variance.p', None)
        # dict_   = self._return_dict_()

        for i in range(self.iteration-13):
            self.dict_c['random_state']  = (i+13)*50
            self.dict_c['path_save']     = './models/variance/shuffle_segmentated/'
            self.dict_c['shuffle_style'] = 'segmentated'
            dict_ = self._train_model(dict_=dict_)

        dict_['shuffle_segmentated']['train_mean'] = np.mean(dict_['shuffle_segmentated']['train'])
        dict_['shuffle_segmentated']['train_std'] = np.std(dict_['shuffle_segmentated']['train'])

        dict_['shuffle_segmentated']['val_mean'] = np.mean(dict_['shuffle_segmentated']['val'])
        dict_['shuffle_segmentated']['val_std'] = np.std(dict_['shuffle_segmentated']['val'])


        for i in range(self.iteration):
            self.dict_c['random_state']  = i*50
            self.dict_c['shuffle_style'] = 'random'
            self.dict_c['path_save']     = './models/variance/shuffle_random/'

            dict_ = self._train_model(dict_ = dict_)


        dict_['shuffle_random']['train_mean'] = np.mean(dict_['shuffle_random']['train'])
        dict_['shuffle_random']['train_std']  = np.std(dict_['shuffle_random']['train'])

        dict_['shuffle_random']['val_mean']   = np.mean(dict_['shuffle_random']['val'])
        dict_['shuffle_random']['val_std']    = np.std(dict_['shuffle_random']['val'])
        pickle_save_(self.path+'variance.p', dict_)


        pickle_save_(self.path+'variance.p', dict_)



    def _train_model(self,dict_):

        if self.testing == True:
            self.dict_c['shuffle_style'] = 'testing'
            self.dict_c['evals']         = 2
            self.dict_c['epochs']        = 3


        tic()
        mm = model_mng(self.dict_c)
        AUC_tr, AUC_v, AUC_t = mm.main(mm.Queue_cma)
        elapsed = toc()

        if self.dict_c['shuffle_style'] == 'random':

            dict_['shuffle_random']['train'].append(AUC_tr)
            dict_['shuffle_random']['val'].append(AUC_v)
            dict_['shuffle_random']['val'].append(AUC_t)
            dict_['time'].append(elapsed)

        elif self.dict_c['shuffle_style'] == 'segmentated':
            dict_['shuffle_segmentated']['train'].append(AUC_tr)
            dict_['shuffle_segmentated']['val'].append(AUC_v)
            dict_['shuffle_segmentated']['val'].append(AUC_t)
            dict_['time'].append(elapsed)


        pickle_save_(self.path+'variance.p', dict_)
        self._plot_results(dict_)


        return dict_


    def _plot_results(self,dict_):
        fig = plt.figure(figsize=(16, 4))

        ax1 = plt.subplot(131)
        ax1.hist(dict_['shuffle_random']['train'],      label = 'random',color = 'g', alpha = 0.5, bins = 10)
        ax1.hist(dict_['shuffle_segmentated']['train'],      label = 'segmentated',color = 'b', alpha = 0.5, bins = 10)

        plt.xlabel('AUC')
        plt.ylabel('Amount')
        plt.legend()
        plt.title('Train distribution')


        ax2 = plt.subplot(132)
        ax2.hist(dict_['shuffle_random']['val'],      label = 'random',color = 'g', alpha = 0.5, bins = 10)
        ax2.hist(dict_['shuffle_segmentated']['val'],      label = 'segmentated',color = 'b', alpha = 0.5, bins = 10)

        plt.xlabel('AUC')
        plt.ylabel('Amount')
        plt.legend()
        plt.title('Val/test distribution')

        ax3 = plt.subplot(133)
        ax3.hist(dict_['time'],      label = 'random',color = 'g', alpha = 0.5, bins = 10)
        plt.xlabel('Time')
        plt.ylabel('amount')
        plt.title('Time distribution')


        plt.savefig(self.path+'distribution.png')
        # plt.show()

        plt.close('all')



    def _return_dict_(self):

        dict_ ={


            'shuffle_random': {
                'train': [],
                'val'  : [],
                            },

            'shuffle_segmentated': {
                'train' : [],
                'val'   : [],
            },

            'time'    : []
        }

        return dict_

   # def _get_rest_of_data(self):
    #     dict_ ={
    #         'shuffle_random': {
    #             'train': [],
    #             'val' : [],
    #             'time' : [],
    #                         },
    #
    #         'shuffle_segmentated': {
    #             'train': [],
    #             'val'  : [],
    #             'time' : [],
    #                         }
    #     }
    #
    #     path = 'models/variance/shuffle_random/'
    #     dir_ = os.listdir(path)
    #
    #     for directory in dir_:
    #         path_v = path+directory+'/AUC_CMA.p'
    #         hist   = pickle_load(path_v,None)
    #         AUC_tr = np.max(hist['AUC'])
    #         AUC_v  = np.max(hist['AUC_v'])
    #         AUC_t  = np.max(hist['AUC_t'])
    #
    #
    #         dict_['shuffle_random']['train'].append(AUC_tr)
    #         dict_['shuffle_random']['val'].append(AUC_v)
    #         dict_['shuffle_random']['val'].append(AUC_t)
    #
    #
    #
    #     path = 'models/variance/shuffle_segmentated/'
    #     dir_ = os.listdir(path)
    #     for directory in dir_:
    #         path_v = path+directory+'/AUC_CMA.p'
    #         hist   = pickle_load(path_v,None)
    #         AUC_tr = np.max(hist['AUC'])
    #         AUC_v  = np.max(hist['AUC_v'])
    #         AUC_t  = np.max(hist['AUC_t'])
    #
    #
    #
    #         dict_['shuffle_segmentated']['train'].append(AUC_tr)
    #         dict_['shuffle_segmentated']['val'].append(AUC_v)
    #         dict_['shuffle_segmentated']['val'].append(AUC_t)
    #
    #
    #     return dict_


if __name__ == '__main__':
    if __name__ == '__main__':
        dict_c, bounds = return_dict_bounds()

        mm = model_tests(dict_c)
        # mm.get_rest_of_data()
        mm.variance_calculation()