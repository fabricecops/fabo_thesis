from src.models.LSTM.configure import return_dict_bounds
from src.models.LSTM.main_LSTM import model_mng
from src.dst.outputhandler.pickle import tic,toc,pickle_save_,pickle_load
import numpy as np
import matplotlib.pyplot as plt
import os
from memory_profiler import profile

class model_tests():


    def __init__(self,dict_c):
        self.dict_c    = dict_c
        self.iteration = 15



    def get_rest_of_data(self):
        dict_ ={
            'shuffle_random': {
                'train': [],
                'val' : [],
                'time' : [],
                            },

            'shuffle_segmentated': {
                'train': [],
                'val'  : [],
                'time' : [],
                            }
        }

        path = 'models/variance/shuffle_random/'
        dir_ = os.listdir(path)

        for directory in dir_:
            path_v = path+directory+'/AUC_CMA.p'
            hist   = pickle_load(path_v,None)
            AUC_tr = np.max(hist['AUC'])
            AUC_v  = np.max(hist['AUC_v'])
            AUC_t  = np.max(hist['AUC_t'])


            dict_['shuffle_random']['train'].append(AUC_tr)
            dict_['shuffle_random']['val'].append(AUC_v)
            dict_['shuffle_random']['val'].append(AUC_t)


        return dict_

    @profile
    def variance_calculation(self):
        dict_ = pickle_load('./models/variance/variance.p', None)

        # ##### normal shuffle
        # for i in range(0):
        #     dict_ = self.train_model(dict_,'random')
        #
        #
        # dict_['shuffle_random']['train_mean'] = np.mean(dict_['shuffle_random']['train'])
        # dict_['shuffle_random']['train_std']  = np.std(dict_['shuffle_random']['train'])
        #
        # dict_['shuffle_random']['val_mean']   = np.mean(dict_['shuffle_random']['val'])
        # dict_['shuffle_random']['val_std']    = np.std(dict_['shuffle_random']['val'])
        #
        #
        # ##### normal shuffle
        # for i in range(self.iteration):
        #     dict_ = self.train_model(dict_,'segmentated')
        #
        #
        # dict_['shuffle_segmentated']['train_mean'] = np.mean(dict_['shuffle_segmentated']['train'])
        # dict_['shuffle_segmentated']['train_std']  = np.std(dict_['shuffle_segmentated']['train'])
        #
        # dict_['shuffle_segmentated']['val_mean']   = np.mean(dict_['shuffle_segmentated']['val'])
        # dict_['shuffle_segmentated']['val_std']    = np.std(dict_['shuffle_segmentated']['val'])




        path = './models/variance/'
        fig = plt.figure(figsize=(16, 4))

        ax1 = plt.subplot(131)
        ax1.hist(dict_['shuffle_random']['train'],      label = 'random',color = 'g', alpha = 0.5, bins = 10)
        ax1.hist(dict_['shuffle_segmentated']['train'], label = 'segmentated', color = 'r', alpha = 0.5, bins = 10)
        plt.xlabel('AUC')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Train distribution')


        ax2 = plt.subplot(132)
        ax2.hist(dict_['shuffle_random']['val'],      label = 'random',color = 'g', alpha = 0.5, bins = 10)
        ax2.hist(dict_['shuffle_segmentated']['val'], label = 'segmentated', color = 'r', alpha = 0.5, bins = 10)
        plt.xlabel('AUC')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('val distribution')

        ax3 = plt.subplot(133)
        ax3.hist(dict_['shuffle_random']['time'],      label = 'random',color = 'g', alpha = 0.5, bins = 10)
        ax3.hist(dict_['shuffle_segmentated']['time'], label = 'segmentated', color = 'r', alpha = 0.5, bins = 10)
        plt.xlabel('AUC')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Time distribution')


        plt.savefig(path+'distribution.png')
        plt.show()

        plt.close('all')

    @profile
    def train_model(self,dict_,mode):

        if(mode == 'random'):
            self.dict_c['path_save']     = './models/variance/shuffle_random/'
            self.dict_c['shuffle_style'] = 'random'
        else:
            self.dict_c['path_save'] = './models/variance/shuffle_segmentated/'
            self.dict_c['shuffle_style'] = 'segmentated'

        tic()
        mm = model_mng(self.dict_c)
        AUC_tr, AUC_v, AUC_t = mm.main(mm.Queue_cma)
        del mm

        elapsed = toc()

        if mode == 'random':

            dict_['shuffle_random']['train'].append(AUC_tr)
            dict_['shuffle_random']['val'].append(AUC_v)
            dict_['shuffle_random']['val'].append(AUC_t)
            dict_['shuffle_random']['time'].append(elapsed)
        else:
            dict_['shuffle_segmentated']['train'].append(AUC_tr)
            dict_['shuffle_segmentated']['val'].append(AUC_v)
            dict_['shuffle_segmentated']['val'].append(AUC_t)
            dict_['shuffle_segmentated']['time'].append(elapsed)

        path = './models/variance/'
        pickle_save_(path+'variance.p', dict_)


        return dict_








if __name__ == '__main__':
    if __name__ == '__main__':
        dict_c, bounds = return_dict_bounds()

        mm = model_tests(dict_c)
        mm.variance_calculation()