from src.models.LSTM.configure import return_dict_bounds
from src.models.LSTM.main_LSTM import model_mng
from src.dst.outputhandler.pickle import tic,toc,pickle_save_
import numpy as np
import matplotlib.pyplot as plt

class model_tests():


    def __init__(self,dict_c):
        self.dict_c    = dict_c
        self.iteration = 15


    def variance_calculation(self):
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

        self.dict_c['path_save']     = './models/variance/shuffle_random/'
        self.dict_c['shuffle_style'] = 'testing'
        ##### normal shuffle
        for i in range(self.iteration):
            print('x'*50)
            print(i)
            print('x'*50)
            tic()
            mm = model_mng(self.dict_c)
            AUC_tr,AUC_v,AUC_t = mm.main(mm.Queue_cma)
            elapsed = toc()

            dict_['shuffle_random']['train'].append(AUC_tr)
            dict_['shuffle_random']['val'].append(AUC_v)
            dict_['shuffle_random']['val'].append(AUC_t)
            dict_['shuffle_random']['time'].append(elapsed)

        dict_['shuffle_random']['train_mean'] = np.mean(dict_['shuffle_random']['train'])
        dict_['shuffle_random']['train_std']  = np.std(dict_['shuffle_random']['train'])

        dict_['shuffle_random']['val_mean']   = np.mean(dict_['shuffle_random']['val'])
        dict_['shuffle_random']['val_std']    = np.std(dict_['shuffle_random']['val'])

        self.dict_c['path_save']     = './models/variance/shuffle_segmentated/'
        self.dict_c['shuffle_style'] = 'testing'
        ##### normal shuffle
        for i in range(self.iteration):
            print('x'*50)
            print(i)
            print('x'*50)

            tic()
            mm = model_mng(self.dict_c)
            AUC_tr,AUC_v,AUC_t = mm.main(mm.Queue_cma)
            elapsed = toc()

            dict_['shuffle_segmentated']['train'].append(AUC_tr)
            dict_['shuffle_segmentated']['val'].append(AUC_v)
            dict_['shuffle_segmentated']['val'].append(AUC_t)
            dict_['shuffle_segmentated']['time'].append(elapsed)

        dict_['shuffle_segmentated']['train_mean'] = np.mean(dict_['shuffle_segmentated']['train'])
        dict_['shuffle_segmentated']['train_std']  = np.std(dict_['shuffle_segmentated']['train'])

        dict_['shuffle_segmentated']['val_mean']   = np.mean(dict_['shuffle_segmentated']['val'])
        dict_['shuffle_segmentated']['val_std']    = np.std(dict_['shuffle_segmentated']['val'])




        path = './models/variance/'
        pickle_save_(path+'variance.p',dict_)
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

        plt.close('all')












if __name__ == '__main__':
    if __name__ == '__main__':
        dict_c, bounds = return_dict_bounds()

        mm = model_tests(dict_c)
        mm.variance_calculation()