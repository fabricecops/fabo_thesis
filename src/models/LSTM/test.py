from src.models.LSTM.configure import return_dict_bounds
from src.models.LSTM.main_LSTM import model_mng
from src.dst.outputhandler.pickle import tic,toc,pickle_save_
import numpy as np
class model_tests():


    def __init__(self,dict_c):
        self.dict_c    = dict_c
        self.iteration = 2


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

        dict_['shuffle_random']['val_mean'] = np.mean(dict_['shuffle_random']['val'])
        dict_['shuffle_random']['val_std']  = np.std(dict_['shuffle_random']['val'])

        self.dict_c['path_save']     = './models/variance/shuffle_segmentated/'
        self.dict_c['shuffle_style'] = 'testing'
        ##### normal shuffle
        for i in range(self.iteration):

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




        path = './models/variance/variance.p'
        pickle_save_(path,dict_)
        print(dict_)













if __name__ == '__main__':
    if __name__ == '__main__':
        dict_c, bounds = return_dict_bounds()

        mm = model_tests(dict_c)
        mm.variance_calculation()