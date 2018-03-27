from src.models.LSTM.configure import return_dict_bounds
from src.models.LSTM.main_LSTM import model_mng
from src.dst.outputhandler.pickle import tic,toc,pickle_save_,pickle_load
import numpy as np
import matplotlib.pyplot as plt
import os

import GPyOpt

class model_tests():


    def __init__(self,dict_c):
        self.dict_c    = dict_c
        self.iteration = 20
        self.path      = './models/variance/'

        self.testing   = False


    def variance_calculation(self):
        dict_ = pickle_load('./models/variance/variance.p', None)
        # dict_   = self._return_dict_()

        for i in range(0):
            self.dict_c['random_state']  = (i+13)*50
            self.dict_c['path_save']     = './models/variance/shuffle_segmentated/'
            self.dict_c['shuffle_style'] = 'segmentated'
            dict_ = self._train_model(dict_=dict_)

        dict_['shuffle_segmentated']['train_mean'] = np.mean(dict_['shuffle_segmentated']['train'])
        dict_['shuffle_segmentated']['train_std'] = np.std(dict_['shuffle_segmentated']['train'])

        dict_['shuffle_segmentated']['val_mean'] = np.mean(dict_['shuffle_segmentated']['val'])
        dict_['shuffle_segmentated']['val_std'] = np.std(dict_['shuffle_segmentated']['val'])


        for i in range(0):
            self.dict_c['random_state']  = i*50
            self.dict_c['shuffle_style'] = 'random'
            self.dict_c['path_save']     = './models/variance/shuffle_random/'

            dict_ = self._train_model(dict_ = dict_)


        dict_['shuffle_random']['train_mean'] = np.mean(dict_['shuffle_random']['train'])
        dict_['shuffle_random']['train_std']  = np.std(dict_['shuffle_random']['train'])

        dict_['shuffle_random']['val_mean']   = np.mean(dict_['shuffle_random']['val'])
        dict_['shuffle_random']['val_std']    = np.std(dict_['shuffle_random']['val'])

        for i in range(13):
            self.dict_c['random_state']  = 50
            self.dict_c['shuffle_style'] = 'segmentated'
            self.dict_c['path_save']     = './models/variance/no_shuffle/'

            dict_ = self._train_model(dict_=dict_)

        dict_['no_shuffle']['train_mean'] = np.mean(dict_['no_shuffle']['train'])
        dict_['no_shuffle']['train_std'] = np.std(dict_['no_shuffle']['train'])

        dict_['no_shuffle']['val_mean'] = np.mean(dict_['no_shuffle']['val'])
        dict_['no_shuffle']['val_std'] = np.std(dict_['no_shuffle']['val'])





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


    def _plot_results(self,dict_,CMA = 'cma'):
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


        plt.savefig(self.path+'distribution'+CMA+'.png')
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

    def _get_rest_of_data(self):
        dict_ = self._return_dict_()

        path = 'models/variance/shuffle_random/'
        dir_ = os.listdir(path)

        for directory in dir_:
            path_v = path+directory+'/hist.p'
            hist   = pickle_load(path_v,None)
            AUC_tr = np.max(hist['AUC'])
            AUC_v  = np.max(hist['AUC_v'])
            AUC_t  = np.max(hist['AUC_t'])


            dict_['shuffle_random']['train'].append(AUC_tr)
            dict_['shuffle_random']['val'].append(AUC_v)
            dict_['shuffle_random']['val'].append(AUC_t)



        path = 'models/variance/shuffle_segmentated/'
        dir_ = os.listdir(path)
        for directory in dir_:
            path_v = path+directory+'/hist.p'
            hist   = pickle_load(path_v,None)
            AUC_tr = np.max(hist['AUC'])
            AUC_v  = np.max(hist['AUC_v'])
            AUC_t  = np.max(hist['AUC_t'])



            dict_['shuffle_segmentated']['train'].append(AUC_tr)
            dict_['shuffle_segmentated']['val'].append(AUC_v)
            dict_['shuffle_segmentated']['val'].append(AUC_t)


            self._plot_results(dict_,CMA = 'no_cma')


        return dict_



class BayesionOpt():

    def __init__(self,bounds,dict_c):
        self.bounds     = bounds
        self.dict_c     = dict_c

        self.len_space  = 30

    #### public functions   ##############

    def main(self):

        train = GPyOpt.methods.BayesianOptimization(f                      = self.opt_function,
                                                    domain                 = self.bounds,
                                                    maximize               = self.dict_c['maximize'],
                                                    initial_design_numdata = self.dict_c['initial_n'],
                                                    initial_design_type    = self.dict_c['initial_dt'],
                                                    eps                    = self.dict_c['eps']

                                                    )


        train.run_optimization(max_iter=self.dict_c['max_iter'])

        print("optimized parameters: {0}".format(train.x_opt))
        print("optimized loss: {0}".format(train.fx_opt))

    #### private functions   ################

    def opt_function(self,x):

            mm = model_mng(self.dict_c)
            AUC_tr, AUC_v, AUC_t = mm.main(mm.Queue_cma)

            return AUC_v

    def configure_bounds(self,x):

        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print()

        array_filters = []

        for i,bound in enumerate(self.bounds):


            key   = bound['name']
            type_ = bound['type']

            boolean       = True

            if('filter' not in key):

                if(type_ == 'continuous'):
                    self.dict_p[key] = float(x[:,i])

                elif(type_ == 'discrete'):
                    self.dict_p[key] = int(x[:,i])


                len_space = self.len_space-len(key)
                string    = key+' '*len_space+': '
                print(string+str(self.dict_p[key]))

            else:

                if('filter_o' in key):


                    if(x[:,i]==0):
                        boolean = False

                    if(boolean == True):
                        tuple_ = (int(x[:,i]),
                                  int(x[:,i+1]),
                                  int(x[:,i+1]),
                                  int(x[:,i+2]),
                                  'relu')

                        len_space = self.len_space - len('filter')
                        string = 'filter' + ' ' * len_space + ': ',tuple_
                        print(string)

                        array_filters.append(tuple_)

        self.dict_p['filters'] = array_filters

        print()
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')



if __name__ == '__main__':
    if __name__ == '__main__':
        dict_c, bounds = return_dict_bounds()

        # mm = model_tests(dict_c)
        # mm._get_rest_of_data()
        # mm.variance_calculation()

        bo = BayesionOpt(dict_c,bounds)