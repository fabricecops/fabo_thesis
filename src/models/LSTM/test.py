from src.models.LSTM.conf_LSTM import return_dict_bounds
from src.models.LSTM.main_LSTM import model_mng
from src.dst.outputhandler.pickle import tic,toc,pickle_save_,pickle_load
import numpy as np
import gc

import objgraph

import matplotlib.pyplot as plt
import os
import GPyOpt
import time
import psutil
import multiprocessing as mp

class model_tests():


    def __init__(self,dict_c):
        self.dict_c    = dict_c
        self.iteration = 20
        self.path      = './models/variance/'

        self.testing   = True

    def variance_calculation(self):
        # dict_ = pickle_load('./models/variance/variance.p', None)
        dict_   = self._return_dict_()

        for i in range(0):
            self.dict_c['random_state']  = (i+13)*50
            self.dict_c['path_save']     = './models/variance/shuffle_segmentated/'
            self.dict_c['shuffle_style'] = 'segmentated'
            dict_ = self._train_model(dict_=dict_)

        dict_['shuffle_segmentated']['train_mean'] = np.mean(dict_['shuffle_segmentated']['train'])
        dict_['shuffle_segmentated']['train_std'] = np.std(dict_['shuffle_segmentated']['train'])

        dict_['shuffle_segmentated']['val_mean'] = np.mean(dict_['shuffle_segmentated']['val'])
        dict_['shuffle_segmentated']['val_std'] = np.std(dict_['shuffle_segmentated']['val'])


        for i in range(2):
            self.dict_c['random_state']  = i*50
            self.dict_c['shuffle_style'] = 'random'
            self.dict_c['path_save']     = './models/variance/shuffle_random/'

            dict_ = self._train_model(dict_ = dict_)


        dict_['shuffle_random']['train_mean'] = np.mean(dict_['shuffle_random']['train'])
        dict_['shuffle_random']['train_std']  = np.std(dict_['shuffle_random']['train'])

        dict_['shuffle_random']['val_mean']   = np.mean(dict_['shuffle_random']['val'])
        dict_['shuffle_random']['val_std']    = np.std(dict_['shuffle_random']['val'])

        for i in range(2):
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
            # self.dict_c['shuffle_style'] = 'testing'
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

    def __init__(self,dict_c,bounds,mode):
        self.bounds     = bounds
        self.dict_c     = dict_c
        self.mode       = mode

        self.time_      = []
        self.AUC_v      = []


        self.memory      = []

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
        if(self.mode == 'DEEP1'):
            dict_c = self.configure_bounds_DEEP1(self.dict_c, x)
        elif(self.mode == 'DEEP2'):
            dict_c = self.configure_bounds_DEEP2(self.dict_c, x)
        else:
            dict_c = self.configure_bounds_DEEP3(self.dict_c, x)



        self.print_parameters(dict_c)
        time_1 = time.time()

        opt_value = self.opt_function_SB()

        time_2  = time.time()-time_1
        self.time_.append(time_2)
        self.AUC_v.append(opt_value)
        self.plot(self.time_,self.AUC_v)
        self.save_memory()


        print('OPT VALUE ' * 3)
        print('OPT VALUE ' * 3)
        print(opt_value)
        print('OPT VALUE ' * 3)
        print('OPT VALUE ' * 3)

        gc.collect()


        return opt_value

    def opt_function_SB(self):
        queue_opt = mp.Queue()

        p = mp.Process(target=self.subprocess, args=(queue_opt,))
        p.daemon = True
        p.start()

        while True:
            if p.is_alive():
                time.sleep(1)
            else:
                opt_value = queue_opt.get()
                p.terminate()
                break

        return opt_value

    def subprocess(self,queue_opt):
        mm = model_mng(self.dict_c)
        mm.main(queue_opt)



    def configure_bounds_DEEP1(self,dict_c,x):

        dict_c['lr']         = float(x[:,0])
        dict_c['time_dim']   = int(x[:,1])
        dict_c['vector']      =int(x[:,2])

        array_encoder = []
        for i in range(1):
            array_encoder.append(int(x[:,3]))

        array_decoder = []



        dict_c['encoder'] = array_encoder
        dict_c['decoder'] = array_decoder

        return dict_c

    def configure_bounds_DEEP2(self,dict_c,x):

        dict_c['time_dim']   = int(x[:,0])
        dict_c['lr']         = float(x[:,1])
        dict_c['vector']     =int(x[:,2])

        array_encoder = []
        for i in range(2):
            array_encoder.append(int(x[:,2+i]))

        array_decoder = []
        for i in range(1):
            array_decoder.append(int(x[:,5]))

        array_features = []
        if(int(x[:,-1]) == 1):
            array_features.append('PCA')

        if (int(x[:, -2]) == 1):
            array_features.append('p')
        if (int(x[:, -2]) == 2):
            array_features.append('v')

        if(array_features == []):
            array_features.append('p')
        array_features = list(set(array_features))

        dict_c['mode_data'] = array_features
        dict_c['encoder']   = array_encoder
        dict_c['decoder']   = array_decoder

        return dict_c

    def configure_bounds_DEEP3(self,dict_c,x):


        dict_c['lr']         = float(x[:,0])
        dict_c['vector']     =int(x[:,1])

        array_encoder = []
        for i in range(3):
            array_encoder.append(int(x[:,2]))

        array_decoder = []
        for i in range(2):
            array_decoder.append(int(x[:,5]))


        dict_c['encoder'] = array_encoder
        dict_c['decoder'] = array_decoder

        return dict_c


    def print_parameters(self,dict_c):
        print('x'*50)
        print('x'*50)
        print('x'*50)
        print()

        len_space = 30
        for element in self.bounds:

            try:
                len_space_t = len_space - len(element['name'])
                string      =   element['name'] + ' ' * len_space_t + ': ', dict_c[element['name']]
                print(string)
            except Exception as e:
                pass
        print(dict_c['encoder'])
        print(dict_c['decoder'])
        print(dict_c['mode_data'])

        print()
        print('x'*50)
        print('x'*50)
        print('x'*50)

    def plot(self,time_,AUC_v):

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)
        ax1.hist(time_,label = 'time')
        plt.xlabel('time')
        plt.ylabel('Amount')
        plt.title('Distribution model fitting time')

        ax2 = plt.subplot(122)
        ax2.hist(AUC_v, label='AUC_v')
        plt.xlabel('AUC_v')
        plt.ylabel('Amount')
        plt.title('Distribution AUC_v')

        plt.savefig(self.dict_c['path_save']+'dist.png')

        dict_data = {'AUC_v' : AUC_v,
                     'time_' : time_}

        pickle_save_(self.dict_c['path_save']+'dict_BO.p',dict_data)

    def save_memory(self):
        fig = plt.figure(figsize=(16, 4))
        # print('x'*50)
        #
        # print(psutil.virtual_memory())
        # print(psutil.virtual_memory()[0])
        # print(psutil.virtual_memory()[1])
        # print(psutil.virtual_memory()[2])
        # print(psutil.virtual_memory()[3]/1000000000.)
        #
        # print('x'*50)
        self.memory.append(psutil.virtual_memory()[3] / 1000000000.)


        plt.plot(self.memory)
        plt.title('memory')
        plt.savefig(self.dict_c['path_save']+'memory.png')


class model_tuning():

    def __init__(self):
        pass


    def main(self,dict_c):

        self.DEEP_1(dict_c)
        self.DEEP_2(dict_c)
        self.DEEP_3(dict_c)

    def DEEP_3(self,dict_c):
        dict_c['path_save'] = './models/ensemble/DEEP3/lr/'
        dict_c['decoder']   = [300,350]
        dict_c['encoder']   = [400,350,300]
        dict_c['vector']    = 400
        dict_c['CMA_ES']    = False

        lr_mapping = self.LR_test(dict_c)
        lr_mapping.sort(key=lambda tup: tup[1])

        dict_c['lr'] = lr_mapping[0][0]

        dict_c['path_save'] = './models/ensemble/DEEP3/hidden/'
        encoder_a = [[350,300,250],[400,350,300]]
        vector_a  = [300]

        for encoder in encoder_a:
            for vector in vector_a:
                dict_c['CMA_ES']  = True
                dict_c['decoder'] = [encoder[1],encoder[0]]
                dict_c['encoder'] = encoder
                dict_c['vector']  = vector
                self.train_model(dict_c)

    def DEEP_2(self,dict_c):
        dict_c['path_save'] = './models/ensemble/DEEP2/lr/'
        dict_c['decoder'] = [350]
        dict_c['encoder'] = [400]
        dict_c['vector'] = 400
        dict_c['CMA_ES'] = False

        lr_mapping = self.LR_test(dict_c)
        lr_mapping.sort(key=lambda tup: tup[1])

        print(lr_mapping)
        dict_c['lr'] = lr_mapping[0][0]

        dict_c['path_save'] = './models/ensemble/DEEP2/hidden/'
        encoder_a = [[250,200]]


        vector_a = [300]

        for encoder in encoder_a:
            for vector in vector_a:
                dict_c['CMA_ES']  = True
                dict_c['decoder'] = [encoder[0]]
                dict_c['encoder'] = encoder
                dict_c['vector']  = vector
                self.train_model(dict_c)

    def DEEP_1(self,dict_c):
            dict_c['path_save'] = './models/ensemble/DEEP1/lr/'
            dict_c['decoder']   = [400]
            dict_c['encoder']   = [350,400]
            dict_c['vector']    = 400
            dict_c['CMA_ES']    = False

            lr_mapping = self.LR_test(dict_c)
            lr_mapping.sort(key=lambda tup: tup[1])

            print(lr_mapping)
            dict_c['lr'] = lr_mapping[0][0]


            dict_c['path_save'] = './models/ensemble/DEEP1/hidden/'
            encoder_a = [200]
            vector_a  = [300]

            for encoder in encoder_a:
                for vector in vector_a:
                    dict_c['CMA_ES']  = True
                    dict_c['decoder'] = []
                    dict_c['encoder'] = [encoder]
                    dict_c['vector']  = vector
                    self.train_model(dict_c)

    def LR_test(self,dict_c):

        lr_a     = [0.1,0.2]
        result_a = []
        for lr in lr_a:

            dict_c['lr'] = lr
            opt_value = self.train_model(dict_c)
            result_a.append((lr,opt_value))

        return result_a

    def train_model(self,dict_c):
        mm = model_mng(dict_c)
        _, opt_value, _ = mm.main(mm.Queue_cma)

        # print()
        # print('OPT VALUE ' * 3)
        # print('OPT VALUE ' * 3)
        # print(opt_value)
        # print('OPT VALUE ' * 3)
        # print('OPT VALUE ' * 3)
        # print()
        return opt_value


if __name__ == '__main__':

    # mm = model_tests(dict_c)
    # mm._get_rest_of_data()
    # mm.variance_calculation()

    # mode = 'DEEP1'
    # dict_c,bounds = return_dict_bounds(bounds = mode)
    # dict_c['path_save'] = './models/bayes_opt/DEEP1/'
    # bo = BayesionOpt(dict_c,bounds,mode)
    # bo.main()


    mode = 'DEEP2'
    dict_c,bounds = return_dict_bounds(bounds = mode)
    dict_c['path_save'] = './models/bayes_opt/DEEP2/'
    bo = BayesionOpt(dict_c,bounds,mode)
    bo.main()

    mode = 'DEEP3'
    dict_c,bounds = return_dict_bounds(bounds = mode)
    dict_c['path_save'] = './models/bayes_opt/DEEP3/'
    bo = BayesionOpt(dict_c,bounds,mode)
    bo.main()


