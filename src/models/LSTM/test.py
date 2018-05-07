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

        dict_c['time_dim']   = int(x[:,0])
        dict_c['lr']         = float(x[:,1])
        dict_c['vector']     =int(x[:,2])

        array_encoder = []
        for i in range(3):
            array_encoder.append(int(x[:, 3 + i]))

        array_decoder = []
        for i in range(2):
            array_decoder.append(int(x[:, 6+i]))

        dict_c['encoder'] = array_encoder
        dict_c['decoder'] = array_decoder

        array_features = []
        if (int(x[:, -1]) == 1):
            array_features.append('PCA')

        if (int(x[:, -2]) == 1):
            array_features.append('p')
        if (int(x[:, -2]) == 2):
            array_features.append('v')

        if (array_features == []):
            array_features.append('p')
        array_features = list(set(array_features))

        dict_c['mode_data'] = array_features

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




if __name__ == '__main__':

    # mm = model_tests(dict_c)
    # mm._get_rest_of_data()
    # mm.variance_calculation()

    # mode = 'DEEP1'
    # dict_c,bounds = return_dict_bounds(bounds = mode)
    # dict_c['path_save'] = './models/bayes_opt/DEEP1/'
    # bo = BayesionOpt(dict_c,bounds,mode)
    # bo.main()




    # mode = 'DEEP3'
    # dict_c,bounds = return_dict_bounds(bounds = mode)
    # dict_c['path_save'] = './models/bayes_opt/DEEP3/'
    # bo = BayesionOpt(dict_c,bounds,mode)
    # bo.main()

    # mode = 'DEEP2'
    # dict_c, bounds = return_dict_bounds(bounds=mode)
    # dict_c['test_class'] = ['object']
    # dict_c['path_save'] = './models/test_shuffle/object/DEEP2/'
    # bo = BayesionOpt(dict_c, bounds, mode)
    # bo.main()

    mode = 'DEEP2'
    dict_c, bounds = return_dict_bounds(bounds=mode)
    dict_c['test_class'] = ['boven']
    dict_c['path_save'] = './models/test_shuffle/above/DEEP2/'
    bo = BayesionOpt(dict_c, bounds, mode)
    bo.main()
