from src.models.spatio_temporal.conf_ST import return_dict,return_bounds
from src.models.spatio_temporal.main_ST import model_mng
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

    def __init__(self,dict_c,bounds):
        self.bounds     = bounds
        self.dict_c     = dict_c

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

        dict_c = self.configure_bounds_LSTM1_DEEP1(self.dict_c,x)

        self.print_parameters(dict_c)
        time_1 = time.time()

        opt_value = self.opt_function_SB()

        time_2  = time.time()-time_1
        self.time_.append(time_2)
        self.AUC_v.append(opt_value)
        self.plot(self.time_,self.AUC_v)
        self.save_memory()



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
        try:
            mm = model_mng(self.dict_c)
            mm.main(queue_opt)
        except:

            queue_opt.put(0.5)


    def configure_bounds_LSTM1_DEEP1(self,dict_c,x):

        dict_c['lr']         = float(x[:,0])
        dict_c['time_dim']   = int(x[:,1])

        dict_c['conv_encoder'] = [(int(x[:,2]),int(x[:,3]),int(x[:,4])),(int(x[:,5]),int(x[:,6]),int(x[:,7]))]
        dict_c['conv_LSTM']    = [(int(x[:,8]),int(x[:,9]))]
        dict_c['middel_LSTM']  = (int(x[:,10]),int(x[:,11]))



        return dict_c



    def print_parameters(self,dict_c):
        print('x'*50)
        print('x'*50)
        print('x'*50)
        print( dict_c['conv_encoder'])
        print(  dict_c['conv_LSTM'])
        print( dict_c['middel_LSTM'])

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


    dict_c = return_dict()
    bounds = return_bounds()
    dict_c['path_save'] = './models/test_shuffle/sneaky/SP/'
    bo = BayesionOpt(dict_c,bounds)
    bo.main()


