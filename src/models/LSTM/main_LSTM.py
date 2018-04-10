
from src.models.LSTM.model_a.s2s import LSTM_
from src.models.LSTM.OPS_LSTM import OPS_LSTM
from src.models.LSTM.conf_LSTM import return_dict_bounds
from keras import backend as K
import matplotlib.pyplot as plt
import psutil
import objgraph
import numpy as np
import gc
from src.dst.outputhandler.pickle import pickle_save_,pickle_load
import time

def tic():
    global time_
    time_ = time.time()

def toc():
    global time_
    tmp = time.time()

    elapsed = tmp - time_

    print('the elapsed time is: ', elapsed)

    return elapsed

def tic_SB():
    global time__
    time__ = time.time()

def toc_SB():
    global time__
    tmp = time.time()

    elapsed = tmp - time__

    print('the elapsed time is: ', elapsed)
class model_mng():

    def __init__(self,dict_c):


        self.dict_c      = dict_c
        self.model       = LSTM_(dict_c)

        self.dict_data   = None

        self.count_no_cma    = 0
        self.count_no_cma_AUC = 0.


        self.max_AUC_val = 0
        self.max_AUC_tr  = 0
        self.max_AUC_t   = 0

        self.min_val_loss = 10

        self.AUC_no_cma  = 0.0


        self.state_loss  = 0.

        self.path_o      = self.model.return_path()
        self.epoch       = 0.


        self.memory      = []
        self.obj_graph   = {}

        lol = objgraph.most_common_types(limit=20)
        self.iter_memwatcher = 0
        for obj in lol:
            self.obj_graph[obj[0]] = np.array([])


        self.best_dict   = {}

        self.dict_loss   = {
                            'loss'    : [],
                            'val_loss': []
        }


    def main(self,queue_opt):

        for i in range(self.dict_c['epochs']):


            tic()
            loss, val_loss = self.process_before_TH(i)
            self.memory.append(psutil.virtual_memory()[3] / 1000000000.)

            if(self.min_val_loss < self.dict_c['TH_val_loss'] and i%self.dict_c['mod_data'] == 0):
                self.process_LSTM(i,loss,val_loss)

            self.plot_obj()
            plt.close('all')
            gc.collect()
            toc()


            if (self.count_no_cma > self.dict_c['SI_no_cma'] or self.count_no_cma_AUC > self.dict_c['SI_no_cma_AUC']):
                epoch = i
                break




        self.process_LSTM(epoch, loss, val_loss)
        self.plot_output(self.best_dict)
        self.best_dict.clear()
        K.clear_session()
        gc.collect()

        queue_opt.put(self.AUC_no_cma)



    def process_before_TH(self,i):
        loss, val_loss = self.model.fit()
        self.dict_loss['loss'].append(loss[0])
        self.dict_loss['val_loss'].append(val_loss[0])
        self.plot(self.dict_loss)
        pickle_save_(self.path_o+'hist_no_cma.p',self.dict_loss)

        if (val_loss[0] >= self.min_val_loss):
            self.count_no_cma += 1
        else:
            self.count_no_cma = 0
            self.min_val_loss = val_loss[0]

        if (self.dict_c['TH_val_loss'] <= self.min_val_loss or i%self.dict_c['mod_data'] == 0):

            print(loss, val_loss,self.count_no_cma )
            print()


        return loss,val_loss


    def process_LSTM(self,i,loss,val_loss):
        self.memory.append(psutil.virtual_memory()[3] / 1000000000.)

        dict_data                = self.model.predict()

        dict_data['loss_f_tr']   = loss[0]
        dict_data['loss_f_v']    = val_loss[0]
        dict_data['epoch']       = i


        OPS_LSTM_                = OPS_LSTM(self.dict_c)
        dict_data                = OPS_LSTM_.main(dict_data)
        self.update_states(dict_data)
        self.memory.append(psutil.virtual_memory()[3] / 1000000000.)





    def update_states(self,dict_data):
        self.print_stats(dict_data)


        if (dict_data['AUC_v']  > self.AUC_no_cma):
            self.AUC_no_cma = dict_data['AUC_v']
            self.count_no_cma_AUC = 0
            self.best_dict        = dict_data
            self.best_dict['path_o'] = self.model.return_path()
        else:
            self.count_no_cma_AUC += 1

    def print_stats(self,dict_data):
        print('epoch      : ' +str(dict_data['epoch']))
        print('SI_val_loss: '+str(self.count_no_cma)+'/'+str(self.dict_c['SI_no_cma'])+'    SI_AUC: '+str(self.count_no_cma_AUC)+'/'+str(self.dict_c['SI_no_cma_AUC']))
        print('Loss       : '+str(round(dict_data['train_f'],3))+'    Val_loss: '+str(round(dict_data['val_f'],3)))
        print('AUC_tr     : '+str(round(dict_data['AUC'],3))+'        AUC_v     : '+str(round(dict_data['AUC_v'],3))+'    Best AUC:' + str(round(self.AUC_no_cma,3)))
        print()

    def plot_output(self,best_dict):
        OPS_LSTM_                = OPS_LSTM(self.dict_c)
        OPS_LSTM_.save_plots_no_cma(best_dict)

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


        plt.plot(self.memory)
        plt.title('memory')
        plt.savefig(self.path_o + 'memory.png')

    def plot_obj(self):
        most_common_types_step = objgraph.most_common_types(limit=20)

        for key, value in most_common_types_step:
            if key not in self.obj_graph.keys():
                self.obj_graph[key] = np.zeros(self.iter_memwatcher)

            self.obj_graph[key] = np.hstack((self.obj_graph[key],[value]))

        for key in self.obj_graph.keys():
            if key not in np.array(most_common_types_step)[:,0]:
                self.obj_graph[key] = np.hstack((self.obj_graph[key],[0]))



        fig = plt.figure(figsize=(16, 4))

        ax1 = plt.subplot(121)

        sum = np.zeros(self.iter_memwatcher + 1)

        for key in self.obj_graph.keys():
            ax1.plot(self.obj_graph[key],label = key)
            sum += self.obj_graph[key]

        ax1.plot(sum, label='sum')
        plt.title('object graph')
        plt.legend()

        # plot total mem
        ax2 = plt.subplot(122)

        ax2.plot(self.memory)
        plt.title('memory')


        plt.savefig(self.path_o + 'objgraph.png')

        self.iter_memwatcher += 1

    def plot(self,dict_):

        fig = plt.figure(figsize=(16, 4))

        plt.plot(dict_['val_loss'],label = 'val_loss')
        plt.plot(dict_['loss'], label = 'loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('validation curve')
        plt.legend()
        plt.savefig(self.path_o+'val_curve_TH.png')



if __name__ == '__main__':

    dict_c, bounds = return_dict_bounds()
    mm    = model_mng(dict_c)
    mm.main()

