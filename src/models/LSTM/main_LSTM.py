
import os
from src.models.LSTM.model_a.s2s import LSTM_
from src.models.LSTM.OPS import OPS_LSTM
from src.models.LSTM.CMA_ES import CMA_ES

from src.models.LSTM.configure import return_dict_bounds
import multiprocessing as mp

import time

class model_mng():

    def __init__(self,dict_c):


        self.dict_c      = dict_c
        self.model       = LSTM_(dict_c)

        self.dict_data   = None
        self.Queue_cma   = mp.Queue()

        self.count       = 0
        self.max_AUC_val = 0
        self.max_AUC_tr  = 0
        self.max_AUC_t   = 0

        self.memory_error=False

    def main(self,Queue_cma):

        for i in range(self.dict_c['epochs']):

            dict_data    = self.process_LSTM(i)
            self.process_Queue(Queue_cma)

            if(i==0):
                p        = mp.Process(target=self.process_output, args=(Queue_cma,dict_data))
                p.daemon = True
                p.start()

            else:
                if p.is_alive() == False:
                    p.terminate()

                    p        = mp.Process(target=self.process_output, args= (Queue_cma,dict_data))
                    p.daemon = True
                    p.start()

            if(self.count > self.dict_c['stop_iterations'] or self.memory_error == True):
                p.terminate()
                break

        p.terminate()

        if(self.memory_error == True):
            self.max_AUC_val = 0.5

        return self.max_AUC_tr,self.max_AUC_val,self.max_AUC_val

    def process_Queue(self,Queue_cma):
        if (Queue_cma.empty() == False):
            dict_ = self.Queue_cma.get()
            if(dict_ != None):
                OPS_LSTM_ = OPS_LSTM(self.dict_c)
                OPS_LSTM_.save_output_CMA(dict_)
                OPS_LSTM_.save_ROC_segment(dict_,'segmentation')
                OPS_LSTM_.save_ROC_segment(dict_,'location')
                OPS_LSTM_.plot_dist(dict_)

                if(dict_['AUC_v']> self.max_AUC_val):
                    self.max_AUC_val = dict_['AUC_v']
                    self.max_AUC_tr  = dict_['AUC']
                    self.max_AUC_t   = dict_['AUC_t']
                    self.count       = 0
                else:
                    self.count      += 1

            else:
                self.memory_error = True

        # check if AUC is better, plot train/val/test seperate
        # plot together

    def process_LSTM(self,i):

        loss,val_loss            = self.model.fit()

        dict_data               = self.model.predict()
        dict_data['loss_f_tr']  = loss[0]
        dict_data['loss_f_v']   = val_loss[0]
        dict_data['epoch']      = i

        OPS_LSTM_ = OPS_LSTM(self.dict_c)
        dict_data['epoch'] = i
        OPS_LSTM_.main(dict_data)


        return dict_data

    def process_output(self,Queue_cma,dict_data):
            try:

                CMA_ES_    = CMA_ES(self.dict_c)
                dict_      = CMA_ES_.main_CMA_ES(dict_data)
            except Exception as e:
                dict_      = None

                print('x'*50)
                print('x'*50)
                print(e)
                print('x'*50)
                print('x'*50)


            Queue_cma.put(dict_)












if __name__ == '__main__':
    dict_c, bounds = return_dict_bounds()

    mm    = model_mng(dict_c)
    mm.main(mm.Queue_cma)

