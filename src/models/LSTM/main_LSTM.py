
import os
from src.models.LSTM.model_a.s2s import LSTM_
from src.models.LSTM.OPS import OPS_LSTM
from src.models.LSTM.CMA_ES import CMA_ES

from src.models.LSTM.configure import return_dict_bounds
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

class model_mng():

    def __init__(self,dict_c):


        self.dict_c      = dict_c
        self.model       = LSTM_(dict_c)

        self.dict_data   = None
        self.Queue_cma   = mp.Queue()

        self.count_cma    = 0
        self.count_no_cma = 0
        self.count_no_cma_AUC = 0.


        self.max_AUC_val = 0
        self.max_AUC_tr  = 0
        self.max_AUC_t   = 0

        self.AUC_no_cma  = 0


        self.state_loss  = 0.

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

            print('x'*50)
            print('x'*50)
            print(self.count_cma ,self.count_no_cma,self.count_no_cma_AUC)
            print('x'*50)
            print('x'*50)

            if(self.count_cma > self.dict_c['SI_cma'] or
               self.count_no_cma > self.dict_c['SI_no_cma'] or
               self.count_no_cma_AUC > self.dict_c['SI_no_cma_AUC']):
                break
            plt.close('all')

        p.terminate()

        while p.is_alive():
            time.sleep(1)

        p.terminate()


        return self.max_AUC_tr,self.max_AUC_val,self.max_AUC_val

    def process_Queue(self,Queue_cma):
        if (Queue_cma.empty() == False):
            dict_ = self.Queue_cma.get()
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
        AUC_v = OPS_LSTM_.main(dict_data)




        if(loss[0] > self.dict_c['TH_loss']):
            self.count_no_cma += 1
        else:
            self.count_no_cma  = 0



        if(AUC_v > self.AUC_no_cma):
            self.AUC_no_cma    = AUC_v
            self.count_no_cma_AUC  = 0
        else:
            self.count_no_cma_AUC  += 1



        return dict_data

    def process_output(self,Queue_cma,dict_data):

        CMA_ES_    = CMA_ES(self.dict_c)
        dict_      = CMA_ES_.main_CMA_ES(dict_data)
        Queue_cma.put(dict_)












if __name__ == '__main__':
    dict_c, bounds = return_dict_bounds()

    mm    = model_mng(dict_c)
    mm.main(mm.Queue_cma)

