
import os
from src.models.LSTM.model_a.s2s import LSTM_
from src.models.LSTM.outputhandler.OPS import OPS_LSTM
from src.models.LSTM.optimizers.CMA_ES import CMA_ES

from src.models.LSTM.configure import return_dict_bounds
import multiprocessing as mp


import queue

class model_mng():

    def __init__(self,dict_c):


        self.dict_c      = dict_c
        self.model       = LSTM_(dict_c)
        self.OPS_LSTM    = OPS_LSTM(self.dict_c)

        self.dict_data = None
        self.Queue_cma = mp.Queue(maxsize=10)

    def main(self,Queue_cma):


        for i in range(self.dict_c['epochs']):
            dict_data    = self.process_LSTM()

            self.OPS_LSTM.save_output(dict_data,i)
            self._conf_FS(dict_data,i)
            self.OPS_LSTM.main(dict_data,i)

            if(Queue_cma.empty() == False):

                df,dict_,path = self.Queue_cma.get()
                self.OPS_LSTM.save_output_CMA(df,dict_,path)

            if(i != 0):
                if p.is_alive() == False:

                    p.terminate()

                    p = mp.Process(target=self.process_output, args= (Queue_cma,dict_data))
                    p.daemon = False
                    p.start()
            if(i==0):
                p = mp.Process(target=self.process_output, args=(Queue_cma,dict_data))
                p.daemon = False
                p.start()




    def process_LSTM(self):

        loss,val_loss            = self.model.fit()

        dict_data               = self.model.predict()
        dict_data['loss_f_tr']  = loss[0]
        dict_data['loss_f_v']   = val_loss[0]


        return dict_data

    def process_output(self,queue,dict_data):


            CMA_ES_    = CMA_ES(self.dict_c)
            tuple_     = CMA_ES_.main_CMA_ES(dict_data)

            queue.put(tuple_)






    def _conf_FS(self,dict_data,epoch):
        string = 'epoch_'+str(epoch)

        dir_   = dict_data['path_o'] + 'predictions/'
        if (os.path.exists(dir_)==False):
            os.mkdir(dir_)
        dir_   = dict_data['path_o'] + 'predictions/'+string
        if (os.path.exists(dir_)==False):
            os.mkdir(dir_)

        dir_   = dict_data['path_o'] + 'predictions/epoch_'+str(epoch)
        path_m = dir_ +'/model.h5'

        dict_data['model'].save(path_m)





if __name__ == '__main__':
    dict_c, bounds = return_dict_bounds()

    mm    = model_mng(dict_c)
    mm.main(mm.Queue_cma)

    ##### train_BO_different_windows ####
    # dict_c, bounds = return_dict_bounds()
    # windows        = [1,5,10,15,20,25,30,35,40]
    # mm.BO_LSTM(dict_c,bounds,windows)