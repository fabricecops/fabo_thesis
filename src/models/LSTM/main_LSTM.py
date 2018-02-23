
import os
from src.models.LSTM.s2s import LSTM_
from src.dst.outputhandler.OPS import OPS
from src.dst.optimizers.CMA_ES.unit import unit
from src.dst.optimizers.CMA_ES.CMA_ES import CMA_ES

from src.models.LSTM.configure import return_dict_bounds
from pathos.helpers import mp
import time

import threading
import queue
class model_mng():

    def __init__(self,dict_c):

        self.dict_c = dict_c
        self.model  = LSTM_(dict_c)
        self.OPS    = OPS(self.dict_c)


        self.dict_data = None
        self.Queue_o   = queue.Queue()

    def main(self):
        for i in range(self.dict_c['epochs']):

            dict_data,model_s    = self.process_LSTM()

            if(i != 0):
                while p.is_alive():
                    time.sleep(0.1)

                p.terminate()

            p = mp.Process(target=self.process_output, args= (i,dict_data,model_s))
            p.daemon = False
            p.start()



    def process_LSTM(self):

        loss                = self.model.fit()
        dict_data,model_s   = self.model.predict()
        dict_data['losses'] = loss

        return dict_data,model_s

    def process_output(self,i,dict_data,model_s):

            self.OPS._save_output(dict_data)

            CMA_ES_    = CMA_ES(self.dict_c)
            dict_data2 = CMA_ES_.main_CMA_ES(dict_data,i)
            dict_data2['path_o'] = dict_data['path_o']

            self.OPS.main_OPS(dict_data2, i,model_s)




if __name__ == '__main__':
    dict_c, bounds = return_dict_bounds()

    mm    = model_mng(dict_c).main()




    ##### train_BO_different_windows ####
    # dict_c, bounds = return_dict_bounds()
    # windows        = [1,5,10,15,20,25,30,35,40]
    # mm.BO_LSTM(dict_c,bounds,windows)