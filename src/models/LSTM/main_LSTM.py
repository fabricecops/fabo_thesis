
from src.models.LSTM.model_a.s2s import LSTM_
from src.models.LSTM.OPS_LSTM import OPS_LSTM
from src.models.LSTM.conf_LSTM import return_dict_bounds
from keras import backend as K
import matplotlib.pyplot as plt
import time
class model_mng():

    def __init__(self,dict_c):


        self.dict_c      = dict_c

        self.OPS_LSTM_                = OPS_LSTM(self.dict_c)
        self.dict_c['shuffle_style']  = 'testing'
        self.model       = LSTM_(dict_c)

        self.dict_data   = None

        self.count_no_cma    = 0
        self.count_no_cma_AUC = 0.


        self.max_AUC_val = 0
        self.max_AUC_tr  = 0
        self.max_AUC_t   = 0

        self.min_val_loss = 10

        self.AUC_no_cma  = 0


        self.state_loss  = 0.

    def main(self):

        for i in range(self.dict_c['epochs']):
            self.process_LSTM(i)

            if(self.count_no_cma > self.dict_c['SI_no_cma'] or self.count_no_cma_AUC > self.dict_c['SI_no_cma_AUC']):
                break
            print(self.dict_c['SI_no_cma'],self.dict_c['SI_no_cma_AUC'],self.dict_c['epochs'])
            print(self.count_no_cma,self.count_no_cma_AUC,i)
            print(plt.get_fignums())

        K.clear_session()

        time.sleep(5)

        return self.AUC_no_cma,self.min_val_loss


    def process_LSTM(self,i):

        loss,val_loss            = self.model.fit()
        dict_data                = self.model.predict()

        dict_data['loss_f_tr']   = loss[0]
        dict_data['loss_f_v']    = val_loss[0]
        dict_data['epoch']       = i

        AUC_v                    = self.OPS_LSTM_.main(dict_data)


        if(AUC_v > self.AUC_no_cma):
            self.AUC_no_cma    = AUC_v
            self.count_no_cma_AUC  = 0
        else:
            self.count_no_cma_AUC  += 1




        if(val_loss[0] >= self.min_val_loss):
            self.count_no_cma += 1
        else:
            self.count_no_cma  = 0
            self.min_val_loss  = val_loss[0]


        return dict_data









if __name__ == '__main__':
    dict_c, bounds = return_dict_bounds()
    mm    = model_mng(dict_c)
    mm.main()

