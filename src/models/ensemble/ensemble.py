from src.models.ensemble.data_manager import data_manager
from src.models.ensemble.config import return_dict
from src.models.ensemble.CMA_ES import CMA_ES

from src.dst.keras_model.FS_manager import FS_manager
from src.models.LSTM.OPS import OPS_LSTM

class ensemble(data_manager,FS_manager):

    def __init__(self):

        FS_manager.__init__(self,dict_c['path_o'])
        data_manager.__init__(self,dict_c)


    def main(self,dict_c):

        data = self.configure_data(dict_c)
        data = self.run_CMA_ES(dict_c,data)

        OPS_LSTM_ = OPS_LSTM(dict_c)
        OPS_LSTM_.save_output_CMA(data)
        OPS_LSTM_.save_ROC_segment(data, 'segmentation')
        OPS_LSTM_.save_ROC_segment(data, 'location')








    def mean(self):
        pass

    def run_CMA_ES(self,dict_c,data):
        cma  = CMA_ES(dict_c)
        data = cma.main(data)

        return data
def return_dict():

    dict_c  = {
        'path_i' : './models/variance/no_shuffle/',
        'path_o' : './models/ensemble/ensemble/',

        'path_ensemble': {'DEEP1': './models/ensemble/DEEP1/hidden/',
                          'DEEP2': './models/ensemble/DEEP2/hidden/',
                          'DEEP3': './models/ensemble/DEEP3/hidden/'
                          },

        'resolution_AUC': 1000,

        ###### CMA_ES    ######
        'CMA_ES'         : True,
        'verbose_CMA'    : 1,
        'verbose_CMA_log': 0,
        'evals'          : 10000,
        'bounds'         : [-100, 100.],
        'sigma'          : 0.4222222222222225,
        'progress_ST'    : 0.3,


        'epoch'          : 0


    }


    return dict_c
if __name__ == '__main__':

    dict_c = return_dict()

    es = ensemble()
    es.main(dict_c)






