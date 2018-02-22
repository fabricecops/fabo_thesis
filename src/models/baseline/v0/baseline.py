from src.models.LSTM.data_manager import data_manager
from src.dst.optimizers.CMA_ES.CMA_ES import CMA_ES
from src.models.baseline.v0.configure import return_dict_bounds
import pandas as pd
import numpy as np
import functools
import cma



class baseline(CMA_ES,data_manager):


    def __init__(self, dict_c=None, path=None):

        data_manager.__init__(self, dict_c)
        CMA_ES.__init__(self,dict_c)

        self.df_true  = None
        self.df_false = None

        self.AUC_max  = None
        self.AUC_min  = None
        self.FPR      = None
        self.TPR      = None

        self.df_f,self.df_f_val,self.df_t = self.return_df()
        self.configure_data()



    def main(self):
        dimension      = self.df_true.iloc[0]['data_X'].shape[2]

        es = cma.fmin(self._opt_function,
                      np.zeros(dimension),
                      self.sigma,
                      {'bounds'              : self.bounds,
                      'maxfevals'            : self.evals,
                       'verb_disp'           : self.verbose,
                       'verb_log'             :self.verbose_log })

        return es[0],-es[1]

    def _opt_function(self, x):

        eval_true  = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_true['data_m']))))
        eval_false = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_false['data_m']))))

        AUC, FPR, TPR = self.get_AUC_score(eval_true, eval_false)
        if (AUC > self.AUC_max):
            self.AUC_max = AUC
            self.FPR = FPR
            self.TPR = TPR

        if (AUC < self.AUC_min):
            self.AUC_min = AUC

        return -AUC



    def _get_error_m(self, row):

        e_f = np.mean(row, axis = 1)

        return e_f



    def _get_error_max(self,e,x):

        eval_ = np.max(np.dot(e, x))
        return eval_




    def configure_data(self):


        self.df_f['error_m']         = list(map(self._get_error_m, np.array(self.df_f['data_X'])))
        self.df_f_val['error_m']     = list(map(self._get_error_m, np.array(self.df_f_val['data_X'])))
        self.df_t['error_m']         = list(map(self._get_error_m, np.array(self.df_t['data_X'])))


        self.df_f     = self.df_f['error_m']
        self.df_f_val = self.df_f_val['error_m']
        self.df_t     = self.df_t['error_m']

if __name__ == '__main__':

    dict_c, _ = return_dict_bounds()

    import os


    baseline  = baseline(dict_c)






