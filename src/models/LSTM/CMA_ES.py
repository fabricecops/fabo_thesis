from src.dst.metrics.AUC import AUC
import numpy as np
import functools
import cma
import pandas as pd
###
class CMA_ES(AUC):


    def __init__(self, dict_c):
        AUC.__init__(self,dict_c)
        self.dict_c    = dict_c

        self.verbose    = dict_c['verbose_CMA']
        self.verbose_log= dict_c['verbose_CMA_log']

        self.evals     = dict_c['evals']
        self.bounds    = dict_c['bounds']
        self.sigma     = dict_c['sigma']

        self.AUC_max  = -5
        self.AUC_min  = 10
        self.FPR      = None
        self.TPR      = None




    def main_CMA_ES(self,dict_data):
        self.configure_data(dict_data)


        dimension = self.df_f_train['error_m'].iloc[0].shape[1]
        array = np.zeros(dimension)


        es = cma.fmin(self._opt_function,
                      array,
                      self.sigma,
                      {'bounds': self.bounds,
                       'maxfevals': self.evals,
                       'verb_disp': self.verbose,
                       'verb_log': self.verbose_log})

        AUC, FPR, TPR       = self.AUC_max, self.FPR, self.TPR
        AUC_v, FPR_v, TPR_v = self._opt_function_(es[0], self.df_f_val['error_m'], self.df_t_val['error_m'])
        AUC_t, FPR_t, TPR_t = self._opt_function_(es[0], self.df_f_test['error_m'], self.df_t_test['error_m'])


        self.df_f_train['error_m'] =  list(map(functools.partial(self._get_error_max, x=es[0]), np.array(self.df_f_train['error_m'])))
        self.df_f_val['error_m']   =  list(map(functools.partial(self._get_error_max, x=es[0]), np.array(self.df_f_val['error_m'])))
        self.df_f_test['error_m']  =  list(map(functools.partial(self._get_error_max, x=es[0]), np.array(self.df_f_test['error_m'])))

        self.df_t_train['error_m'] =  list(map(functools.partial(self._get_error_max, x=es[0]), np.array(self.df_t_train['error_m'])))
        self.df_t_val['error_m']   =  list(map(functools.partial(self._get_error_max, x=es[0]), np.array(self.df_t_val['error_m'])))
        self.df_t_test['error_m']  =  list(map(functools.partial(self._get_error_max, x=es[0]), np.array(self.df_t_test['error_m'])))


        dict_ = {
            'x'      : es[0],
            'AUC'    : -es[1],
            'AUC_v'  : AUC_v,
            'AUC_t'  : AUC_t,

            'FPR'    : FPR,
            'TPR'    : TPR,
            'FPR_v'  : FPR_v,
            'TPR_v'  : TPR_v,
            'FPR_t'  : FPR_t,
            'TPR_t'  : TPR_t,

            'df_f_train': self.df_f_train,
            'df_f_val'  : self.df_f_val,
            'df_f_test' : self.df_f_test,

            'df_t_train' : self.df_t_train,
            'df_t_val'   : self.df_t_val,
            'df_t_test'  : self.df_t_test,

            'path_o'     : dict_data['path_o'],
            'epoch'      : dict_data['epoch']

        }



        return dict_



    def _opt_function(self, x):

        eval_true = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_t_train['error_m']))))
        eval_false = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_f_train['error_m']))))

        AUC, FPR, TPR = self.get_AUC_score(eval_true, eval_false)
        if (AUC > self.AUC_max):
            self.AUC_max = AUC
            self.FPR = FPR
            self.TPR = TPR

        if (AUC < self.AUC_min):
            self.AUC_min = AUC

        return -AUC

    def _opt_function_(self, x, f, t):

        eval_true = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(t))))
        eval_false = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(f))))

        AUC, FPR, TPR = self.get_AUC_score(eval_true, eval_false)

        return AUC, FPR, TPR


    def _get_error_m(self, row):

        y     = row[0]
        y_p   = row[1]


        e_f = np.mean(np.power((y - y_p),2),axis=1)


        return e_f

    def _get_error_mean(self, row):

        y     = row[0]
        y_p   = row[1]

        e_f = np.mean(np.power((y - y_p), 2))



        return e_f

    def _get_error_max(self,e,x):

        eval_ = np.max(np.dot(e, x))


        return eval_

    def configure_data(self,dict_data):
        self.df_t_train      = dict_data['df_t_train']
        self.df_t_val        = dict_data['df_t_val']
        self.df_t_test       = dict_data['df_t_test']

        self.df_f_train      = dict_data['df_f_train']
        self.df_f_val        = dict_data['df_f_val']
        self.df_f_test       = dict_data['df_f_test']

        self.dimension    = self.df_t_train.iloc[0]['data_X'].shape[2]

        array_t                        = zip(list(self.df_t_train['data_y']),list(self.df_t_train['data_y_p']))
        self.df_t_train['error_m']     = list(map(self._get_error_m,array_t))
        self.df_t_train.drop(['data_X','data_y'],axis = 1,inplace = True)

        array_t                        = zip(list(self.df_t_val['data_y']),list(self.df_t_val['data_y_p']))
        self.df_t_val['error_m']       = list(map(self._get_error_m,array_t))
        self.df_t_val.drop(['data_X','data_y'],axis = 1,inplace = True)

        array_t                        = zip(list(self.df_t_test['data_y']),list(self.df_t_test['data_y_p']))
        self.df_t_test['error_m']      = list(map(self._get_error_m,array_t))
        self.df_t_test.drop(['data_X','data_y'],axis = 1,inplace = True)

        array_f                        = zip(list(self.df_f_train['data_y']),list(self.df_f_train['data_y_p']))
        self.df_f_train['error_m']     = list(map(self._get_error_m,array_f))
        self.df_f_train.drop(['data_X','data_y'],axis = 1,inplace = True)

        array_f                        = zip(list(self.df_f_val['data_y']),list(self.df_f_val['data_y_p']))
        self.df_f_val['error_m']       = list(map(self._get_error_m,array_f))
        self.df_f_val.drop(['data_X','data_y'],axis = 1,inplace = True)

        array_f                        = zip(list(self.df_f_test['data_y']),list(self.df_f_test['data_y_p']))
        self.df_f_test['error_m']      = list(map(self._get_error_m,array_f))
        self.df_f_test.drop(['data_X','data_y'],axis = 1,inplace = True)

