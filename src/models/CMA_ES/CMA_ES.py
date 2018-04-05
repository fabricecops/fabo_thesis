from src.dst.metrics.AUC import AUC
import numpy as np
import functools
import cma
from src.models.CMA_ES.data_manager_CMA import data_manager


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

        self.data    = data_manager(dict_c).configure_data(dict_c)


        self.df_f_train = self.data['df_f_train']
        self.df_f_val   = self.data['df_f_val']
        self.df_f_test  = self.data['df_f_test']

        self.df_t_train = self.data['df_t_train']
        self.df_t_val   = self.data['df_t_val']
        self.df_t_test  = self.data['df_t_test']

        print(self.df_f_train.columns)



    def main_CMA_ES(self):

        dimension = self.df_f_train['error_e'].iloc[0].shape[1]
        array = np.zeros(dimension)

        es = cma.CMAEvolutionStrategy(dimension* [0.5], self.sigma,  {'bounds': self.bounds,
                                                                       'maxfevals': self.evals,
                                                                       'verb_disp': self.verbose,
                                                                       'verb_log': self.verbose_log})
        es.optimize(self._opt_function)

        while True:
            sol = es.ask_and_eval(self._opt_function)
            print(sol[1])


        AUC, FPR, TPR       = self.AUC_max, self.FPR, self.TPR
        AUC_v, FPR_v, TPR_v = self._opt_function_(es[0], self.df_f_val['error_m'], self.df_t_val['error_m'])
        AUC_t, FPR_t, TPR_t = self._opt_function_(es[0], self.df_f_test['error_m'], self.df_t_test['error_m'])


        self.df_f_train['error_e'] =  list(map(functools.partial(self._get_error_ensemble, x=es[0]), np.array(self.df_f_train['error_m'])))
        self.df_f_val['error_e']   =  list(map(functools.partial(self._get_error_ensemble, x=es[0]), np.array(self.df_f_val['error_m'])))
        self.df_f_test['error_e']  =  list(map(functools.partial(self._get_error_ensemble, x=es[0]), np.array(self.df_f_test['error_m'])))

        self.df_t_train['error_e'] =  list(map(functools.partial(self._get_error_ensemble, x=es[0]), np.array(self.df_t_train['error_m'])))
        self.df_t_val['error_e']   =  list(map(functools.partial(self._get_error_ensemble, x=es[0]), np.array(self.df_t_val['error_m'])))
        self.df_t_test['error_e']  =  list(map(functools.partial(self._get_error_ensemble, x=es[0]), np.array(self.df_t_test['error_m'])))


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





    def _get_error_max(self,e,x):

        eval_ = np.max(np.dot(e, x))


        return eval_

    def _get_error_ensemble(self,e,x):

        eval_ = np.dot(e, x)


        return eval_





if __name__ == '__main__':
    def return_dict():
        dict_c = {
            'path_i': './models/bayes_opt/DEEP1/',
            'path_o': './models/ensemble/ensemble/',

            'resolution_AUC': 1000,

            ###### CMA_ES    ######
            'CMA_ES': True,
            'verbose_CMA': 1,
            'verbose_CMA_log': 0,
            'evals': 10000,
            'bounds': [-100, 100.],
            'sigma': 0.4222222222222225,
            'progress_ST': 0.3,

            'epoch': 0

        }

        return dict_c

    dict_c = return_dict()
    CMA_ES(dict_c).main_CMA_ES()