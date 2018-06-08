from src.dst.metrics.AUC import AUC
import numpy as np
import functools
import cma
from src.dst.outputhandler.pickle import pickle_save,pickle_load,pickle_save_
import os
import pandas as pd
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class CMA_ES(AUC):


    def __init__(self, dict_c):
        AUC.__init__(self,dict_c)
        self.dict_c    = dict_c

        self.verbose    = dict_c['verbose_CMA']
        self.verbose_log= dict_c['verbose_CMA_log']

        self.evals     = dict_c['evals']
        self.bounds    = dict_c['bounds']
        self.sigma     = dict_c['sigma']
        self.popsize   = dict_c['popsize']

        self.AUC_tr_max  = -5
        self.FPR_tr      = None
        self.TPR_tr      = None

        self.AUC_v_max   = -5
        self.FPR_v       = None
        self.TPR_v       = None

        self.AUC_t_maxv  = -5
        self.FPR_t       = None
        self.TPR_t       = None



        self.counter       = 1
        self.epoch         = 0

        self.dict_df        = {

                            'population_tr': np.zeros(21),
                            'population_v' : np.zeros(21),
                            'population_t' : np.zeros(21),

                            'epoch'        : None

                                }
        self.array_v_x       = []
        self.array_df        = []

        self.best_x_v        = None
        self.best_x_tr       = None


    def main_SB(self,dict_):
        Queue = mp.Queue()

        p = mp.Process(target=self.main_CMA_ES, args=(dict_,Queue))
        p.daemon = True
        p.start()

        while True:
            if p.is_alive():
                time.sleep(1)
            else:
                p.terminate()
                break

        AUC_v = Queue.get()
        return AUC_v

    def main_CMA_ES(self,*args):


        data = args[0]
        Queue= args[1]


        self.df_f_train = data['df_f_train']
        self.df_f_val   = data['df_f_val']
        self.df_f_test  = data['df_f_test']

        self.df_t_train = data['df_t_train']
        self.df_t_val   = data['df_t_val']
        self.df_t_test  = data['df_t_test']

        dimension = self.df_f_train['error_e'].iloc[0].shape[1]


        es = cma.fmin(self._opt_function, dimension * [1], self.sigma,{'bounds'  : self.bounds,
                                                                       'maxfevals': self.evals,
                                                                       'verb_disp': self.verbose,
                                                                       'verb_log' : self.verbose_log})



        data = (self.AUC_tr_max,self.AUC_v_max,self.AUC_t_maxv)

        Queue.put(self.AUC_v_max)

        print('Solution CMA')
        print(es[0])
        print(self.AUC_tr_max,self.AUC_v_max,self.AUC_t_maxv)
        print()

        df       = pd.DataFrame([{'AUC':self.AUC_tr_max,'AUC_v':self.AUC_v_max,'AUC_t':self.AUC_t_maxv}])[['AUC', 'AUC_v', 'AUC_t']]
        path     = self.dict_c['path_save']+ 'hist.p'
        df_saved = pickle_load(path, None)
        df_saved = df_saved.append(df, ignore_index=False)
        pickle_save_(path, df_saved)
        self.plot(df_saved)
    def plot(self,df):
        fig = plt.figure(figsize=(16, 4))
        plt.plot(list(df['AUC']), label ='train')
        plt.plot(list(df['AUC_v']), label = 'val')
        plt.plot(list(df['AUC_t']), label = 'test')
        plt.savefig(self.dict_c['path_save']+'plots.png')




    def _opt_function(self, x):

        eval_true_tr  = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_t_train['error_e']))))
        eval_false_tr = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_f_train['error_e']))))

        eval_true_v   = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_t_val['error_e']))))
        eval_false_v  = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_f_val['error_e']))))

        eval_true_t   = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_t_test['error_e']))))
        eval_false_t  = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_f_test['error_e']))))


        AUC_tr, FPR_tr, TPR_tr = self.get_AUC_score(eval_true_tr, eval_false_tr)
        AUC_v, FPR_v, TPR_v    = self.get_AUC_score(eval_true_v, eval_false_v)
        AUC_t, FPR_t, TPR_t    = self.get_AUC_score(eval_true_t, eval_false_t)



        if (AUC_v > self.AUC_v_max):
            self.AUC_v_max  = AUC_v
            self.FPR_v      = FPR_v
            self.TPR_v      = TPR_v
            self.best_x_v   = x

            self.AUC_t_maxv = AUC_t
            self.FPR_t      = FPR_t
            self.TPR_t      = TPR_t

            self.AUC_tr_max = AUC_tr
            self.FPR_tr     = FPR_tr
            self.TPR_tr     = TPR_tr

        self.dict_df['population_tr'] = AUC_tr
        self.dict_df['population_v']  = AUC_v
        self.dict_df['population_t']  = AUC_t
        self.dict_df['epoch']         = self.epoch

        self.array_v_x.append(x)


        if(self.epoch%100 == 0):
            self.save_data()

        if(self.counter%self.popsize == 0):
            self.counter             = 1
            self.epoch              += 1
            self.array_df.append(self.dict_df)
            self.dict_df              = {}

            df = pd.DataFrame(self.array_df,columns=['population_tr', 'population_v', 'population_t', 'best_x_v', 'epoch'])


            pickle_save(self.dict_c['path_save'] + '/best/df.p', df)



        else:
            self.counter += 1
            self.array_df.append(self.dict_df)
            self.dict_df              = {}







        return -AUC_tr

    def save_data(self):

        AUC, FPR, TPR       = self.AUC_tr_max, self.FPR_tr, self.TPR_tr
        AUC_v, FPR_v, TPR_v = self._opt_function_(self.best_x_v, self.df_f_val['error_e'], self.df_t_val['error_e'])
        AUC_t, FPR_t, TPR_t = self._opt_function_(self.best_x_v, self.df_f_test['error_e'], self.df_t_test['error_e'])


        path = self.dict_c['path_save'] + 'hist.p'
        df_saved = pickle_load(path, None)
        if(AUC_v > max(df_saved['AUC_v']) ):
            self.df_f_train['error_v'] =  list(map(functools.partial(self._get_error_ensemble, x=self.best_x_v), np.array(self.df_f_train['error_e'])))
            self.df_f_val['error_v']   =  list(map(functools.partial(self._get_error_ensemble, x=self.best_x_v), np.array(self.df_f_val['error_e'])))
            self.df_f_test['error_v']  =  list(map(functools.partial(self._get_error_ensemble, x=self.best_x_v), np.array(self.df_f_test['error_e'])))

            self.df_t_train['error_v'] =  list(map(functools.partial(self._get_error_ensemble, x=self.best_x_v), np.array(self.df_t_train['error_e'])))
            self.df_t_val['error_v']   =  list(map(functools.partial(self._get_error_ensemble, x=self.best_x_v), np.array(self.df_t_val['error_e'])))
            self.df_t_test['error_v']  =  list(map(functools.partial(self._get_error_ensemble, x=self.best_x_v), np.array(self.df_t_test['error_e'])))


            self.df_f_train['error_m'] =  list(map(functools.partial(self._get_error_max, x=self.best_x_v), np.array(self.df_f_train['error_e'])))
            self.df_f_val['error_m']   =  list(map(functools.partial(self._get_error_max, x=self.best_x_v), np.array(self.df_f_val['error_e'])))
            self.df_f_test['error_m']  =  list(map(functools.partial(self._get_error_max, x=self.best_x_v), np.array(self.df_f_test['error_e'])))

            self.df_t_train['error_m'] =  list(map(functools.partial(self._get_error_max, x=self.best_x_v), np.array(self.df_t_train['error_e'])))
            self.df_t_val['error_m']   =  list(map(functools.partial(self._get_error_max, x=self.best_x_v), np.array(self.df_t_val['error_e'])))
            self.df_t_test['error_m']  =  list(map(functools.partial(self._get_error_max, x=self.best_x_v), np.array(self.df_t_test['error_e'])))


            dict_ = {
                'x_v'      : self.best_x_v,
                'x_tr'     : self.best_x_tr,
                'AUC_tr'   : self.AUC_tr_max,
                'AUC_v'    : AUC_v,
                'AUC_t'    : AUC_t,

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

            }

            self._configure_dir(self.dict_c['path_save']+'best')
            pickle_save(self.dict_c['path_save']+'best/data_best.p',dict_)

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

    def _configure_dir(self,path):
        path = path+'/best'
        string_a = path.split('/')
        path = ''

        for string in string_a:
            if string != '':
                path += string+'/'

                if (os.path.exists(path) == False):
                    os.mkdir(path)

