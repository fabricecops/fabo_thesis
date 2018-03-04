

from src.dst.datamanager.data_manager import data_manager
from src.models.LSTM.optimizers.CMA_ES.CMA_ES import CMA_ES
from src.models.baseline.v0.configure import return_dict_bounds
import numpy as np
import functools
import cma
from src.dst.outputhandler.pickle import pickle_save
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd
class baseline(CMA_ES,data_manager):


    def __init__(self, dict_c=None):

        data_manager.__init__(self, dict_c)
        CMA_ES.__init__(self,dict_c)

        self.df_true  = None
        self.df_false = None

        self.AUC_max  = None
        self.AUC_min  = None
        self.FPR      = None
        self.TPR      = None

        self.df_f_train, self.df_t_train, self.df_f_test, self.df_t_test= self.return_split()
        self.configure_data()



    def main(self):


        dimension      = self.df_f_train[0].shape[1]
        array          = np.zeros(dimension)

        parameters     = dimension*self.dict_c['time_dim']
        data_CV        = self.split_CV(self.df_f_train,self.df_t_train,self.dict_c['folds'])

        array_dict   = []

        for data in data_CV:
            self.AUC     = -10

            self.train_f = data[0]
            self.val_f   = data[1]

            self.train_t = data[2]
            self.val_t   = data[3]


            es = cma.fmin(self._opt_function,
                          array,
                          self.sigma,
                          {'bounds'              : self.bounds,
                          'maxfevals'            : self.evals,
                           'verb_disp'           : self.verbose,
                           'verb_log'             :self.verbose_log })

            AUC,FPR,TPR       = self.AUC_max,self.FPR,self.TPR
            AUC_v,FPR_v,TPR_v = self._opt_function_val(es[0])
            AUC_t,FPR_t,TPR_t = self._opt_function_val(es[0])



            dict_ = {
                     'x'     :  es[0],
                    'AUC'    : -es[1],
                    'AUC_v'  :  AUC_v,
                    'AUC_t'  :  AUC_t,

                    'FPR'    : FPR,
                    'TPR'    : TPR,
                    'FPR_v'  : FPR_v,
                    'TPR_v'  : TPR_v,
                    'FPR_t'  : FPR_t,
                    'TPR_t'  : TPR_t,

            }
            array_dict.append(dict_)

        df       = pd.DataFrame(array_dict,columns=['x','AUC','AUC_v','AUC_t','FPR',
                                                    'FPR_v','FPR_t','TPR','TPR_v','TPR_t'])
        CV_score =  self.save_output(df)



        return CV_score,-parameters

    def _opt_function(self, x):


        eval_true  = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.train_t))))
        eval_false = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.train_f))))

        AUC, FPR, TPR = self.get_AUC_score(eval_true, eval_false)
        if (AUC > self.AUC_max):
            self.AUC_max = AUC
            self.FPR = FPR
            self.TPR = TPR

        if (AUC < self.AUC_min):
            self.AUC_min = AUC

        return -AUC

    def _opt_function_val(self, x):


        eval_true  = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.val_t))))
        eval_false = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.val_f))))

        AUC, FPR, TPR = self.get_AUC_score(eval_true, eval_false)


        return AUC,FPR,TPR

    def _opt_function_test(self, x):

        eval_true  = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_t_test))))
        eval_false = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_f_test))))

        AUC, FPR, TPR = self.get_AUC_score(eval_true, eval_false)


        return AUC,FPR,TPR

    def _get_error_m(self, row):

        e_f = np.mean(row, axis = 1)
        return e_f

    def _get_error_max(self,e,x):

        eval_ = np.max(np.dot(e, x))
        return eval_

    def configure_data(self):

        self.df_f_train     = list(map(self._get_error_m, np.array(self.df_f_train['data_X'])))
        self.df_t_train     = list(map(self._get_error_m, np.array(self.df_t_train['data_X'])))

        self.df_f_test      = list(map(self._get_error_m, np.array(self.df_f_test['data_X'])))
        self.df_t_test      = list(map(self._get_error_m, np.array(self.df_t_test['data_X'])))

    def split_CV(self,f,t,folds):
        step_f = len(f)//folds
        step_t  = len(t)/folds


        array = []
        for i in range(folds):
            id_f = i*step_f
            id_t = i*step_t


            val_f   = f[id_f:id_f+step_f]
            train_f = f[0:id_f]
            train_f.extend(f[id_f+step_f:])

            val_t   = t[id_t:id_t + step_t]
            train_t = t[0:id_t]
            train_t.extend(t[id_t + step_t:])

            print(len(val_f),len(val_t),len(train_f),len(train_t))

            array.append((train_f,val_f,train_t,val_t))

        return array

    def save_output(self,df):
        path   = self.dict_c['path_save']
        string = 'experiment_'+str(len(os.listdir(path))-1)
        path   = path+string
        path_d = path+'/data'

        path_n = path_d + '/nr_'+str(len(os.listdir(path_d)))
        if (os.path.exists(path_n) == False):
            os.mkdir(path_n)

        pickle_save(path_n+'/df.p',df)
        fig = plt.figure(figsize=(16, 4))

        ax1 = plt.subplot(131)
        for i in range(len(df)):
            ax1.plot(df['FPR'].iloc[i],df['TPR'].iloc[i], label = 'CV_'+str(i)+'_'+str(round(df['AUC'].iloc[i],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('Train AUC')


        ax2 = plt.subplot(132)
        for i in range(len(df)):
            ax2.plot(df['FPR_v'].iloc[i],df['TPR_v'].iloc[i], label = 'CV_'+str(i)+'_'+str(round(df['AUC_v'].iloc[i],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('val AUC')

        ax3 = plt.subplot(133)
        for i in range(len(df)):
            ax3.plot(df['FPR_t'].iloc[i],df['TPR_t'].iloc[i], label = 'CV_'+str(i)+'_'+str(round(df['AUC_t'].iloc[i],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('test AUC')
        plt.savefig(path_n +'/AUC_curve')



        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)
        for i in range(len(df)):
            ax1.plot(df['x'].iloc[i], label = 'x_'+str(i)+'_'+str(round(df['AUC'].iloc[i],3)))
        plt.xlabel('weigths')
        plt.ylabel('value')
        plt.legend()
        plt.title('Weights cma')


        ax2 = plt.subplot(122)
        for i in range(len(df)):
            ax2.plot(df['FPR_v'].iloc[i],df['TPR_v'].iloc[i], label = 'CV_'+str(i)+'_'+str(round(df['AUC_v'].iloc[i],3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.title('val AUC')

        plt.show()

if __name__ == '__main__':

    dict_c, _ = return_dict_bounds()



    baseline_  = baseline(dict_c)

    baseline_.main()







