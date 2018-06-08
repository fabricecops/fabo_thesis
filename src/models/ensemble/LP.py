import numpy as np
from src.models.ensemble.model_selection import model_selection
import pandas as pd
from src.models.ensemble.data_manager import data_manager
import functools
from src.dst.metrics.AUC import AUC
import GPyOpt
import time
import matplotlib.pyplot as plt
from src.models.ensemble.CMA_ES import CMA_ES
import os
from src.dst.outputhandler.pickle import pickle_save_,pickle_load

def tic():
    global time_
    time_ = time.time()

def toc():
    global time_
    tmp = time.time()

    elapsed = tmp - time_

    print('the elapsed time is: ', elapsed)

    return elapsed

class BO(data_manager,AUC):

    def __init__(self,dict_c):

        data_manager.__init__(self,dict_c)
        AUC.__init__(self,dict_c)


        self.dict_c = dict_c
        self._configure_dir(self.dict_c['path_save'])
        self.df     = self.configure_data()
        # self.df     = self.df[self.df['AUC_v']> self.dict_c['threshold']]
        MS          = model_selection(self.dict_c)
        self.df     = MS.main(self.df)

        print('MAX AUC_v is equal to:'+str(max(self.df['AUC_v'])))
        print()
        self.df['AUC_v'].plot(kind = 'hist')
        # plt.show()

        self.epoch  = 0
    def bayesian_opt(self):
        bounds = []
        tic()

        path = self.dict_c['path_save'] + 'hist.p'
        df = pd.DataFrame([{'AUC':0.5,'AUC_v':0.5,'AUC_t':0.5}])[['AUC', 'AUC_v', 'AUC_t']]

        if ('hist.p' not in os.listdir(self.dict_c['path_save'])):
            pickle_save_(path, df)
            df_saved = df


        for group in self.df.groupby('clusters'):
            if(self.dict_c['mode'] == 'Kmeans'):
                tmp  = tuple([i for i in range(len(group[1]))])
            else:
                tmp  = tuple([i for i in range(len(self.df))])
            dict_= {'name': 'cluster_'+str(group[0]), 'type': 'discrete','domain': tmp}
            print(dict_)
            bounds.append(dict_)

        print()


        train = GPyOpt.methods.BayesianOptimization(f                      = self.obj,
                                                    domain                 = bounds,
                                                    maximize               = self.dict_c['maximize'],
                                                    initial_design_numdata = self.dict_c['initial_n'],
                                                    initial_design_type    = self.dict_c['initial_dt'],
                                                    eps                    = self.dict_c['eps']
                                                    )


        train.run_optimization(max_iter=self.dict_c['max_iter'])

        print("optimized parameters: {0}".format(train.x_opt))
        print("optimized loss: {0}".format(train.fx_opt))




    def obj(self,x):
        toc()
        print()

        df = pd.DataFrame( columns=['error_m', 'path', 'AUC_v'])
        if(self.dict_c['mode'] == 'Kmeans'):
            for group in self.df.groupby('clusters'):
                df = df.append(group[1].iloc[x[:,group[0]]])
        else:
            for i in range(self.dict_c['clusters']):
                df = df.append(self.df.iloc[x[:, i]])

        dict_ = self.load_data(df)


        if('paper' == self.dict_c['output_f']):
            AUC_v = self.proposed_method_paper(dict_,self.epoch)
        else:
            cma_es = CMA_ES(self.dict_c)
            AUC_v  = cma_es.main_SB(dict_)




        self.epoch+= 1


        print(AUC_v,self.epoch)
        tic()

        return AUC_v

    def proposed_method_paper(self,dict_,epoch):

        bias = np.median(np.concatenate(dict_['df_f_train']['error_e'],axis=0),axis=0)

        for key in dict_.keys():
            dict_[key]['error_v'] = list(map(functools.partial(self.leaky_relu, bias=bias), dict_[key]['error_e']))
            dict_[key]['error_m'] = list(map(np.max, dict_[key]['error_v']))

        AUC_v  = self.get_data_dict(dict_,epoch)
        return AUC_v



    def leaky_relu(self,x,bias=None):
        x         = x-bias
        relu      = np.maximum(0.001*x,x)
        ensemble  = np.mean(relu,axis = 1)

        return ensemble





if __name__ == '__main__':

    dict_c = {
                'path_save': './models/ensemble/',
                'path_a'   : ['./models/segmentated_shuffle/bayes_opt/DEEP2/'],
                'clusters' : 2,
                'mode'     : 'Kmeadns',
                'KM_n_init': 10,
                'threshold': 0.70,

                'output_f' : 'not',

                'resolution_AUC': 1000,
                ###### Bayes opt ######
                'max_iter': 200,
                'initial_n': 30,
                'initial_dt': 'latin',
                'eps': -1,
                'maximize': True,

                ###### CMA_ES    ######
                'CMA_ES': True,
                'verbose_CMA': 0,
                'verbose_CMA_log': 0,
                'evals': 100,
                'bounds': [0, 1.],
                'sigma': 0.4222222222222225,
                'progress_ST': 0.3,
                'popsize': 21,

                'epoch': 0

    }

    MS = BO(dict_c).bayesian_opt()