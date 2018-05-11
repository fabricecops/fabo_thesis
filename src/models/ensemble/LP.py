import os
from gurobipy import *
from src.dst.outputhandler.pickle import pickle_load
import numpy as np
from src.models.ensemble.model_selection import model_selection
import pandas as pd
from src.models.ensemble.data_manager import data_manager
import functools
from src.dst.metrics.AUC import AUC
from src.dst.outputhandler.pickle import pickle_save_,pickle_load

class LP(data_manager,AUC):

    def __init__(self,dict_c):

        data_manager.__init__(self,dict_c)
        AUC.__init__(self,dict_c)


        self.dict_c = dict_c
        self._configure_dir(self.dict_c['path_save'])
        self.df     = self.configure_data()
        MS          = model_selection(self.dict_c)
        self.df     = MS.main(self.df)

        self.epoch  = 0
    def LP_with_clustering(self):

        self.model        = Model('Model_selection')
        self.vars         = {}
        self.len_clusters = {}


        for group in self.df.groupby('clusters'):
            for j in range(len(group[1])):
                self.len_clusters[group[0]] = len(group[1])
                self.vars[group[0], j] = self.model.addVar(vtype=GRB.BINARY,name='e' + str(group[0]) + '_' + str(j))
        self.model.update()



        for i in range(self.dict_c['clusters']):
            self.model.addConstr((quicksum(self.vars[i, j] for j in range(self.len_clusters[i]))==1),'cluster_'+str(self.len_clusters[i]))
        self.model.update()

        # self.model.setParam('MIPFocus',1)

        obj              = self.obj()
        self.model.setObjective(obj, GRB.MAXIMIZE)
        self.model.update()

        self.model.optimize()


    def obj(self):

        df = pd.DataFrame( columns=['error_m', 'path', 'AUC_v'])
        nr = 0
        for group in self.df.groupby('clusters'):
            for j in range(self.len_clusters[group[0]]):
                if(self.vars[group[0],j]==1):
                    df = df.append(group[1].iloc[j])
                    nr+= 1




        dict_ = self.load_data(df,nr)
        AUC_v = self.proposed_method_paper(dict_,self.epoch)

        self.epoch+= 1
        print(AUC_v)

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
                'mode'     : 'no_cma.p',
                'path_a'   : ['./models/segmentated_shuffle/bayes_opt/DEEP2/'],
                'clusters' : 5,
                'KM_n_init': 10,
                'threshold': 0.6,


                'resolution_AUC': 1000,
    }

    MS = LP(dict_c).LP_with_clustering()