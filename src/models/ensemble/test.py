from src.models.ensemble.ensemble import ensemble



class model_e_tests():

    def __init__(self,dict_c):
        self.dict_c = dict_c

    def random_vs_not_random(self):


        for i in range(1,10):
            self.dict_c['random']    = True
            self.dict_c['clusters']  = i
            self.dict_c['path_save'] = './models/ensemble/RvsNR/'+str(self.dict_c['random'])+'/clusters_'+str(i)
            ES                       = ensemble(self.dict_c)
            ES.main()

            self.dict_c['random']    = False
            self.dict_c['path_save'] = './models/ensemble/RvsNR/'+str(self.dict_c['random'])+'/clusters_'+str(i)
            ES                       = ensemble(self.dict_c)
            ES.main()

    def CMA_vs_MEAN(self):


        for i in range(2,10):
            self.dict_c['random']    = False
            self.dict_c['mean']      = False

            self.dict_c['clusters']  = i
            self.dict_c['path_save'] = './models/ensemble/CMA_vs_mean/'+str(self.dict_c['mean'])+'/clusters_'+str(i)
            ES                       = ensemble(self.dict_c)
            ES.main()

            self.dict_c['random']    = False
            self.dict_c['mean']      = True

            self.dict_c['path_save'] = './models/ensemble/CMA_vs_mean/'+str(self.dict_c['mean'])+'/clusters_'+str(i)
            ES                       = ensemble(self.dict_c)
            ES.main()


def return_dict():

    dict_c  = {
        'path_save'       : './models/ensemble/',
        'mode'            : 'no_cma.p',
        'path_a'          : ['./models/bayes_opt/DEEP2/'],
        'clusters'        : 10,
        'KM_n_init'       : 10,
        'threshold'       : 0.6,
        'mean'            : True,
        'random'          : False,
        'resolution_AUC'  : 1000,

        ###### CMA_ES    ######
        'CMA_ES'         : True,
        'verbose_CMA'    : 1,
        'verbose_CMA_log': 0,
        'evals'          : 21*150,
        'bounds'         : [-100, 100.],
        'sigma'          : 0.4222222222222225,
        'progress_ST'    : 0.3,
        'popsize'        : 21,


        'epoch'          : 0


    }


    return dict_c
if __name__ == '__main__':

    dict_c = return_dict()

    model_e_tests(dict_c).random_vs_not_random()

    # model_e_tests(dict_c).CMA_vs_MEAN()
