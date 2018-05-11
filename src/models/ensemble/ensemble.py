
class ensemble():

    def __init__(self,dict_c):
        pass


    def main(self):
        pass





if __name__ == '__main__':
    def return_dict():

        dict_c = {
            'path_save'     : './models/ensemble/test/',
            'path_a'        : ['./models/bayes_opt/DEEP2/'],

            'resolution_AUC': 1000,
            'mode'          : 'no_cma.p',
            'clusters'      : 2,
            'KM_n_init'     : 10,
            'threshold'     : 0.6,


            #### fit ############
            'iterations'    : 10,


            ###### CMA_ES    ######
            'CMA_ES'          : True,
            'verbose_CMA'     : 1,
            'verbose_CMA_log' : 0,
            'evals'           : 21*1,
            'bounds'          : [-100., 100.],
            'sigma'           : 0.4222222222222225,
            'progress_ST'     : 0.3,
            'popsize'         : 21,

            'epoch': 0

        }
        return dict_c

    dict_c = return_dict()


    dict_c['path_a']    = ['./models/bayes_opt/DEEP2/']
    dict_c['path_save'] = './models/ensemble/test/df_'+str(dict_c['clusters'])+'.p'
    ensemble(dict_c).main()


