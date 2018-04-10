

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
