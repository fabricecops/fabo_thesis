
def return_dict_bounds():
    dict_c = {
        #### prints             ####
        'print_nr_mov'    : True,
        'print_sum'       : True,
        'print_TB_com'    : False,


        ##### Filesystem manager ####
        'path_data'        : './data/raw/configured_raw/',
        'path_save'        : './models/bayes_opt/',
        'name'             : None,

        #### Preprocessing ############
        ## background subtractions ####
        'threshold'        : 200,
        'nr_contours'      : 4,

        ## Peak derivation #############
        'resolution'       : 6,
        'area'             : 200,
        'min_h'            : 20,
        'max_h'            : 200,

        ## PCA componentes #########
        'PCA_components'   : 50,

        #### Data manager  #########
        'mode_data'        : ['p','PCA'],
        'shuffle_style'    : 'segmentated',

        ###### CMA_ES    ######
        'CMA_ES'           : True,
        'verbose_CMA'      : 0,
        'verbose_CMA_log'  : 0,
        'evals'            : 5,
        'bounds'           : [-100,100.],
        'sigma'            : 0.4222222222222225,
        'progress_ST'      : 0.3,


        ###### Bayes opt ######
        'max_iter'         : 500,
        'initial_n'        : 20,
        'initial_dt'       : 'latin',
        'eps'              : -1,
        'maximize'         : False,

        #### optimizer         #####
        'optimizer'        : 'adam',
        'lr'               : 0.0001,

        ##### model definition  #####
        'window'           : 0,
        'time_dim'         : 10,
        'pred_seq'         : True,
        'stateful'         : False,

        'encoder'          : [400,350],
        'vector'           : 300,
        'decoder'          : [400],


        ##### fit                    #####
        'random_state'       : 2,
        'val_split_f'        : 0.2,
        'test_split_f'       : 0.2,

        'val_split_t'        : 0.25,
        'test_split_t'       : 0.25,

        'verbose'          : 2,
        'epochs'           : 10000,
        'batch_size'       : 1024,

        'SI_cma'           : 1,
        'SI_no_cma'        : 3,
        'SI_no_cma_AUC'    : 1,

        'TH_loss'          : 0.025,


        #### data gathering statefull ####

        ##### callbacks   #########
        ## Early stopping
        'ES'                    : False,
        'ES_patience'           : 5,

        ## LRonplateau
        'LR_P'                  : False,
        'LR_factor'             : 0.3,
        'LR_patience'           : 8,

        ## Ratio ES
        'ESR'                   : False,
        'early_ratio_val'       : 100,

        ## TB
        'TB'                    : False,
        'hist_freq'             : True,
        'w_graphs'              : True,
        'w_images'              : True,
        'w_grads'               : True,

        ## Monitor callback
        'MT'                    : False,

        ## model checkpoint
        'MC'                    : False,
        'mode_MC'               : 'min',
        'save_best_only'        : False,
        'verbose_MC'            : 0,

        ## history
        'hist'                  : False,

        ##CSV append
        'CSV'                   : False,
        'CSV_append'            : False,

        ##Calc AUC
        'AUC'                   : True,
        'resolution_AUC'        : 1000,
        'epoch_mod_AUC'         : 1,
        'verbose_AUC'           : 1,

        ##TH stopper
        'TH_stopper'            : False

    }


    bounds = [
                {'name': 'lr',           'type': 'continuous', 'domain': (0.0001, 0.1)},
                {'name': 'sigma',        'type': 'continuous', 'domain': (0.0001,2)},
                {'name': 'time_dim',     'type': 'continuous', 'domain': (5, 17)},

                {'name': 'hidden_e_1', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_e_2', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_e_3', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_e_4', 'type': 'continuous', 'domain': (100, 500)},

                {'name': 'vector',     'type': 'continuous', 'domain': (200, 800)},

                {'name': 'hidden_d_1', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_d_2', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_d_3', 'type': 'continuous', 'domain': (100, 500)},





    ]
    bounds = [
                {'name': 'lr',           'type': 'continuous', 'domain': (0.0001, 0.1)},
                {'name': 'sigma',        'type': 'continuous', 'domain': (0.0001,2)},
                {'name': 'time_dim',     'type': 'continuous', 'domain': (5, 17)},

                {'name': 'hidden_e_1', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_e_2', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_e_3', 'type': 'continuous', 'domain': (100, 500)},


                {'name': 'hidden_d_1', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_d_2', 'type': 'continuous', 'domain': (100, 500)},

                {'name': 'vector', 'type': 'continuous', 'domain': (200, 800)},

    ]



    return dict_c,bounds



