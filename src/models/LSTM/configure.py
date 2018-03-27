
def return_dict_bounds():
    dict_c = {
        #### prints             ####
        'print_nr_mov'    : True,
        'print_sum'       : True,
        'print_TB_com': False,


        ##### Filesystem manager ####
        'path_data'        : './data/raw/configured_raw/',
        'path_save'        : './models/LSTM/',
        'name'             : None,

        #### Preprocessing ############
        ## background subtractions ####
        'threshold'        : 200,
        'nr_contours'      : 2,

        ## Peak derivation #############
        'resolution'       : 3,
        'area'             : 200,
        'min_h'            : 20,
        'max_h'            : 200,

        ## PCA componentes #########
        'PCA_components'   : 40,

        #### Data manager  #########
        'mode_data'        : ['p','v','PCA'],
        'shuffle_style'    : 'segmentated',

        ###### CMA_ES    ######
        'verbose_CMA'      : 0,
        'verbose_CMA_log'  : 0,
        'evals'            : 3000,
        'bounds'           : [-100,100.],
        'sigma'            : 0.4222222222222225,
        'progress_ST'      : 0.3,


        ###### Bayes opt ######
        'max_iter'         : 30,
        'initial_n'        : 5,
        'initial_dt'       : 'latin',
        'eps'              : -1,
        'maximize'         : False,

        #### optimizer         #####
        'optimizer'        : 'adam',
        'lr'               : 0.0001,

        ##### model definition  #####
        'window'           : 0,
        'time_dim'         : 20,
        'pred_seq'         : True,
        'stateful'         : False,

        'encoder'          : [500,500,500,500],
        'vector'           : 500,
        'decoder'          : [500,500,500],


        ##### fit                    #####
        'random_state'       : 2,
        'val_split_f'        : 0.2,
        'test_split_f'       : 0.2,

        'val_split_t'        : 0.25,
        'test_split_t'       : 0.25,

        'verbose'          : 2,
        'epochs'           : 10000,
        'batch_size'       : 500,

        'stop_iterations'  : 10,


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
                {'name': 'time_dim',     'type': 'continuous', 'domain': (10, 20)},

                {'name': 'hidden_e_1', 'type': 'continuous', 'domain': (0, 500)},
                {'name': 'hidden_e_2', 'type': 'continuous', 'domain': (0, 500)},
                {'name': 'hidden_e_3', 'type': 'continuous', 'domain': (0, 500)},
                {'name': 'hidden_e_4', 'type': 'continuous', 'domain': (0, 500)},

                {'name': 'vector',     'type': 'continuous', 'domain': (0, 500)},

                {'name': 'hidden_d_1', 'type': 'continuous', 'domain': (0, 500)},
                {'name': 'hidden_d_2', 'type': 'continuous', 'domain': (0, 500)},
                {'name': 'hidden_d_3', 'type': 'continuous', 'domain': (0, 500)},

                {'name': 'position',     'type': 'discrete',   'domain': (0,1)},
                {'name': 'PCA',          'type': 'discrete',   'domain': (0,1)},
                {'name': 'v',            'type': 'discrete',   'domain': (0,1)},

                {'name': 'hidden_e_1',   'type': 'discrete',   'domain': (0,1)},
                {'name': 'hidden_e_2',   'type': 'discrete',   'domain': (0,1)},
                {'name': 'hidden_e_3',   'type': 'discrete',   'domain': (0,1)},
                {'name': 'hidden_e_4',   'type': 'discrete', 'domain': (0, 1)},

                {'name': 'hidden_e_1',   'type': 'discrete',   'domain': (0,1)},
                {'name': 'hidden_e_2',   'type': 'discrete',   'domain': (0,1)},
                {'name': 'hidden_e_3',   'type': 'discrete',   'domain': (0,1)},



    ]

    return dict_c,bounds



