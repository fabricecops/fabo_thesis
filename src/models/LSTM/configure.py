
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
        'nr_contours'      : 4,

        ## Peak derivation #############
        'resolution'       : 4,
        'area'             : 200,
        'min_h'            : 20,
        'max_h'            : 200,

        ## PCA componentes #########
        'PCA_components'   : 50,

        #### Data manager  #########
        'mode_data'        : ['p','PCA'],
        'shuffle_style'    : 'testing',

        ###### CMA_ES    ######
        'verbose_CMA'      : 0,
        'verbose_CMA_log'  : 0,
        'evals'            : 2000,
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
        'lr'               : 0.001,

        ##### model definition  #####
        'window'           : 0,
        'time_dim'         : 10,
        'pred_seq'         : True,
        'stateful'         : False,


        ##### fit                    #####
        'val_split_f'        : 0.2,
        'test_split_f'       : 0.2,

        'val_split_t'        : 0.2,
        'test_split_t'       : 0.2,

        'verbose'          : 2,
        'epochs'           : 10000,
        'batch_size'       : 500,

        'stop_iterations'  : 1,


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

    bounds = [{'name': 'dropout',  'type': 'continuous', 'domain': (0.0, 0.4)},
              {'name': 'lr',       'type': 'continuous', 'domain': (0.0001, 0.01)},
              {'name': 'time_dim', 'type': 'discrete',   'domain': (5, 10, 20, 40)},
              {'name': 'latent',   'type': 'discrete',   'domain': (100, 150, 200, 250, 300)}]

    return dict_c,bounds



