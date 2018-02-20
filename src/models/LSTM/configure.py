def return_dict_bounds():
    dict_c = {

        ##### prints  #######
        'print_sum'        : True,
        'print_TB_com'     : False,
        'print_BO_bounds'  : False,
        'print_nr_mov'     : True,

        ##### Filesystem manager ####
        'path_data'        : './data/raw/configured_raw/',
        'path_save'        : './models/LSTM/stateful/',
        'name'             : None,

        #### Preprocessing ############
        ## background subtractions ####
        'threshold'        : 100,
        'nr_contours'      : 4,
        'nr_features'      : 3,

        ## Peak derivation #############
        'resolution'       : 6,
        'area'             : 200,
        'min_h'            : 0,
        'max_h'            : 140,

        #### Data manager  #########
        'mode_data'        : ['p'],
        'train'            : 'df_f_tr',
        'val'              : 'df_f_val',
        'anomaly'          : 'df_t',

        ###### CMA_ES    ######
        'verbose_CMA'      : 1,
        'verbose_CMA_log'  : 0,
        'evals'            : 20000,
        'bounds'           : [-100,100],
        'sigma'            : 0.5,
        'progress_ST'      : 0.3,


        ###### Bayes opt ######
        'max_iter'         : 30,
        'initial_n'        : 5,
        'initial_dt'       : 'latin',
        'eps'              : -1,
        'maximize'         : True,

        #### optimizer         #####
        'optimizer'        : 'adam',
        'lr'               : 0.0001,

        ##### model definition  #####
        'window'           : 0,
        'time_dim'         : 10,
        'pred_seq'         : True,
        'dropout'          : 0.03,
        'hidden'           : 500,
        'stateful'         : True,


        ##### fit                    #####
        'val_split'        : 0.15,
        'verbose'          : 1,
        'epochs'           : 10000,
        'batch_size'       : 1,


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
        'save_best_only'        : True,
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
