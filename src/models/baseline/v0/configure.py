def return_dict_bounds():
    dict_c = {

        ##### prints  #######
        'print_sum'        : True,
        'print_TB_com'     : False,
        'print_BO_bounds'  : False,
        'print_nr_mov'     : True,

        ##### Filesystem manager ####
        'path_data': './data/raw/configured_raw/',
        'path_save': './models/BO_BL/',
        'name': None,


        #### Preprocessing ############
        ## background subtractions ####
        'threshold'        : 100,
        'nr_contours'      : 4,
        'nr_features'      : 3,

        ## Peak derivation #############
        'resolution'       : 50,
        'area'             : 200,
        'min_h'            : 10,
        'max_h'            : 150,

        ## PCA componentes #########
        'PCA_components': 100,

        #### Data manager  #########
        'mode_data'        : ['p'],
        'train'            : 'df_f_tr',
        'val'              : 'df_f_val',
        'anomaly'          : 'df_t',

        ###### CMA_ES    ######
        'verbose_CMA'      : 1,
        'verbose_CMA_log'  : 0,
        'evals'            : 10000,
        'bounds'           : [0,100],
        'sigma'            : 0.5,
        'progress_ST'      : 0.3,


        ###### Bayes opt ######
        'nr_iter'         : 100,
        'initial_n'        : 10,
        'initial_dt'       : 'latin',
        'eps'              : -1,


        ##### fit                    #####
        'folds'            : 5,
        'bootstrap_split': 0.3,
        'val_split_f': 0.2,
        'val_split_t': 0.2,
        'stateful'   : None,


        ##### model definition  #####
        'window': 0,
        'time_dim': 10,
        'pred_seq': True,
        'resolution_AUC': 1000,

        #### data gathering statefull ####

    }

    bounds = [{'name': 'dropout',  'type': 'continuous', 'domain': (0.0, 0.4)},
              {'name': 'lr',       'type': 'continuous', 'domain': (0.0001, 0.01)},
              {'name': 'time_dim', 'type': 'discrete',   'domain': (5, 10, 20, 40)},
              {'name': 'latent',   'type': 'discrete',   'domain': (100, 150, 200, 250, 300)}]

    return dict_c,bounds
