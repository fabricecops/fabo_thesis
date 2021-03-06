
def return_dict_bounds(bounds = 'DEEP1'):
    dict_c = {
        #### prints             ####
        'print_nr_mov'    : True,
        'print_sum'       : True,
        'print_TB_com'    : False,


        ##### Filesystem manager ####
        'path_data'        : './data/raw/configured_raw/',
        'path_save'        : './models/test_shuffle/sneaky/',
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
        'tracker'          : False,
        'state'            : False,


        ## PCA componentes #########
        'PCA_components'   : 50,

        #### Data manager  #########
        'mode_data'        : ['p'],
        'shuffle_style'    : 'segmentated',
        'test_class'       : ['sneaky'],

        ###### Bayes opt ######
        'max_iter'         : 200,
        'initial_n'        : 30,
        'initial_dt'       : 'latin',
        'eps'              : -1,
        'maximize'         : True,

        #### optimizer         #####
        'optimizer'        : 'adam',
        'lr'               : 0.001267345303142497,

        ##### model definition  #####
        'window'           : 0,
        'time_dim'         : 20,
        'pred_seq'         : True,
        'stateful'         : False,

        'encoder'          : [240, 443],
        'vector'           : 240,
        'decoder'          : [470],



        ##### fit                    #####
        'random_state'       : 2,
        'val_split_f'        : 0.2,
        'test_split_f'       : 0.2,

        'val_split_t'        : 0.25,
        'test_split_t'       : 0.25,

        'verbose'          : 1.,
        'epochs'           : 10000,
        'batch_size'       : 1024,

        'SI_no_cma'        : 50,
        'SI_no_cma_AUC'    : 50,
        'TH_val_loss'      : 1,
        'time_stop'        : 10000000000,

        'mod_data'         : 1,



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


    bounds_DEEP1 = [
                {'name': 'lr',           'type': 'continuous', 'domain': (0.0001, 0.1)},
                {'name': 'time_dim',     'type': 'continuous', 'domain': (5, 17)},

                {'name': 'vector', 'type': 'continuous', 'domain': (200, 800)},

                {'name': 'hidden_e_1', 'type': 'continuous', 'domain': (100, 500)}]



    bounds_DEEP2 = [
                 {'name': 'time_dim', 'type': 'continuous', 'domain': (15, 25)},

                 {'name': 'lr',           'type': 'continuous', 'domain': (0.00001, 0.01)},

                {'name': 'vector', 'type': 'continuous', 'domain': (100, 800)},

                {'name': 'hidden_e_1', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_e_2', 'type': 'continuous', 'domain': (100, 500)},


                {'name': 'hidden_d_1', 'type': 'continuous', 'domain': (100, 500)},

                {'name': 'p', 'type': 'discrete', 'domain': [0, 1,2]},
                {'name': 'v', 'type': 'discrete', 'domain': [0, 1]}

                  ]


    bounds_DEEP3 = [
                {'name': 'time_dim',     'type': 'continuous', 'domain': (15, 25)},
                {'name': 'lr',           'type': 'continuous', 'domain': (0.0001, 0.1)},
                {'name': 'vector',       'type': 'continuous', 'domain': (50, 800)},

                {'name': 'hidden_e_1', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_e_2', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_e_3', 'type': 'continuous', 'domain': (100, 500)},


                {'name': 'hidden_d_1', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'hidden_d_2', 'type': 'continuous', 'domain': (100, 500)},
                {'name': 'p', 'type': 'discrete', 'domain': [0, 1,2]},
                {'name': 'v', 'type': 'discrete', 'domain': [0, 1]}]



    if(bounds == 'DEEP1'):
        return dict_c,bounds_DEEP1

    if(bounds == 'DEEP2'):
        return dict_c,bounds_DEEP2

    if(bounds == 'DEEP3'):
        return dict_c,bounds_DEEP3






