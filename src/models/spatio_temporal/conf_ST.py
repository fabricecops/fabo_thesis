
def return_dict():
    dict_c = {
        #### prints             ####
        'print_nr_mov'    : True,
        'print_sum'       : True,
        'print_TB_com'    : False,


        ##### Filesystem manager ####
        'path_data'        : './data/raw/configured_raw/',
        'path_save'        : './models/test_shuffle/sneaky/ST',
        'name'             : None,

        #### Preprocessing ############
        ## background subtractions ####
        'width'   : 64,
        'heigth'  : 64,
        'resolution_AUC' : 1000,


        #### Data manager  #########
        'shuffle_style'    : 'test_class',
        'test_class'       : ['sneaky'],

        ###### Bayes opt ######
        'max_iter'         : 200,
        'initial_n'        : 30,
        'initial_dt'       : 'latin',
        'eps'              : -1,
        'maximize'         : True,

        #### optimizer         #####
        'optimizer'        : 'adam',
        'lr'               : 0.0001,

        ##### model definition  #####
        'window'           : 0,
        'time_dim'         : 3,
        'pred_seq'         : True,
        'stateful'         : False,

        'conv_encoder'     : [('filters','ks','strides')],
        'LSTM_encoder'     : [('filters', 'ks')],

        'middel_LSTM'      : ('filters', 'ks'),
        'conv_decoder'     : [('filters', 'ks', 'strides')],

        ##### fit                    #####
        'random_state'       : 1,
        'val_split_f'        : 0.2,
        'test_split_f'       : 0.2,

        'val_split_t'        : 0.25,
        'test_split_t'       : 0.25,

        'steps_per_epoch'    : 100,
        'steps_per_epoch_val': 30,
        'verbose'          : 1,
        'epochs'           : 10000,
        'batch_size'       : 1024,

        'SI_no_cma'        : 20,
        'SI_no_cma_AUC'    : 20,
        'TH_val_loss'      : 0.1,
        'time_stop'        : 12000,

        'mod_data'         : 10,

        'max_batch_size'   : 512,

    }
    return dict_c

def return_bounds():
    bounds = [
        {'name': 'lr', 'type': 'continuous', 'domain': (0.0001, 0.1)},
        {'name': 'time_dim', 'type': 'continuous', 'domain': (5, 20)},

        {'name': 'conv1_f',  'type': 'discrete', 'domain': [8,16,32,64]},
        {'name': 'conv1_ks', 'type': 'discrete', 'domain': [2, 4,8]},
        {'name': 'conv1_st', 'type': 'discrete', 'domain': [2,4,8]},

        {'name': 'conv2_f', 'type': 'discrete', 'domain': [8,16,32,64]},
        {'name': 'conv2_ks', 'type': 'discrete', 'domain': [2, 4,8]},
        {'name': 'conv2_st', 'type': 'discrete', 'domain': [2,4]},

        {'name': 'conv_lstm_1_f', 'type': 'discrete', 'domain':  [8,16,32,64]},
        {'name': 'conv_lstm_1_ks', 'type': 'discrete', 'domain':[1, 2,4]},

        {'name': 'conv_lstm_2_f', 'type': 'discrete', 'domain': [8,16,32,64]},
        {'name': 'conv_lstm_2_ks', 'type': 'discrete', 'domain': [1, 2,4]}]


    return bounds









