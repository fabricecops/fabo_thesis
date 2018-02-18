def return_conf():

    dict_p = {

        ###### Bayes opt ######
        'max_iter'    : 400,
        'initial_n'   : 15,
        'initial_dt'  : 'latin',
        'eps'         : -1,
        'min_acc'     : 0.99,
        'maximize'    : False,

        ##### Filesystem manager ####
        'path_save'             : './models/conv/BO_no_data_aug/',
        'name'                  : None,

        ##### Generator         ####
        'train_data_dir'        : './data/processed/yale/BO_data/train/',
        'validation_data_dir'   : './data/processed/yale/BO_data/val/',

        'train_steps'           : 100,
        'epochs'                : 100,
        'batch_size'            : 64,
        'val_steps'             : 50,
        'verbose'               : 2,
        'shuffle_val'           : False,
        'val_step_eval'         : 200,

        'img_width'             : 160,
        'img_height'            : 220,
        'colormode'             : 'grayscale',
        'rescale'               : 1. / (255. / 2.) - 1,
        'shear_range'           : 0,
        'zoom_range'            : 0,
        'horizontal_flip'       : False,

        ##### model definition  #####
        'downsampling'          : [(16, 3, 3, 1,'relu')],
        'upsampling'            : [(1, 3, 3, 1,'relu')],
        'output'                : 28,

        'lr'                    : 0.005,
        'dropout'               : 0.3,
        'batch_norm'            : True,
        'optimizer'             : 'adam',
        'loss'                  : 'categorical_crossentropy',

        ##### callbacks   #########
        ## Early stopping
        'ES'                    : True,
        'ES_patience'           : 5,

        ## LRonplateau
        'LR_P'                  : False,
        'LR_factor'             : 0.3,
        'LR_patience'           : 8,

        ## Ratio ES
        'ESR'                   : True,
        'early_ratio_val'       : 100,

        ## TB
        'TB'                    : False,

        ## Monitor time
        'MT'                    : True,

        ## model checkpoint
        'MC'                    : True,

        ## history
        'hist'                  : True,

        ##TH_stopper
        'TH_stopper'            : True,
        'TH_value'              : 0.995,

        ##CSV append
        'CSV'                   : True,
        'CSV_append'            : False,

    }

    bounds = [{'name': 'dropout',         'type': 'continuous', 'domain': (0.0, 0.4)},
              {'name': 'lr',              'type': 'continuous', 'domain': (0.0001, 0.01)},
              {'name': 'batch_norm',      'type': 'discrete' ,  'domain': (0,1)},

              {'name': 'filter_o_1',      'type':  'discrete',  'domain': (4,8,16,32,64)},
              {'name': 'filter_fd_1',     'type':  'discrete',  'domain': (2, 3, 4, 5)},
              {'name': 'filter_ds_1',     'type':  'discrete',  'domain': (2, 3, 4, 5)},

              {'name': 'filter_o_2',      'type': 'discrete',   'domain':  (4, 8, 16, 32)},
              {'name': 'filter_fd_2',     'type': 'discrete',   'domain':  (2, 3, 4, 5)},
              {'name': 'filter_ds_2',     'type': 'discrete',   'domain':  (2, 3, 4, 5)},

              {'name': 'filter_o_3',      'type': 'discrete',   'domain':  (4, 8, 16, 32)},
              {'name': 'filter_fd_3',     'type': 'discrete',   'domain':  (2, 3, 4, 5)},
              {'name': 'filter_ds_3',     'type': 'discrete',   'domain':  (2, 3, 4, 5)},

              {'name': 'filter_o_4',      'type': 'discrete',   'domain':  (0,4, 8, 16, 32)},
              {'name': 'filter_fd_4',     'type': 'discrete',   'domain':  (2, 3, 4, 5)},
              {'name': 'filter_ds_4',     'type': 'discrete',   'domain':  (2, 3, 4, 5)},

              {'name': 'filter_o_5',      'type': 'discrete',   'domain':  (0,4, 8, 16)},
              {'name': 'filter_fd_5',     'type': 'discrete',   'domain':  (2, 3, 4, 5)},
              {'name': 'filter_ds_5',     'type': 'discrete',   'domain':  (2, 3, 4, 5)},
              ]

    return dict_p,bounds



class model_manager():

    def __init__(self):
        pass

    def BO_no_data_aug(self, **kwargs):
        dict_p, bounds = return_conf_BOnodataaug()
        BO = bo.BayesionOpt(bounds, dict_p)
        BO.main()

    def learning_curve(self):
        dict_,bounds,nr_a  = return_conf_LC()
        LC_                = LC.learning_curve(dict_,bounds,nr_a)
        LC_.main()

    def BO_own_generated_data(self):
        dict_p, bounds = return_conf_OGD()
        BO = bo.BayesionOpt(bounds, dict_p)
        BO.main()
if __name__ == '__main__':

    mm = model_manager()
    # mm.BO_no_data_aug()
    # mm.learning_curve()
    mm.BO_own_generated_data()



    # dict_,bounds = return_conf_OGD()
    # lol = cv.Conv_net(dict_)
    # gen = lol.fit_generator()

