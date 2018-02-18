
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.callbacks import History
from keras.models import load_model
from keras.optimizers import Adam

from src.dst.keras_model.FS_manager import *
from src.dst.keras_model.callbacks  import *
from src.dst.outputhandler.pickle import pickle_load,pickle_save

class model(FS_manager):

    def __init__(self,dict_c=None,path=None):

        if(dict_c != None):
            FS_manager.__init__(self, dict_c['path_save'])
            pickle_save(self.return_path_dict(),dict_c)
            self.dict_c  = dict_c

        elif(path != None):
            FS_manager.__init__(self,path)
            self.dict_c = pickle_load(self.return_path_dict(),None)
        else:
            print('WRONG CONF, GIVE PATH OR DICT')


    #### return functions ####
    def save_model(self):

        path = self.return_path_model()
        self.model.save(path)

    def return_model(self):

        return self.model

    def load_model(self):

        return load_model(self.return_path_model())

    def return_TB_command(self):
        path    = '~/Dropbox/Code/Beeldverwerking/git'+self.return_path_TB()[1:]
        command = 'tensorboard --logdir '+path

        return command

    def conf_optimizer(self):
        if self.dict_c['optimizer'] == 'adam':
            return Adam(lr=self.dict_c['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    def create_callbacks(self, **kwargs):
        callbacks = []

        if (self.dict_c['MT'] == True):
            time_CB = TimeHistory()
            callbacks.append(time_CB)

        if(self.dict_c['hist'] == True):
            hist   = History()
            callbacks.append(hist)

        if(self.dict_c['ES'] == True):
            ES         = EarlyStopping(patience=self.dict_c['ES_patience'],
                                    verbose=0)
            callbacks.append(ES)

        if(self.dict_c['TB'] == True):
            # test_datagen = ImageDataGenerator(rescale=self.dict_c['rescale'])

            # validation_generator = test_datagen.flow_from_directory(
            #     self.dict_c['validation_data_dir'],
            #     target_size=(self.dict_c['img_width'], self.dict_c['img_height']),
            #     batch_size=self.dict_c['batch_size'],
            #     color_mode=self.dict_c['colormode'],
            #     shuffle=self.dict_c['shuffle_val'])
            #
            #
            # tensorboard = TensorBoardWrapper(validation_generator,20,
            #                           log_dir=self.return_path_TB(),
            #                           histogram_freq=1,
            #                           write_graph=True,
            #                           write_images=True,
            #                           embeddings_freq=0,
            #                           write_grads=True,
            #                           embeddings_layer_names=True,
            #                           embeddings_metadata=True)

            tensorboard = TensorBoard(
                                      log_dir=self.return_path_TB(),
                                      histogram_freq           = self.dict_c['hist_freq'],
                                      write_graph              = self.dict_c['w_graphs'],
                                      write_images             = self.dict_c['w_images'],
                                      write_grads              = self.dict_c['w_grads'])
            callbacks.append(tensorboard)

        if(self.dict_c['MC']== True):
            Model_c     = ModelCheckpoint(self.return_path_model(),
                                          save_best_only = self.dict_c['save_best_only'],
                                          mode           = self.dict_c['mode_MC'],
                                          verbose        = self.dict_c['verbose_MC'])
            callbacks.append(Model_c)

        if(self.dict_c['LR_P'] == True):
            R_LR_plat   = ReduceLROnPlateau(monitor      = 'val_loss',
                                      factor       = self.dict_c['LR_factor'],
                                      patience     = self.dict_c['LR_patience'],
                                      verbose      = 0,
                                      mode         = 'auto',
                                      epsilon      = 0.0001,
                                      cooldown     = 0,
                                      min_lr       = 0)
            callbacks.append(R_LR_plat)

        if(self.dict_c['TH_stopper']==True):
            TH_stopper = Stop_reach_trehshold(value=self.dict_c['TH_value'])
            callbacks.append(TH_stopper)

        if(self.dict_c['ESR'] == True):
            Early_ratio = EarlyStoppingratio(value= self.dict_c['early_ratio_val'], verbose = 1)
            callbacks.append(Early_ratio)

        if(self.dict_c['CSV'] == True):
            csv         = CSVLogger(filename=self.return_path_CSV(),
                                    append  =self.dict_c['CSV_append'])
            callbacks.append(csv)

        # if(self.dict_c['AUC'] == True):
        #     df_true  = kwargs['df_true']
        #     df_false = kwargs['df_false']
        #
        #     if(self.dict_c['stateful'] == False):
        #
        #         OPS         = OPS_CB(  self.path_gen,
        #                                self.model,
        #                                df_true,
        #                                df_false,
        #                                self.dict_c
        #
        #                            )
        #         callbacks.append(OPS)





        return callbacks

    def print_summary(self):
        if(self.dict_c['print_sum']==True):
            print(self.model.summary())

    def print_TB_command(self):
        command = self.return_TB_command()
        if(self.dict_c['print_TB_com']== True):
            print(command)

