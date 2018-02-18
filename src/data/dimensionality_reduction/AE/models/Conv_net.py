

from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.layers import Dropout, Flatten, Dense,BatchNormalization
from keras.optimizers import Adam


from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import cv2

from src.data.dimensionality_reduction.AE.configure import return_conf

from PIL import Image

from src.dst.keras_model.model import model

from keras.layers import merge, Input,MaxPooling2D,Flatten,UpSampling2D


class AE(model):

    def __init__(self,dict_c=None,path=None):



        model.__init__(self, dict_c=dict_c,path=path)


        self.model = self.create_model()
        if(path != None ):
            self.model = self.load_model()




    def create_model(self):



        input_shape = (self.dict_c['img_width'], self.dict_c['img_height'], 1)
        model = Sequential()
        for i, filter_ in enumerate(self.dict_c['downsampling']):

            if (i == 0):
                conv_layer = Conv2D(filter_[0], (filter_[1], filter_[2]), activation='relu', padding='same',
                                    input_shape=input_shape)
                max_pool = MaxPooling2D((filter_[3], filter_[3]), padding='same')
            else:
                conv_layer = Conv2D(filter_[0], (filter_[1], filter_[2]), activation='relu', padding='same')
                max_pool = MaxPooling2D((filter_[3], filter_[3]), padding='same')

            model.add(conv_layer)
            model.add(max_pool)

        for up_sample in self.dict_c['upsampling']:
            conv_layer = Conv2D(up_sample[0], (up_sample[1], up_sample[2]), activation='relu', padding='same')
            upsample = UpSampling2D((up_sample[3], up_sample[3]))
            model.add(conv_layer)
            model.add(upsample)



        optimizer = self.conf_optimizer()
        loss      = self.dict_c['loss']

        model.compile(optimizer=optimizer, loss=loss)
        assert (model.output_shape == model.input_shape)


        return model

    def fit_generator(self):
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
                rescale=self.rescale,
                shear_range=self.shear_range,
                zoom_range=self.shear_range,
                horizontal_flip=self.shear_range)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=self.rescale)


        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size= self.batch_size,
            color_mode= self.colormode)


        validation_generator = test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            color_mode= self.colormode)

        history=self.model.fit_generator(
                self.fixed_generator(train_generator),
                steps_per_epoch=self.nb_train_samples//self.batch_size,
                epochs=self.epochs,
                validation_data=self.fixed_generator(validation_generator),
                validation_steps=self.nb_validation_samples//self.batch_size,
                callbacks = [self.tensorboard,self.history]
                )
        return history.history
    def predict(self,image):

                test_datagen = self.return_img_gen()
                ### LOAD IMG replicator
                img = image
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    pass

                img = Image.fromarray(img)

                if (img.mode != 'L'):
                    img = img.convert('L')


                img = img.resize((self.dict_c['img_height'], self.dict_c['img_width']), 2)
                img = img_to_array(img, data_format='channels_last')
                img = img.reshape(1, 160, 220, 1)
                img = test_datagen.flow(img)[0]

                pred = self.model.predict(img)

                return pred

    def _fixed_generator(self, generator):
        for batch in generator:
            yield (batch[0], batch[0])

if __name__ == '__main__':

    dict_c,_ = return_conf()
    AE_ = AE(dict_c)

    # AE.fit_generator()