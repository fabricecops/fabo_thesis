
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
import time
import warnings
import numpy as np
from sklearn import metrics
import pickle
from src.dst.outputhandler.OPS import OPS




#### callbacks ##############
class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class EarlyStoppingratio(Callback):
    def __init__(self, monitor=['val_loss','loss'], value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):

        current_val   = logs.get(self.monitor[0])
        current_train = logs.get(self.monitor[1])

        ratio         = current_val/current_train

        if ratio is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if ratio > self.value:
            if self.verbose > 0:
                print()
                print("Epoch %05d: early stopping RATIO STOPPER" % epoch)
            self.model.stop_training = True

class Stop_reach_trehshold(Callback):
    def __init__(self, monitor=['val_acc'], value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):

        val_acc = logs.get(self.monitor[0])

        if val_acc is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if val_acc > self.value:
            if self.verbose > 0:
                print()
                print("Epoch %05d: early stopping THRESHOLD STOPPER" % epoch)
            self.model.stop_training = True
