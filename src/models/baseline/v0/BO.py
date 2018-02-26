
from src.models.baseline.v0.configure import return_dict_bounds
from src.models.baseline.v0.baseline import baseline
import tensorflow as tf
config = tf.ConfigProto(
        device_count = {'CPU': 0}
    )
sess = tf.Session(config=config)



import numpy as np

import gpflow
import gpflowopt


class BO():

    def __init__(self):
        self.domain = None
        self._configure()


    def optimizer(self):
        design = gpflowopt.design.LatinHyperCube(11, self.domain)

        X = design.generate()
        print(X)
        # Y = self.opt_function(X)

    def opt_function(self,x):



        BL         = baseline(dict_c)
        AUC,result = BL.main()

        return AUC

    def _configure(self):
        ### time dim
        self.domain = gpflowopt.domain.ContinuousParameter('TD',1,3)
        ### min h
        self.domain+= gpflowopt.domain.ContinuousParameter('MH',100,200)
        ### max h
        self.domain+= gpflowopt.domain.ContinuousParameter('MiH',0,100)
        ### resolution
        self.domain+=gpflowopt.domain.ContinuousParameter('R',100,110)
        ### nr_contours
        self.domain+=gpflowopt.domain.ContinuousParameter('nr_C',1,6)
        ### threshold
        self.domain+=gpflowopt.domain.ContinuousParameter('threshold',100,300)
        ### area
        self.domain+=gpflowopt.domain.ContinuousParameter('A',100,500)
        ### pos/speed
        self.domain =gpflowopt.domain.ContinuousParameter('PV',0,2)






if __name__ == '__main__':

    dict_c, _ = return_dict_bounds()
    BO().optimizer()


