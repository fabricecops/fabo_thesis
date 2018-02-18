import src.models.tests.model.Conv_net as km
import src.models.tests.BO  as bo

import os
class learning_curve():

    def __init__(self,dict_p,bounds,nr_a):

        self.dict_p  = dict_p
        self.bounds  = bounds
        self.nr_a    = nr_a
        print(self.nr_a)
    def main(self):

        path = './models/conv/learning_curve/'
        for nr in self.nr_a:
            name   = str(nr)
            path_l = path+name+'/'

            if (os.path.exists(path_l) == False):
                os.mkdir(path_l)



            self.dict_p['path_save']           = path_l
            self.dict_p['train_data_dir']      = './data/processed/yale/LC/'+name+'/train/'
            self.dict_p['validation_data_dir'] = './data/processed/yale/LC/'+name+'/val/'


            # BO = bo.BayesionOpt(self.bounds, self.dict_p)
            # BO.main()
            conv =km.Conv_net(self.dict_p)
            conv.fit_generator()




