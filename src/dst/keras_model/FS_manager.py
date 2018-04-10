import os
import pickle

class FS_manager():

    def __init__(self,path=None):

        self.path                  = path
        self._configure_dir(self.path)
        self.path_gen,self.path_TB,self.path_output = self._create_dir()


    def return_path(self):
        return self.path_gen

    def return_path_model(self):
        path_model = self.path_gen+'model.h5'
        return path_model

    def return_path_TB(self):
        return self.path_TB

    def return_path_dict(self):
        path_dict =  self.path_gen+'dict.p'
        return path_dict

    def return_path_hist(self):
        path_dict = self.path_gen + 'hist.p'
        return path_dict

    def return_path_CSV(self):
        path_dict = self.path_gen + 'hist.csv'
        return path_dict

    def return_path_output(self):
        return self.path_output

    def _create_dir(self):
        list_ = os.listdir(self.path)
        if('dict.p' not in  list_):
            path_gen = self.path+str(len(os.listdir(self.path)))+'/'
        else:
            path_gen = self.path


        path_TB     =  path_gen +'tensorboard/'
        path_output =  path_gen + 'output/'

        if (os.path.exists(self.path)==False):
            os.mkdir(self.path)
        if (os.path.exists(path_gen)==False):
            os.mkdir(path_gen)
        # if (os.path.exists(path_TB) == False):
        #     os.mkdir(path_TB)
        # if (os.path.exists(path_output) == False):
        #     os.mkdir(path_output)

        return path_gen,path_TB,path_output

    def _configure_dir(self,path):
        string_a = path.split('/')
        path = ''

        for string in string_a:
            if string != '':
                path += string+'/'

                if (os.path.exists(path) == False):
                    os.mkdir(path)





