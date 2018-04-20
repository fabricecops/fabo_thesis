from src.models.ensemble.data_manager import data_manager

from src.models.ensemble.CMA_ES import CMA_ES

import os
class ensemble(data_manager):

    def __init__(self,dict_c):
        self.dict_c = dict_c
        self._configure_dir(self.dict_c['path_save'])

        data_manager.__init__(self,dict_c)


    def main(self):
        data    = self.main_DM()


        if(self.dict_c['mean'] == False):
            CMA_ES_ = CMA_ES(self.dict_c).main_CMA_ES(data)
        else:
            CMA_ES_ = CMA_ES(self.dict_c).main_MEAN(data)





    def _configure_dir(self,path):
        path = path+'/best'
        string_a = path.split('/')
        path = ''

        for string in string_a:
            if string != '':
                path += string+'/'

                if (os.path.exists(path) == False):
                    os.mkdir(path)



    def run_CMA_ES(self,dict_c,data):
        cma  = CMA_ES(dict_c)
        data = cma.main(data)

        return data




