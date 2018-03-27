from src.models.ensemble.data_manager import data_manager
from src.models.ensemble.config import return_dict
from src.dst.keras_model.FS_manager import FS_manager

class ensemble(data_manager,FS_manager):

    def __init__(self,dict_c):

        self.dict_c = dict_c
        FS_manager.__init__(self,dict_c['path_o'])
        data_manager.__init__(self,dict_c)


    def mean(self):
        pass

    def CMA_ES(self):
        pass


if __name__ == '__main__':
    dict_c = return_dict()
    ensemble(dict_c)