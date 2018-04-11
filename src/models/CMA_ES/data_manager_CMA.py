import os

from src.dst.outputhandler.pickle import pickle_save_,pickle_load

class data_manager():


    def __init__(self,dict_c):

        self.dict_c = dict_c

    def configure_data(self,dict_c):
        data,path,array_dict      = self.load_data(dict_c['path_i'])
        data,path,dict_           = self.pick_best(data,path,array_dict)


        return data,path,dict_




    def pick_best(self,data,path,dicts):

        max_AUC = 0.
        for model,path_,dict_ in zip(data,path,dicts):
            if(model['AUC_v']> max_AUC):
                best_model = model
                max_AUC    = model['AUC_v']
                best_path  = path_
                best_dict  = dict_



        return best_model,best_path,best_dict



    def load_data(self,path):
        list_names = os.listdir(path)
        array_data = []
        array_path = []

        array_dict = []

        for name in list_names:
            try:
                path_best = path+name+'/best/data_best.p'
                path_app  = path+name
                path_app  = path_app.replace('bayes_opt','CMA_ES')
                data      = pickle_load(path_best,None)
                dict_c    = pickle_load(path+name+'/dict.p',None)
                array_data.append(data)
                array_path.append(path_app)
                array_dict.append(dict_c)

            except Exception as e:
                print(e)
        return array_data,array_path,array_dict
    def _configure_dir(self,path):
        path = path+'/best'
        string_a = path.split('/')
        path = ''

        for string in string_a:
            if string != '':
                path += string+'/'

                if (os.path.exists(path) == False):
                    os.mkdir(path)


if __name__ == '__main__':
    def return_dict():
        dict_c = {
            'path_i': './models/bayes_opt/DEEP1/',
            'path_o': './models/ensemble/ensemble/',

            'resolution_AUC': 1000,

            ###### CMA_ES    ######
            'CMA_ES': True,
            'verbose_CMA': 1,
            'verbose_CMA_log': 0,
            'evals': 10000,
            'bounds': [-100, 100.],
            'sigma': 0.4222222222222225,
            'progress_ST': 0.3,

            'epoch': 0

        }

        return dict_c

    dict_c = return_dict()
    data_manager(dict_c).configure_data(dict_c)