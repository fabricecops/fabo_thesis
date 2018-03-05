import os

import src.data.dimensionality_reduction.HCF.preprocessing as pp
from src.dst.outputhandler.pickle import pickle_load,pickle_save
from src.models.LSTM.configure import return_dict_bounds
from sklearn.preprocessing import MinMaxScaler
from src.dst.helper.apply_mp import *


class pipe_line_data():

    def __init__(self,dict_c):

        self.dict_c     = dict_c
        self.path       = dict_c['path_data']
        self.list_names = os.listdir(self.path)

        self.scaler_p   = None
        self.scaler_v   = None

    def peak_derivation(self,*args):
        path_pd,path_sc_p,path_sc_v,path = self.return_path_pd(self.dict_c)


        #### get dataframe ############
        GD     = pp.get_df(self.path)
        path_o = './data/processed/df/df_1.p'
        df     = pickle_load(path_o,GD.get_df_data, *self.list_names)


        ### movie pictures to interim stage
        print('MOVING PICTURES')
        # MV = pp.Move_p()
        # df = MV.move_p    ictures(df)

        # apply background subtraction
        print('BGS')
        # bgs = pp.BGS(self.dict_c)
        # bgs.main(df)


        # peak derivation
        print('Peak derivation')
        PP                = pp.path_Generation(df, self.dict_c )
        df                = pickle_load(path_pd,PP.main,())


        ## Train scaler ####
        print('Scaler')
        scaler           = pp.scaler()
        self.scaler_p    = pickle_load(path_sc_p,scaler._train_scaler, df,'data_p')
        self.scaler_v    = pickle_load(path_sc_v,scaler._train_scaler, df,'data_v')
        df               = scaler.main(df,self.scaler_p,self.scaler_v)


        print('PCA')
        ## PCA          ####
        PCA_mod    = pp.PCA_()
        df         = PCA_mod.main(df,path)


        return df



    def return_path_pd(self,dict_c):
        path_o = './data/processed/df/df_pd/df'

        TH     = '_T_'+str(dict_c['threshold'])
        A      = '_A_'+str(dict_c['area'])
        C      = '_C_'+str(dict_c['nr_contours'])
        R      = '_R_'+str(dict_c['resolution'])

        path = path_o + TH + A + C + R

        if (os.path.exists(path)==False):
            os.mkdir(path)

        path_df   = path + '/df_pd.p'

        path_sc_v = path + '/sc_v.p'
        path_sc_p = path + '/sc_p.p'




        return path_df,path_sc_v,path_sc_p,path


if __name__ == '__main__':

    dict_c,_   = return_dict_bounds()
    path       = './data/raw/configured_raw/'
    list_names = os.listdir(path)

    PL = pipe_line_data(dict_c)
    df = PL.peak_derivation(path,list_names)

