import os

import src.data.dimensionality_reduction.HCF.preprocessing as pp
from src.dst.outputhandler.pickle import pickle_load,pickle_save
from src.models.LSTM.configure import return_dict_bounds


class pipe_line_data():

    def __init__(self,dict_c):

        self.dict_c     = dict_c
        self.path       = dict_c['path_data']
        self.list_names = os.listdir(self.path)

    def peak_derivation(self,*args):

        # ### get dataframe ############
        GD     = pp.get_df(self.path)
        path_o = './data/processed/df/df_1.p'
        df     = pickle_load(path_o,GD.get_df_data, *self.list_names).iloc[0:50]


        # # # movie pictures to interim stage
        print('MOVING PICTURES')
        # MV = pp.Move_p()
        # df = MV.move_p    ictures(df)

        # apply background subtraction
        print('BGS')
        # bgs = pp.BGS(self.dict_c)
        # bgs.main(df)

        # get min_max heigth
        # path_i = './data/interim/BGS/bit_8/'
        # path_o = './data/processed/stats/minmax.json'
        # min_max = pp.get_min_max_h(path_i,path_o).main()

        # peak derivation
        print('Peak derivation')
        PP         = pp.path_Generation(df, self.dict_c )
        path,_,_   = self.return_path_pd(self.dict_c)
        df         = pickle_load(path,PP.main,())

        print('XXXXXXXXXXXXXXXXXXXXXXXXX')
        print(len(df))

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




        return path_df,path_sc_v,path_sc_p


if __name__ == '__main__':

    dict_c,_   = return_dict_bounds()
    path       = './data/raw/configured_raw/'
    list_names = os.listdir(path)

    PL = pipe_line_data(dict_c)
    df = PL.peak_derivation(path,list_names)

