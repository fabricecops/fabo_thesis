from src.dst.helper.apply_mp import *
import cma
from src.dst.metrics.AUC import AUC
import functools

class CMA_ES(AUC):


    def __init__(self, dict_c,
                    ):

        AUC.__init__(self,dict_c)
        self.dict_c    = dict_c


        self.verbose    = dict_c['verbose_CMA']
        self.verbose_log= dict_c['verbose_CMA_log']

        self.evals     = dict_c['evals']
        self.bounds    = dict_c['bounds']
        self.sigma     = dict_c['sigma']

        self.AUC_max   = -100
        self.AUC_min   = 100
        self.FPR       = None
        self.TPR       = None

        self.x         = None
        self.dimension = None

    def main_CMA_ES(self,dict_data,epoch):

        self.df_true      = dict_data['df_true']
        self.df_false     = dict_data['df_false']
        self.df_false_val = dict_data['df_false_val']
        self.dimension    = self.df_true.iloc[0]['data_X'].shape[2]

        train_loss   = dict_data['losses']

        self.AUC_max   = 0
        self.AUC_min   = 100



        # print(self.df_true.columns)
        # print(self.df_false.columns)
        # print(self.df_false_val.columns)
        #
        print('df_false :',len(self.df_false))
        print('df_falsev :',len(self.df_false_val))
        print('df_true :',len(self.df_true))

        self.df_true      = apply_by_multiprocessing(self.df_true, self._get_error_m, axis=1, workers=4)
        self.df_true.drop(columns = ['data_y','data_X'],inplace = True)


        self.df_false     = apply_by_multiprocessing(self.df_false, self._get_error_m,  axis=1, workers=4)
        self.df_false.drop(columns = ['data_y','data_X'],inplace = True)



        self.df_false_val = apply_by_multiprocessing(self.df_false_val, self._get_error_m,  axis=1, workers=4)
        self.df_false_val.drop(columns = ['data_y','data_X'],inplace = True)




        result,AUC    = self._CMA_ES_()
        self.x        = result



        self.df_true = apply_by_multiprocessing(self.df_true, self._get_error_tm,x=result,  axis=1, workers=6)
        self.df_false = apply_by_multiprocessing(self.df_false, self._get_error_tm,x = result,  axis=1, workers=6)
        self.df_false_val = apply_by_multiprocessing(self.df_false_val, self._get_error_tm,x= result , axis=1, workers=6)

        val_t        = apply_by_multiprocessing(self.df_true, self._get_loss, axis=1, workers=6)['loss']
        val_f        = apply_by_multiprocessing(self.df_false_val, self._get_loss, axis=1, workers=6)['loss']


        loss_f_t     = np.mean(train_loss)
        loss_f_v     = np.mean(np.array(val_f))
        loss_t       = np.mean(np.array(val_t))




        AUC_v,FPR_v,TPR_v = self._opt_function_val(result)




        loss_t_std      = np.std(val_t)
        loss_f_t_std   = np.std(train_loss)
        loss_f_v_std   = np.std(val_f)

        # print(self.df_true.columns)
        # print(self.df_false.columns)
        # print(self.df_false_val.columns)

        dict_data   =  {'AUC_min'        : self.AUC_min,
                        'AUC_max'        : AUC,
                        'FPR'            : self.FPR,
                        'TPR'            : self.TPR,
                        'x'              : result,
                        'val_f'          : loss_f_v,
                        'val_t'          : loss_t,
                        'train_f'        : loss_f_t,
                        'train_std'      : loss_f_t_std,
                        'val_std_f'      : loss_f_v_std,
                        'val_std_t'      : loss_t_std,

                        'df_true'        : self.df_true,
                        'df_false'       : self.df_false,
                        'df_false_val'   : self.df_false_val,

                        'AUC_v'          : AUC_v,
                        'TPR_v'          : TPR_v,
                        'FPR_v'          : FPR_v
                        }


        return dict_data


    def _CMA_ES_(self):

        if(self.dict_c['stateful'] == False):
            result = self._CMA_ES_FE()

        else:
            result = self._CMA_ES_FE()

        return result

    def _CMA_ES_FE(self):


        es = cma.fmin(self._opt_function,
                      np.zeros(self.dimension),
                      self.sigma,
                      {'bounds'              : self.bounds,
                      'maxfevals'            : self.evals,
                       'verb_disp'           : self.verbose,
                       'verb_log'             :self.verbose_log })

        return es[0],-es[1]



    def _opt_function(self,x):

        eval_true = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_true['error_m']))))
        eval_false = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_false['error_m']))))


        AUC,FPR,TPR        = self.get_AUC_score(eval_true, eval_false)
        if(AUC>self.AUC_max):
            self.AUC_max = AUC
            self.FPR     = FPR
            self.TPR     = TPR

        if(AUC<self.AUC_min):
            self.AUC_min = AUC



        return -AUC

    def _opt_function_val(self,x):

        eval_true = np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_true['error_m']))))
        eval_false= np.array(list(map(functools.partial(self._get_error_max, x=x), np.array(self.df_false_val['error_m']))))

        AUC,FPR,TPR        = self.get_AUC_score(eval_true, eval_false)
        if(AUC>self.AUC_max):
            self.AUC_max = AUC
            self.FPR     = FPR
            self.TPR     = TPR

        if(AUC<self.AUC_min):
            self.AUC_min = AUC

        return AUC,FPR,TPR


    def _get_error_m(self, row):
        y   = row['data_y']
        y_p = row['data_y_p']


        e_f = np.square(y - y_p)
        row['error_m'] = np.mean(e_f,axis = 1)

        return row


    # def _get_error_f(self, row,x):
    #     y   = row['data_y']
    #     y_p = row['data_y_p']
    #
    #     e_f  = np.square(y - y_p)
    #     e_tm = np.max(np.mean(np.dot(e_f,x),axis = 1))
    #
    #
    #
    #     row['error_f'] = e_tm
    #
    #     return row

    def _get_error_tm(self,row,x):

        e_tm            = np.max(np.dot(row['error_m'],x))
        row['error_tm'] = e_tm

        return row



    def _get_loss(self,row):

        row['loss'] = np.mean(row['error_m'])

        return row


    def _get_error_max(self,e,x):

        eval_ = np.max(np.dot(e, x))
        return eval_