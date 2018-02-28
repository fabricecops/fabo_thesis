from src.dst.helper.apply_mp import *
import cma
from src.dst.metrics.AUC import AUC
import functools
from keras.losses import mean_squared_error as mse
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


        print('df_false :',len(self.df_false))
        print('df_falsev :',len(self.df_false_val))
        print('df_true :',len(self.df_true))


        array_t                             = zip(list(self.df_true['data_y']),list(self.df_true['data_y_p']))
        self.df_true['error_m']             = list(map(self._get_error_m,array_t))
        array_t                             = zip(list(self.df_true['data_y']),list(self.df_true['data_y_p']))
        val_t                               = list(map(self._get_error_mean,array_t))
        self.df_true.drop(columns = ['data_y','data_X'],inplace = True)


        array_f                             = zip(list(self.df_false['data_y']),list(self.df_false['data_y_p']))
        train_f                             = list(map(self._get_error_mean,array_f))
        array_f                             = zip(list(self.df_false['data_y']),list(self.df_false['data_y_p']))
        self.df_false['error_m']            = list(map(self._get_error_m,array_f))
        self.df_false.drop(columns = ['data_y','data_X'],inplace = True)

        array_fv                            = zip(list(self.df_false_val['data_y']),list(self.df_false_val['data_y_p']))
        val_f                               = list(map(self._get_error_mean,array_fv))
        array_fv                            = zip(list(self.df_false_val['data_y']),list(self.df_false_val['data_y_p']))
        self.df_false_val['error_m']        = list(map(self._get_error_m,array_fv))
        self.df_false_val.drop(columns      = ['data_y','data_X'],inplace = True)


        result,AUC    = self._CMA_ES_()
        self.x        = result


        self.df_true['error_tm']      = np.array(list(map(functools.partial(self._get_error_max, x=result), np.array(self.df_true['error_m']))))
        self.df_false['error_tm']     = np.array(list(map(functools.partial(self._get_error_max, x=result), np.array(self.df_false['error_m']))))
        self.df_false_val['error_tm'] = np.array(list(map(functools.partial(self._get_error_max, x=result), np.array(self.df_false_val['error_m']))))




        loss_f_t     = np.mean(train_f)
        loss_f_v     = np.mean(val_f)
        loss_t       = np.mean(val_t)

        AUC_v,FPR_v,TPR_v = self._opt_function_val(result)

        loss_t_std      = np.std(val_t)
        loss_f_t_std   = np.std(train_f)
        loss_f_v_std   = np.std(val_f)

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
        print('lol')


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

        y     = row[0]
        y_p   = row[1]


        e_f = np.mean(np.power((y - y_p),2),axis=1)

        return e_f

    def _get_error_mean(self, row):

        y     = row[0]
        y_p   = row[1]

        e_f = np.mean(np.power((y - y_p), 2))



        return e_f


    def _get_error_max(self,e,x):

        eval_ = np.max(np.dot(e, x))
        return eval_