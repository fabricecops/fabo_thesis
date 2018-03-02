import GPyOpt
import src.helper.helper as hf

import src.models.LSTM.model_a.s2s as s2s


class BayesionOpt():

    def __init__(self,bounds,dict_c):
        self.bounds     = bounds
        self.dict_c  = dict_c

    def main(self):

        train = GPyOpt.methods.BayesianOptimization(f                      = self._opt_function,
                                                    domain                 = self.bounds,
                                                    maximize               = self.dict_c['maximize'],
                                                    initial_design_numdata = self.dict_c['initial_n'],
                                                    initial_design_type    = self.dict_c['initial_dt'],
                                                    eps                    = self.dict_c['eps']

                                                    )


        train.run_optimization(max_iter=self.dict_c['max_iter'])

        print("optimized parameters: {0}".format(train.x_opt))
        print("optimized loss: {0}".format(train.fx_opt))

    def _opt_function(self,x):


        hf.tic()


        model      = s2s.LSTM_(self.dict_c)
        hist       = model.fit()




        history    = hist.history

        data_true  = model.configure_data_movie(True)
        data_false = model.configure_data_movie(False)

        _,ev_true  = model.predict(data_true)
        _,ev_false = model.predict(data_false)

        AUC,TN,FN  = model.calc_AUC(ev_false,ev_true)

        print('The area under the curve = ',AUC)
        elapsed    = hf.toc()


        data       = {
                        'AUC'      : AUC,
                        'TN'       : TN,
                        'FN'       : FN,
                        'hist'     : history,
                        'setting'  : self.dict_c,
                        'elapsed'  : elapsed,
                        'ev_true'  : ev_true,
                        'ev_false' : ev_false,
        }

        path_h     = hf.return_conf_path(self.dict_c['save_hist_dir'], mode='pickle')
        hf.pickle_save(path_h, data)

        model_k    = model.return_model()
        path_m     = hf.return_conf_path(self.dict_c['save_model'], mode = 'model')
        model_k.save(path_m)

        return AUC

    def _configure_bounds(self, x):
            print_bounds_a = []

            for i, bound in enumerate(self.bounds):

                key   = bound['name']
                type_ = bound['type']

                if (type_ == 'continuous'):
                    self.dict_c[key] = float(x[:, i])

                elif (type_ == 'discrete'):
                    self.dict_c[key] = int(x[:, i])


                len_space = 20 - len(key)
                string    = key + ' ' * len_space + ': '+str(self.dict_p[key])
                print_bounds_a.append(string)

                self._print_bounds(print_bounds_a)

    def _print_bounds(self,print_bounds_a):

        if(self.dict_c['print_BO_bounds'] == True):
            print('XXXXXXXXXXX BOUNDS XXXXXXXXXXXXXXX')
            print()
            for bound in print_bounds_a:
                print(bound)
            print()
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')







