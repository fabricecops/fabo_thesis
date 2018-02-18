import GPyOpt

import Machine_learning.src.helper.helper as hf
import Machine_learning.src.models.tests.model.Conv_net as cv
import numpy as np

class BayesionOpt():

    def __init__(self,bounds,dict_p):
        self.bounds     = bounds
        self.dict_p     = dict_p

        self.len_space  = 30

    #### public functions   ##############

    def main(self):

        train = GPyOpt.methods.BayesianOptimization(f                      = self.opt_function,
                                                    domain                 = self.bounds,
                                                    maximize               = self.dict_p['maximize'],
                                                    initial_design_numdata = self.dict_p['initial_n'],
                                                    initial_design_type    = self.dict_p['initial_dt'],
                                                    eps                    = self.dict_p['eps']

                                                    )


        train.run_optimization(max_iter=self.dict_p['max_iter'])

        print("optimized parameters: {0}".format(train.x_opt))
        print("optimized loss: {0}".format(train.fx_opt))

    #### private functions   ################

    def opt_function(self,x):
        try:
            hf.tic()
            self.configure_bounds(x)

            model       = cv.Conv_net(self.dict_p)
            hist,count  = model.fit_generator()


            print()
            hf.toc()
            val  = max(hist['val_acc'])

            # if(val> self.dict_p['min_acc']):
            #     opt_value = count/val
            #
            # else:
            #     opt_value = count/val + 10000

            print()
            print('min val is: ',val)
            print('mean t  is: ',count)
            print('optimal value is: ',val)
            print()

            return val

        except Exception as e:
            print('XXXXXXXXXXXXXXXXXxx')
            print(e)
            print('XXXXXXXXXXXXXXXXXXX')
            opt_value = 10000000.

            return opt_value

    def configure_bounds(self,x):

        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print()

        array_filters = []

        for i,bound in enumerate(self.bounds):


            key   = bound['name']
            type_ = bound['type']

            boolean       = True

            if('filter' not in key):

                if(type_ == 'continuous'):
                    self.dict_p[key] = float(x[:,i])

                elif(type_ == 'discrete'):
                    self.dict_p[key] = int(x[:,i])


                len_space = self.len_space-len(key)
                string    = key+' '*len_space+': '
                print(string+str(self.dict_p[key]))

            else:

                if('filter_o' in key):


                    if(x[:,i]==0):
                        boolean = False

                    if(boolean == True):
                        tuple_ = (int(x[:,i]),
                                  int(x[:,i+1]),
                                  int(x[:,i+1]),
                                  int(x[:,i+2]),
                                  'relu')

                        len_space = self.len_space - len('filter')
                        string = 'filter' + ' ' * len_space + ': ',tuple_
                        print(string)

                        array_filters.append(tuple_)

        self.dict_p['filters'] = array_filters

        print()
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

