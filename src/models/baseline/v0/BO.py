from src.models.baseline.v0.configure import return_dict_bounds
from src.models.baseline.v0.baseline import baseline
from src.dst.outputhandler.pickle import pickle_save_
import numpy as np
import gpflowopt
import gpflow
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os

class BO():

    def __init__(self,dict_c):
        self.dict_c = dict_c
        self.domain = None
        self._configure()


    def optimization(self):

        design = gpflowopt.design.LatinHyperCube(self.dict_c['initial_n'], self.domain)

        X = design.generate()
        Y = self.opt_function(X)

        objective_models = [gpflow.gpr.GPR(X.copy(), Y[:, [i]].copy(), gpflow.kernels.Matern52(2, ARD=True)) for i in
                            range(Y.shape[1])]
        for model in objective_models:
            model.likelihood.variance = 0.01

        hvpoi = gpflowopt.acquisition.HVProbabilityOfImprovement(objective_models)

        # First setup the optimization strategy for the acquisition function
        # Combining MC step followed by L-BFGS-B
        acquisition_opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(self.domain, 1000),
                                                           gpflowopt.optim.SciPyOptimizer(self.domain)])

        # Then run the BayesianOptimizer for 20 iterations
        optimizer = gpflowopt.BayesianOptimizer(self.domain, hvpoi, optimizer=acquisition_opt)
        # with optimizer.silent():
        result = optimizer.optimize([self.opt_function], n_iter=self.dict_c['nr_iter'])



        self.plot(hvpoi)

    def opt_function(self,X):
        for i,x in enumerate(X):

            self._configure_dict_c(x)
            BL                 = baseline(self.dict_c)
            y                  = BL.main()


            y = np.array([y[0], y[1]]).reshape((1, 2))


            if(i == 0):
                array_y = y
            else:
                array_y = np.vstack((array_y,y))




        return array_y

    def _configure(self):

        path = self.dict_c['path_save']
        string = 'experiment_' + str(len(os.listdir(path)))
        path = path + string
        self.path = path
        if (os.path.exists(path) == False):
            os.mkdir(path)

        path_d = path + '/data'
        if (os.path.exists(path_d) == False):
            os.mkdir(path_d)

        ### sigma_CMA
        self.domain = gpflowopt.domain.ContinuousParameter('s_CMA',0.1,3)
        ### time dim
        self.domain += gpflowopt.domain.ContinuousParameter('TD',2,20)
        ### max h
        self.domain+= gpflowopt.domain.ContinuousParameter('MH',100,250)
        ### min h
        self.domain+= gpflowopt.domain.ContinuousParameter('MiH',0,100)
        ### resolution
        self.domain+=gpflowopt.domain.ContinuousParameter('R',1,10)
        ### nr_contours
        self.domain+=gpflowopt.domain.ContinuousParameter('nr_C',1,6)
        ### threshold
        self.domain+=gpflowopt.domain.ContinuousParameter('threshold',100,500)
        ### area
        self.domain+=gpflowopt.domain.ContinuousParameter('A',100,500)
        ### pos
        self.domain +=gpflowopt.domain.ContinuousParameter('V',0.,1.)
        ### speed
        self.domain +=gpflowopt.domain.ContinuousParameter('V',0.,1.)
        ### PCA
        self.domain +=gpflowopt.domain.ContinuousParameter('PCA',0.,1.)


    def _configure_dict_c(self,x):

        self.dict_c['sigma']         = x[0]
        self.dict_c['time_dim']      = int(round(x[1]))
        self.dict_c['max_h']         = int(round(x[2]))
        self.dict_c['min_h']         = int(round(x[3]))
        self.dict_c['resolution']    = int(round(x[4]))
        self.dict_c['nr_contours']   = int(round(x[5]))
        self.dict_c['threshold']     = int(round(x[6]))
        self.dict_c['area']          = int(round(x[7]))



        array = []

        if(round(x[8])== 1):
            array.append('p')
        if(round(x[9])== 1):
            array.append('v')
        if(round(x[10])== 1):
            array.append('PCA')

        self.dict_c['mode_data'] = array



        print('SIGMA = ',self.dict_c['sigma'])
        print('time_dim = ',self.dict_c['time_dim'])
        print('min_h = ',self.dict_c['min_h'])
        print('max_h = ',self.dict_c['max_h'])
        print('resolution = ',self.dict_c['resolution'])
        print('nr_contours = ',self.dict_c['nr_contours'])
        print('threshold = ',self.dict_c['threshold'])
        print('area = ',self.dict_c['area'])
        print('attributes = ',self.dict_c['mode_data'])

    def plot(self,hvpoi):
        # plot pareto front
        plt.figure(figsize=(9, 4))

        R = np.array([1.5, 1.5])
        print('R:', R)
        hv = hvpoi.pareto.hypervolume(R)
        print('Hypervolume indicator:', hv)

        plt.figure(figsize=(7, 7))

        pf, dom = gpflowopt.pareto.non_dominated_sort(hvpoi.data[1])

        plt.scatter(hvpoi.data[1][:, 0], hvpoi.data[1][:, 1], c=dom)
        plt.title('Pareto set')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.savefig(self.path+'/paretofront.png')
        pickle_save_(self.path+'/hvpoi.p',hvpoi)



if __name__ == '__main__':

    dict_c, _ = return_dict_bounds()
    BL = BO(dict_c).optimization()