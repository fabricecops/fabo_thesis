import numpy as np
from sklearn import metrics
import functools

class AUC():

    def __init__(self,dict_c):
        self.resolution_AUC = dict_c['resolution_AUC']

        self.min_th = None
        self.max_th = None

    def get_AUC_score(self,eval_true,eval_false):
        # self.min_th, self.max_th, eval_true = self.configure_bounds(eval_true, eval_false)

        thresholds = np.linspace(-0.005, 1.1, self.resolution_AUC)

        min_       = min(min(eval_true),min(eval_false))
        max_       = max(max(eval_true),max(eval_false))

        eval_false = ((eval_false - min_)/(max_-min_+0.000001))
        eval_true  = ((eval_true  - min_)/(max_-min_+0.000001))


        eval_false = np.array(sorted(eval_false))
        eval_true  = np.array(sorted(eval_true))

        FPR        = sorted(list(map(functools.partial(self._calc_rates, eval_=eval_false), thresholds)))
        TPR        = sorted(list(map(functools.partial(self._calc_rates, eval_=eval_true), thresholds)))

        AUC = metrics.auc(FPR,TPR)

        return AUC, FPR, TPR

    def configure_bounds(self, eval_true, eval_false):
        min_ = np.min(eval_false)
        max_ = np.max(eval_false) + 0.5

        mask_true = eval_true > max_
        mask_false = eval_true < max_
        eval_true_n = mask_true * (max_ + 0.5) + np.multiply(mask_false, eval_true)

        return min_, max_, eval_true_n

    def _calc_rates(self,threshold,eval_ = None):

        rate = np.sum((eval_ > threshold))/ float(len(eval_))

        return rate


class MSE():

    def __init__(self):
        pass




