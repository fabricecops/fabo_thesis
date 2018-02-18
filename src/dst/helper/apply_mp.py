import pandas as pd

from pathos.multiprocessing import Pool
import numpy as np


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = Pool(processes=workers)



    result = pool.map(_apply_df, [(d, func, kwargs)
                                  for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))

from multiprocessing import Process,Queue

class MP():

    def __init__(self):
        self.Queue_Terminate = Queue()
        self.Queue_Start     = Queue()

    def spawn(self,func,*args):

        self.t = Process(target=func, args=args)
        self.t.daemon = True
        self.t.start()


    def put_start(self):
        self.Queue_Start.put('START')

    def put_terminate(self):
        self.Queue_Terminate.put('STOP')

    def get_start(self):
        self.Queue_Start.get()

    def get_terminate(self):
        self.Queue_Terminate.get()

    def terminate_process(self):
        self.t.terminate()

