import pickle
import os
import time
import shutil
import pandas as pd
import dill

def pickle_save(path, data):
    with open(path, 'wb') as handle:
        dill.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_save_(path, data):
    with open(path, 'wb') as handle:
        dill.dump(data, handle)
def return_conf_path(path, mode = 'pickle'):
    if(mode == 'pickle'):
        name = str(len(os.listdir(path))) + '.p'
    elif(mode == 'model'):
        name = str(len(os.listdir(path))) + '.h5'

    path = path + name

    return path

def diagnostics(data):

    try:
        print('dtype = ',data.dtype)
    except:
        pass

    try:
        print('type = ',type(data))
    except:
        pass

    try:
        print('len = ',len(data))
    except:
        pass


    try:
        print('keys = ',data.keys())
    except:
        pass

def tic():
    global time_
    time_ = time.time()

def toc():
    global time_
    tmp = time.time()

    elapsed = tmp - time_

    print('the elapsed time is: ', elapsed)

    return elapsed

def copy_data(src,dst):
    shutil.copy(src, dst)


def pickle_load(path,func,*args):

    try:
        data = dill.load(open(path, "rb"))
    except:
        data =  func(args)
        with open(path, 'wb') as handle:
            dill.dump(data, handle)

    return data

def json_load(path,func,*args):

    try:
        df = pd.read_json(path)
    except:
        df =  func(args)
        df.to_json(path)

    return df

def df_pickle_load(path,func,*args):
    try:
        df = pd.read_pickle(path)
    except:
        df =  func(args)
        df.to_pickle(path)

    return df
