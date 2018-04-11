import os
from src.dst.outputhandler.pickle import pickle_save_,pickle_load
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import mixture

class model_selection():



    def __init__(self,dict_c):

        self.dict_c = dict_c


    def main(self):
        self._configure_dir(self.dict_c['path_save'])
        df       = pickle_load(self.dict_c['path_save']+self.dict_c['mode'],self.configure_data)
        df       = df[df['AUC_v']> 0.6]
        df,coeff,tsne_results = self.calculate_correlation(df)
        # df     = self.calculate_Kmeans(df,coeff,tsne_results)
        df       = self.calculate_GMM(df,coeff,tsne_results)
        df       = df.apply(self.apply_P_AUC_V,axis = 1)

        df_groups = pd.DataFrame(columns = df.columns)
        for group in df.groupby('clusters'):
            x   = np.argmax(np.array(group[1]['Score']))
            row = group[1].iloc[x]
            df_groups = df_groups.append(row)

        df_groups,coeff,tsne_results = self.calculate_correlation(df_groups)


        fig = plt.figure(figsize=(16, 4))

        plt.matshow(coeff)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.show()

        fig = plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(121)
        ax1 = sns.swarmplot(x='x-tsne', y='y-tsne', hue='clusters', data=df)

        ax2 = plt.subplot(122)
        df['clusters'].value_counts().plot(kind = 'bar')
        plt.show()



    def calculate_GMM(self,df,coeff,tsne_results):

        clf = mixture.BayesianGaussianMixture(n_components=self.dict_c['clusters'], covariance_type='full')
        clf.fit(coeff)

        clusters   = clf.predict(coeff)
        probas     = clf.predict_proba(coeff)

        df['clusters'] = clusters

        df['probas']=   [[] for x in range(len(df))]

        for i,c in enumerate(probas):
            df['probas'].iloc[i] = probas[i]




        return df

    def apply_P_AUC_V(self,row):

        row['Score'] = row['AUC_v']*row['probas'][row['clusters']]

        return row

    def calculate_Kmeans(self,df,coeff,tsne_results):


        kmeans = KMeans(n_clusters=self.dict_c['clusters'],n_init = self.dict_c['KM_n_init'])
        kmeans.fit(coeff)
        clusters = kmeans.predict(coeff)

        df['clusters'] = clusters


        kmeans = KMeans(n_clusters=self.dict_c['clusters'],n_init = self.dict_c['KM_n_init'])
        kmeans.fit(tsne_results)
        clusters = kmeans.predict(tsne_results)

        df['clusters_tsne'] = clusters


        sns.swarmplot(x='x-tsne', y='y-tsne', hue='clusters', data=df)
        plt.show()

        sns.swarmplot(x='x-tsne', y='y-tsne', hue='clusters_tsne', data=df)
        plt.show()

        return df

    def calculate_correlation(self,df):

        for i in range(len(df)):
            if(i==0):
                data = df['error_m'].iloc[i]
            else:
                data = np.vstack((data,df['error_m'].iloc[i]))


        coef         = np.corrcoef(data)
        tsne         = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(coef)

        df['coeff']=   [[] for x in range(len(df))]
        df['tsne']=   [[]  for x  in range(len(df))]

        for i,c in enumerate(coef):
            df['coeff'].iloc[i] = coef[i]
            df['tsne'].iloc[i]  = tsne_results[i]


        df['x-tsne'] = tsne_results[:,0]
        df['y-tsne'] = tsne_results[:,1]

        return df,coef,tsne_results

    def configure_data(self,*args):

        array = []
        for path in self.dict_c['path_a']:

            list_names = os.listdir(path)

            for name in list_names:
                try:
                    path_best = path+name+'/best/data_best.p'
                    data      = pickle_load(path_best,None)

                    error_    = list(data['df_f_train']['error_m'])
                    error_.extend(list(data['df_t_train']['error_m']))
                    error_    = np.array(error_)
                    dict_     = {
                                 'error_m': error_,
                                 'path'   : path+name+'/',
                                 'AUC_v'  : data['AUC_v']
                    }
                    array.append(dict_)
                except Exception as e:
                    pass

        df = pd.DataFrame(array, columns = ['error_m','path','AUC_v'])
        return df

    def _configure_dir(self,path):
        path = path
        string_a = path.split('/')
        path = ''

        for string in string_a:
            if string != '':
                path += string+'/'

                if (os.path.exists(path) == False):
                    os.mkdir(path)


if __name__ == '__main__':

    dict_c = {
                'path_save': './models/ensemble/',
                'mode'     : 'no_cma.p',
                'path_a'   : ['./models/bayes_opt/DEEP2/'],
                'clusters' : 10,
                'KM_n_init': 10,
    }

    MS = model_selection(dict_c).main()