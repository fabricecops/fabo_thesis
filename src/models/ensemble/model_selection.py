import os
from src.dst.outputhandler.pickle import pickle_load
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import mixture
import matplotlib.pyplot as plt

class model_selection():

    def __init__(self,dict_c):
        self.dict_c = dict_c

    def main(self,df):
        df                    = df[df['AUC_v']> self.dict_c['threshold']].reset_index()
        df,coeff,tsne_results = self.calculate_correlation(df)
        df                    = self.calculate_Kmeans(df,coeff,tsne_results)


        # df_groups, coef = self.pick_lowest_corr(df,coeff,self.dict_c['clusters'])
        # df_random       = df.sample(self.dict_c['clusters'])

        return df

    def plot_matrix(self):

        df,df_groups,df_random, coef = self.main()

        fig = plt.figure(figsize=(16, 4))

        plt.matshow(coef,vmin= 0.,vmax= 1.)
        plt.title('Correlation matrix')
        plt.colorbar()
        plt.savefig('./plots/pic/matrix_'+str(self.dict_c['clusters'])+str(self.dict_c['threshold'])+'.png')

        plt.show()

    def pick_lowest_corr(self,df,coeff,clusters):
        ids     = [df['AUC_v'].idxmax()]
        array  = coeff[ids[0]].reshape(1,-1)

        for i in range(1,clusters):
            tmp   = np.sum(array,axis=0)

            for id in ids:
                tmp[id] += 100
            id    = np.argmin(tmp)
            ids.append(id)
            array = np.vstack((coeff[id],array))

        df_groups = pd.DataFrame(columns = df.columns)
        for group in ids:
            row = df.iloc[group]
            df_groups = df_groups.append(row)

        for i in range(len(df_groups)):
            if (i == 0):
                data = df['error_m'].iloc[i]
            else:
                data = np.vstack((data, df['error_m'].iloc[i]))

        coef = np.corrcoef(data)


        return df_groups,coef

    def calculate_GMM(self,df,coeff,tsne_results):

        clf = mixture.BayesianGaussianMixture(n_components=self.dict_c['clusters'], covariance_type='full')
        clf.fit(coeff)

        clusters   = clf.predict(coeff)
        probas     = clf.predict_proba(coeff)

        df['clusters'] = clusters

        df['probas']=   [[] for x in range(len(df))]

        for i,c in enumerate(probas):
            df['probas'].iloc[i] = probas[i]


        df       = df.apply(self.apply_P_AUC_V,axis = 1)

        df_groups = pd.DataFrame(columns = df.columns)
        for group in df.groupby('clusters'):
            x   = np.argmax(np.array(group[1]['Score']))
            row = group[1].iloc[x]
            df_groups = df_groups.append(row)

        df_groups,coeff,tsne_results = self.calculate_correlation(df_groups)
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



if __name__ == '__main__':

    dict_c = {
                'path_save': './models/ensemble/',
                'mode'     : 'no_cma.p',
                'path_a'   : ['./models/bayes_opt/DEEP2/','./models/bayes_opt/DEEP3/'],
                'clusters' : 2,
                'KM_n_init': 10,
                'threshold': 0.6,
    }

    MS = model_selection(dict_c).main()