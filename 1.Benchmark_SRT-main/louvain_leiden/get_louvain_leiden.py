from scipy.sparse import issparse
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
import sys
import os
import pandas as pd
from sklearn import metrics
import anndata as ad
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder

sys.path.append('../') #../表上一级，所以可以引用util_for_all
import utils_for_all as usa

def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        # label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        # completeness = completeness_score(label_df["True"], label_df["Pred"])
        # hm = homogeneity_score(label_df["True"], label_df["Pred"])
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami=adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
    return  ari,nmi,ami



def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)     #等价于pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum()) #提供了前n个主成分（n由PCA给定）解释的总方差的比例，可以帮助我们了解数据集中主成分所保留的信息量
    return X_PC


from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
def Kmeans_cluster(X_embedding, n_clusters, merge=False):
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(X_embedding)
    # merge clusters with less than 3 cells
    # if merge:
    #     cluster_labels = merge_cluser(X_embedding, cluster_labels)
    score = metrics.silhouette_score(X_embedding, cluster_labels, metric='euclidean')
    return cluster_labels, score


def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.02):
    if cluster_type == 'leiden':
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == fixed_clus_count:
                break
    elif cluster_type == 'louvain':
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            if count_unique_louvain == fixed_clus_count:
                break
    return res



Dataset1 = ['151507', '151508', '151509', '151510', '151669', '151670',
            '151671', '151672', '151673', '151674', '151675', '151676']
Dataset2=["Mouse_brain","Breast_cancer","PDAC","SeqFish","Stereo","STARmap"] #
cluster_method='leiden'
results_cluster = pd.DataFrame()

for dataset in Dataset2:
    print(f"==============================running data is：{dataset}=======================")
    save_data_path = f'../../Output/{cluster_method}/'
    if dataset.startswith('15'):
        save_data_path = f'../../Output/{cluster_method}/DLPFC/'
    else:
        save_data_path = f'../../Output/{cluster_method}/'

    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    adata, n_clusters = usa.get_adata(dataset, data_path='../../Dataset/')
    le = LabelEncoder()
    ground_truth_le = le.fit_transform(list(adata.obs["ground_truth"].values))

    adata.X = adata.X.A if issparse(adata.X) else adata.X
    X_embedding = adata.X
    print(f'begin {cluster_method} cluster')
    if cluster_method == 'kmeans':
        X_embedding = PCA_process(X_embedding, nps=30)
        #X_data_PCA = PCA_process(X_data, nps=X_embedding.shape[1])
        # concate
        #X_embedding = np.concatenate((X_embedding, X_data), axis=1)
        print('Shape of data to cluster:', X_embedding.shape)
        cluster_labels, score = Kmeans_cluster(X_embedding, n_clusters)
    else:
        sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50) # 20
        eval_resolution = res_search_fixed_clus(cluster_method, adata, n_clusters)
        if cluster_method == 'leiden':
            sc.tl.leiden(adata, key_added="leiden", resolution=eval_resolution)
            cluster_labels = np.array(adata.obs['leiden'])
        if cluster_method == 'louvain':
            sc.tl.louvain(adata, key_added="louvain", resolution=eval_resolution)
            cluster_labels = np.array(adata.obs['louvain'])

        # adata.write(results_file)
        cluster_labels = [ int(x) for x in cluster_labels ]
    predict_df=pd.DataFrame()
    predict_df['spot']=adata.obs_names.values
    predict_df['ground_truth'] = adata.obs['ground_truth'].values
    predict_df[cluster_method]= cluster_labels

    ari, nmi, ami = eval_model(predict_df[cluster_method], ground_truth_le) #['ground_truth']为字符注释，也可以计算
    SC = silhouette_score(X_embedding, predict_df[cluster_method])
    if dataset.startswith('15'):
        SC_revise=0
    else:
        SC_revise = silhouette_score(X_embedding, ground_truth_le)
    print(f"{dataset}'s ari,nmi,ami,SC",ari,nmi,ami,SC)

    res = {}
    res['method'] = cluster_method
    res["dataset"] = dataset
    res["ari"] = ari
    res["nmi"] = nmi
    res["ami"] = ami
    res["sc"] = SC
    res["sc_revise"] = SC_revise
    results_cluster = results_cluster._append(res, ignore_index=True)
print( results_cluster.head())
results_cluster.to_csv(f'{save_data_path}predict_result.csv', header=True)



