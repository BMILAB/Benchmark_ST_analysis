import time

import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import squidpy as sq
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score



def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    print("开始clust聚类")
    start=time.time()
    np.random.seed(random_seed)
    import rpy2.robjects as rp_robjects  ###因为之前rpy2.robjects as robjects,所以后面再调用robjects混淆了是rpy2.robjects中的，还是后面的
    rp_robjects.r.library("mclust") ###导入R语言中mclust数据库
   # print("type(rmclust)",type(rmclust)) # <class 'rpy2.robjects.vectors.StrVector'>
    ##robjects.r.source(‘XX.R’)可调用R功能函数

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate() #将numpy转化为rpy2
    r_random_seed = rp_robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust11 = rp_robjects.r['Mclust'] #调用r中Mclust对象
   # print(type(rmclust11)) #<class 'rpy2.robjects.functions.SignatureTranslatedFunction'>
    a=adata.obsm[used_obsm] #获得嵌入的数组形式
    #used_obsm为obsm中的emb_pca
    a=rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]) #将嵌入数组转化为rpy2
    res = rmclust11(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames) #聚类，结果返回一个list
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category') #将结果保存为category形式
    end=time.time()
    print("clust聚类完成！,聚类时间为：",end-start)
    return adata

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """
    
    pca = PCA(n_components=20, random_state=42)  #将原本64维的嵌入降到20维
    embedding = pca.fit_transform(adata.obsm['emb'].copy()) #训练得到的嵌入(4226,20)
    adata.obsm['emb_pca'] = embedding #将训练结果保存在obsm['emb_pca']
    
    if method == 'mclust': #将嵌入降维后，进行聚类
       adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:  
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]
    return new_type



def read_data(dataset, data_path='../../Dataset/'):
    if dataset.startswith('15'):  # DLPFC dataset
        print("load DLPFC dataset:")
        file_fold = f"{data_path}DLPFC/{dataset}/"
        raw = sc.read_visium(path=f'../../Dataset/DLPFC/{dataset}/', count_file='filtered_feature_bc_matrix.h5')
        raw.var_names_make_unique()
        Ann_df = pd.read_csv(f'../../Dataset/DLPFC/{dataset}/metadata.tsv', sep='\t')
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        raw.obs['ground_truth'] = Ann_df.loc[raw.obs_names, 'Ground Truth']
        n_clusters = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7

    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset)
        raw = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        raw.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
        raw.obs['ground_truth'] = df_meta.loc[
            raw.obs_names, 'fine_annot_type'].values
        print("Breast_cancer clustering type", len(raw.obs['ground_truth'].unique()))
        n_clusters = 20

    if dataset == "Mouse_brain":
        raw = sq.datasets.visium_hne_adata()
        raw.var_names_make_unique()
        raw.obs['ground_truth'] = raw.obs["cluster"]
        print(f"Mouse_brain shape：{raw.shape}，clustering type：{len(raw.obs['ground_truth'].unique())}")

        n_clusters = 15

    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")  # (428, 19736)
        raw.obs['ground_truth'] = raw.obs['Ground Truth']
        print("PDAC clustering type", raw.obs['ground_truth'].unique())
        n_clusters = 4

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + '/Adult_stereo.h5ad')
        print(raw.shape)  # (8243, 22144)
        raw.obs["ground_truth"] = raw.obs['Annotation']
        print("Stereo clustering type：", len(raw.obs['Annotation'].unique()))
        n_clusters = 16

    if dataset == 'SeqFish':
        raw = sq.datasets.seqfish()
        raw.obs['ground_truth'] = raw.obs['celltype_mapped_refined']
        print("SeqFish clustering type：", len(raw.obs['ground_truth'].unique()))
        n_clusters = 22

        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/STARmap_1207_1020.h5ad")
        print(f"STARmap shape：{raw.shape}，clustering type：{len(raw.obs['ground_truth'].unique())}")  # ,raw.obs['ground_truth'].unique())
        n_clusters = 16

    return raw, n_clusters

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path
    
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    
