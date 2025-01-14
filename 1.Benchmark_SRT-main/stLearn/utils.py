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


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    print("start mclust")
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
       new_type = refine_label(adata, radius, key='domain') #将聚类结果mclust赋值给domain后，对结果进行修正
       adata.obs['domain'] = new_type 
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values #原始的标签
    
    #calculate distance
    position = adata.obsm['spatial'] #获得每个spot点坐标
    distance = ot.dist(position, position, metric='euclidean') #计算spot-spot两两距离
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort() #返回的是数组值从小到大的索引值，即先对距离从小到大排序，然后返回它们的索引
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]]) #获得50个邻居的索引
        max_type = max(neigh_type, key=neigh_type.count) #获得邻居最多的类型
        new_type.append(max_type) #获得所有spot点的新label
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type #label变成str后返回

from DeepST.his_feat import image_feature, image_crop
from matplotlib.image import imread
def read_data(dataset, data_path='../../Dataset/'):

    if dataset == "Mouse_brain":
        adata = sq.datasets.visium_hne_adata()
        library_id='V1_Adult_Mouse_Brain'
        quality = 'hires'
        adata.uns["spatial"][library_id]["use_quality"] = quality  # quality='hires'
        adata.var_names_make_unique()
        adata.obs['ground_truth'] = adata.obs["cluster"]
        print("Mouse_brain的类别数",adata.obs['ground_truth'].unique())
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset) #please replace 'file_fold' with the download path
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5',
                               load_images=True)
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv',sep='\t',  index_col=0)
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'fine_annot_type'].values #fine_annot_type代替annot_type
        print("Breast_cancer的类别数\n", adata.obs['ground_truth'].value_counts())
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()
        # # 还有图片的数据进行处理，保存HE的CSV特征
        # library_id = list(adata.uns["spatial"].keys())[0] #'V1_Breast Cancer_Block A Section_1'
        # scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"] #获取放缩比：0.08250825
        # image_coor = adata.obsm["spatial"] * scale
        # adata.obs["imagecol"] = image_coor[:, 0] #由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        # adata.obs["imagerow"] = image_coor[:, 1]
        # adata.uns["spatial"][library_id]["use_quality"] = "hires" #给library_id的use_quality赋值"hires"
        # from pathlib import Path
        # save_path_image_crop = Path(os.path.join("../../Output/DeepST/temp/", 'Image_crop', f'{dataset}')) #获得tile位置
        # save_path_image_crop.mkdir(parents=True, exist_ok=True)
        # adata = image_crop(adata, save_path=save_path_image_crop) #切割tile
        # adata = image_feature(adata, pca_components=50, cnnType='ResNet50').extract_image_feat() #对这些tile进行切割，获得HE特征
        # print("HE未经PCA前的维度：", adata.obsm['image_feat'].shape)
        # HE_feature = pd.DataFrame(data=adata.obsm['image_feat'])  # X_tile_feature有2048维，X_morphology仅50维
        # HE_feature.to_csv('../../Output/DeepST/DeepST_HE_feature_1000.csv')



    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        # 读入原始数据
        # adata = sc.read(file_fold+"/STARmap_20180505_BY3_1k.h5ad") #(1207, 1020)
        # adata.var_names_make_unique()
        # df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt', sep='\t', index_col=0)
        # adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,'Annotation'].values
        adata = sc.read(file_fold + "/STARmap_1207_1020.h5ad")
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")  # (428, 19736)
        adata.obs['ground_truth']=adata.obs['Ground Truth']
        adata.obs['ground_truth'].value_counts().plot(kind='bar')
        plt.tight_layout()  # 调整画布在正中间
        plt.show()

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold +'/Adult_stereo.h5ad')
        print(adata.shape)  # (8243, 22144)
        adata.obs["ground_truth"]=adata.obs['Annotation'] #后面需要obs['Annotation'] 格式
        print("标签类别数：", len(adata.obs['Annotation'].unique()))
        #######   尝试对Stereo-seq数据进行HE特征提取  #######
        image_coor = adata.obsm["spatial"] #直接获取像素点位置
        adata.obs["imagecol"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        adata.obs["imagerow"] = image_coor[:, 1]
        adata.uns["spatial"] = dict() #因为adata没有.uns，所以手工生成
        library_id = 'Stereo'  # 构造
        adata.uns["spatial"][library_id] = dict()

        hires_image_file='../../Dataset/Stereo/Stereo.png'
        # a=cv2.imread(hires_image_file) # (18380,30565,3)
        # print("方式一读入：tif:",len(a),type(a))
        b=imread(hires_image_file) # imread（）只能读png，所以将tif转化为png
        # print("方式二读入：tif:", type(b),b.shape)

        adata.uns["spatial"][library_id]['images'] = dict()
        adata.uns["spatial"][library_id]['images']['hires']=b
        adata.uns["spatial"][library_id]["use_quality"] = "hires"  # 给library_id的use_quality赋值"hires"
        # from pathlib import Path
        # save_path_image_crop = Path(os.path.join("../../Output/DeepST/temp/", 'Image_crop', f'{dataset}'))  # 获得tile位置
        # save_path_image_crop.mkdir(parents=True, exist_ok=True)
        # adata = image_crop(adata, save_path=save_path_image_crop)  # 切割tile
        # adata = image_feature(adata, pca_components=50, cnnType='ResNet50').extract_image_feat()  # 对这些tile进行切割，获得HE特征
        # print("HE未经PCA前的维度：", adata.obsm['image_feat'].shape)

    if dataset == 'SeqFish':
        adata = sq.datasets.seqfish()
        # print('Seqfish.shape',adata.shape)  # (19416, 351)
        adata.obs['ground_truth'] = adata.obs['celltype_mapped_refined']
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset in ["Mouse_olfactory", "MOB_without_label"]:
        # please replace 'file_fold' with the download path
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold + '/filtered_feature_bc_matrix.h5ad')  # (3739,36601)
        adata.var_names_make_unique()

    if dataset == "Mouse_hippocampus":
        adata = sq.datasets.slideseqv2()
        adata.var_names_make_unique()
    return adata

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
