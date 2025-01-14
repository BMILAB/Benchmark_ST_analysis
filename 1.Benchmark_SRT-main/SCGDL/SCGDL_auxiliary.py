import pandas as pd
import numpy as np
import sklearn.neighbors
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
import squidpy as sq
import scanpy as sc
import os

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def Adata2Torch_data(adata): 
    G_df = adata.uns['Spatial_Net'].copy() 
    spots = np.array(adata.obs_names) 
    spots_id_tran = dict(zip(spots, range(spots.shape[0]))) 
    G_df['Spot1'] = G_df['Spot1'].map(spots_id_tran) 
    G_df['Spot2'] = G_df['Spot2'].map(spots_id_tran) 

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Spot1'], G_df['Spot2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G) 
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  
    return data

def Spatial_Dis_Cal(adata, rad_dis=None, knn_dis=None, model='Radius', verbose=True):
    """\
    Calculate the spatial neighbor networks, as the distance between two spots.
    Parameters
    ----------
    adata:  AnnData object of scanpy package.
    rad_dis:  radius distance when model='Radius' 
    knn_dis:  The number of nearest neighbors when model='KNN'
    model:
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_dis. 
        When model=='KNN', the spot is connected to its first knn_dis nearest neighbors.
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    assert(model in ['Radius', 'KNN']) 
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial']) 
    coor.index = adata.obs.index 
    coor.columns = ['Spatial_X', 'Spatial_Y'] 

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_dis).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices[spot].shape[0], indices[spot], distances[spot]))) 
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=knn_dis+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices.shape[1],indices[spot,:], distances[spot,:])))

    KNN_df = pd.concat(KNN_list) 
    KNN_df.columns = ['Spot1', 'Spot2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_spot_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), )) 
    Spatial_Net['Spot1'] = Spatial_Net['Spot1'].map(id_spot_trans) 
    Spatial_Net['Spot2'] = Spatial_Net['Spot2'].map(id_spot_trans) 
    if verbose:
        print('The graph contains %d edges, %d spots.' %(Spatial_Net.shape[0], adata.n_obs)) 
        print('%.4f neighbors per spot on average.' %(Spatial_Net.shape[0]/adata.n_obs)) 

    adata.uns['Spatial_Net'] = Spatial_Net

def Spatial_Dis_Draw(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Spot1'].shape[0] 
    Mean_edge = Num_edge/adata.shape[0] 
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Spot1'])) 
    plot_df = plot_df/adata.shape[0]  
    fig, ax = plt.subplots(figsize=[4,4],dpi=300)
    plt.ylabel('Percentage')
    plt.xlabel('Edge Numbers per Spot')
    plt.title('Number of Neighbors for Spots (Average=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df,color="#aa40fc",edgecolor="#f7b6d2",linewidth=2)

# def Cal_Spatial_variable_genes(adata):
#     import SpatialDE
#     counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
#     coor = pd.DataFrame(adata.obsm['spatial'], columns=['Spatial_X', 'Spatial_Y'], index=adata.obs_names)
#     Spatial_var_genes = SpatialDE.run(coor, counts)
#     Spatial_3000_var_genes = Spatial_var_genes["g"].values[0:3000]
#     Spatial_3000_var_genes = pd.DataFrame(Spatial_3000_var_genes)
#     all_genes = counts.columns.to_frame()
#     for i in range(len(all_genes.values)):
#         if all_genes.values[i] in Spatial_3000_var_genes.values:
#             all_genes.values[i] =1
#         else:
#             all_genes.values[i] =0
#     Spatial_highly_genes = all_genes.squeeze()
#     adata.var["Spatial_highly_variable_genes"] = Spatial_highly_genes.astype(bool)



def get_adata(dataset,data_path='../../Dataset/'):
    if  dataset.startswith('15'): #DLPFC dataset
        print("load DLPFC dataset:")
        file_fold = f"{data_path}DLPFC/{dataset}/"
        # 读入count
        raw = sc.read_visium(path=f'../../Dataset/DLPFC/{dataset}/', count_file='filtered_feature_bc_matrix.h5')
        # raw = sc.read_visium(path=file_fold, count_file=dataset + '_filtered_feature_bc_matrix.h5')



        raw.var_names_make_unique()
        # 读入真实标签
        Ann_df = pd.read_csv(f'../../Dataset/DLPFC/{dataset}/metadata.tsv', sep='\t')
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        raw.obs['ground_truth'] = Ann_df.loc[raw.obs_names, 'Ground Truth']

        n_clusters = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7


    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset)  # please replace 'file_fold' with the download path
        raw = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        raw.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
        raw.obs['ground_truth'] = df_meta.loc[raw.obs_names, 'fine_annot_type'].values  # fine_annot_type代替annot_type
        print("Breast_cancer的类别数", len(raw.obs['ground_truth'].unique()))

        n_clusters = 20

    if dataset == "Mouse_brain":
        raw = sq.datasets.visium_hne_adata()
        raw.var_names_make_unique()
        raw.obs['ground_truth'] = raw.obs["cluster"]
        print(f"Mouse_brain数据大小：{raw.shape}，类别数：{len(raw.obs['ground_truth'].unique())}")  # ,raw.obs['ground_truth'].unique())

        n_clusters = 15

    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")  # (428, 19736)
        raw.obs['ground_truth'] = raw.obs['Ground Truth']
        print("PDAC的类别数", raw.obs['ground_truth'].unique())
        n_clusters = 4

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + '/Adult_stereo.h5ad')
        print(raw.shape)  # (8243, 22144)
        raw.obs["ground_truth"] = raw.obs['Annotation']  # 后面需要obs['Annotation'] 格式
        print("Stereo标签类别数：", len(raw.obs['Annotation'].unique()))
        n_clusters = 16


    if dataset == 'SeqFish':
        raw = sq.datasets.seqfish()
        # print('Seqfish.shape',adata.shape)  # (19416, 351)
        raw.obs['ground_truth'] = raw.obs['celltype_mapped_refined']
        print("SeqFish标签类别数：", len(raw.obs['ground_truth'].unique()))
        n_clusters = 22

        image_coor = raw.obsm["spatial"]  # 直接获取像素点位置
        x_array = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        raw.obs["y_array"] = image_coor[:, 1]

    if dataset == "STARmap":
        file_fold = data_path + str(dataset)

        raw = sc.read(file_fold + "/STARmap_1207_1020.h5ad")
        print( f"STARmap数据大小：{raw.shape}，类别数：{len(raw.obs['ground_truth'].unique())}")  # ,raw.obs['ground_truth'].unique())
        n_clusters = 16

    return  raw,n_clusters

def DGI_loss_Draw(adata):
    import matplotlib.pyplot as plt
    if "SCGDL_loss" not in adata.uns.keys():
        raise ValueError("Please Train DGI Model using SCGDL_Train function first!") 
    Train_loss = adata.uns["SCGDL_loss"]
    plt.style.use('default') 
    plt.plot(Train_loss,label='Training loss',linewidth=2)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss of DGI model")
    plt.legend()
    plt.grid()

def BGMM(adata,n_cluster,used_obsm='SCGDL'):
    """
    BayesianGaussianMixture for spatial clustering.
    """

    knowledge = BayesianGaussianMixture(n_components=n_cluster,
                                        weight_concentration_prior_type ='dirichlet_process', ##'dirichlet_process' or dirichlet_distribution'
                                        weight_concentration_prior = 50).fit(adata.obsm[used_obsm])                                  
    # load ground truth for ARI and NMI computation.
    Ann_df = pd.read_csv("/home/tengliu/Torch_pyG/SCGDL_Upload_Files/data/Human_DLPFC/151675_truth.txt", sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    method = "BayesianGaussianMixture"
    labels = knowledge.predict(adata.obsm[used_obsm])+1
    adata.obs[method] = adata.obs['Ground Truth']
    adata.obs[method] = labels 
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df[method], obs_df['Ground Truth'])
    print("ARI:",ARI)
    adata.uns["ARI"] = ARI 
    return adata


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
