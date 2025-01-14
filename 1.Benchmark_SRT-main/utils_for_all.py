import squidpy as sq
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score

import psutil,time,tracemalloc



platform_map = {"Mouse_hippocampus": "slideseq", "Mouse_olfactory_slide_seqv2": "slideseq", "MOB_without_label": "stereoseq",
                "PDAC": "ST", "DLPFC": '10 X', "Breast_cancer": '10 X', "Mouse_brain": '10 X',
                "SeqFish": "Seqfish", "STARmap": "STARmap"
                }

n_clusters_map = {"Stereo": 16, "STARmap": 16, "SeqFish": 22,
                  "DLPFC": '5-7', "Breast_cancer": 20, "Mouse_brain": 15,
                  "PDAC": 4}
# cluster_num = n_clusters_map[dataset]

def spatial_obs_loction(adata,library_id="V1_Mouse_Brain_Sagittal_Anterior",quality='hires'):
    image_coor = adata.obsm["spatial"]

    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality  # quality='hires'

    adata.obs["array_row"] = adata.obs["array_row"].astype(int)
    adata.obs["array_col"] = adata.obs["array_col"].astype(int)
    adata.obsm["spatial"] = adata.obsm["spatial"].astype("int64")
    return adata



def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


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



def get_adata(dataset,data_path='../../Dataset/'):
    if  dataset.startswith('15'): #DLPFC dataset
        print("load DLPFC dataset:")
        file_fold = f"{data_path}DLPFC/{dataset}/"  # data_path + str(dataset)+str(section_id)
        # input_dir = os.path.join(file_fold, section_id)
        # # with open('../../adata/Raw_adata/DL')
        raw = sc.read_visium(path=file_fold, count_file='filtered_feature_bc_matrix.h5')
        spatial = pd.read_csv(file_fold + "spatial/tissue_positions_list.csv",
                              sep=",",
                              header=None,
                              na_filter=False,
                              index_col=0,
                              )
        spatial.columns = ["X0", "X1", "X2", "X3", "X4"]
        raw.obs["x_array"] = spatial.loc[raw.obs_names, "X1"]
        raw.obs["y_array"] = spatial.loc[raw.obs_names, "X2"]
        raw.obs["x_pixel"] = spatial.loc[raw.obs_names, "X3"]  # 坐标文件中X4,X5表示X,Y像素点位置
        raw.obs["y_pixel"] = spatial.loc[raw.obs_names, "X4"]
        x_array = raw.obs["x_array"].tolist()  # 存储所有X横坐标位置
        y_array = raw.obs["y_array"].tolist()

        Ann_df = pd.read_csv(f'../../Dataset/DLPFC/{dataset}/metadata.tsv', sep='\t')
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        raw.obs['ground_truth'] = Ann_df.loc[raw.obs_names, 'Ground Truth']


        n_clusters = 7

    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset)  # please replace 'file_fold' with the download path
        raw = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        raw.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
        raw.obs['ground_truth'] = df_meta.loc[raw.obs_names, 'fine_annot_type'].values  # fine_annot_type代替annot_type
        print("Breast_cancer的类别数", len(raw.obs['ground_truth'].unique()))

        spatial = pd.read_csv(file_fold + '/spatial/tissue_positions_list.csv',
                              sep=",",
                              header=None,
                              na_filter=False,
                              index_col=0,
                              )
        spatial.columns = ["X0", "X1", "X2", "X3", "X4"]
        raw.obs["x_array"] = spatial.loc[raw.obs_names, "X1"]
        raw.obs["y_array"] = spatial.loc[raw.obs_names, "X2"]
        raw.obs["x_pixel"] = spatial.loc[raw.obs_names, "X3"]  # 坐标文件中X4,X5表示X,Y像素点位置
        raw.obs["y_pixel"] = spatial.loc[raw.obs_names, "X4"]
        x_array = raw.obs["x_array"].tolist()  # 存储所有X横坐标位置
        y_array = raw.obs["y_array"].tolist()
        # x_pixel = raw.obs["x_pixel"].tolist()
        # y_pixel = raw.obs["y_pixel"].tolist()
        n_clusters = 20

    if dataset == "Mouse_brain":
        raw = sq.datasets.visium_hne_adata()
        raw.var_names_make_unique()
        raw.obs['ground_truth'] = raw.obs["cluster"]
        print(
            f"Mouse_brain数据大小：{raw.shape}，类别数：{len(raw.obs['ground_truth'].unique())}")  # ,raw.obs['ground_truth'].unique())

        x_array = raw.obs['array_row']
        y_array = raw.obs['array_col']
        raw.obs["x_array"] = x_array
        raw.obs["y_array"] = y_array
        n_clusters = 15

    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")  # (428, 19736)
        raw.obs['ground_truth'] = raw.obs['Ground Truth']
        print("PDAC的类别数", raw.obs['ground_truth'].unique())
        x_array = raw.obs['x_array']
        y_array = raw.obs['y_array']

        n_clusters = 4

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + '/Adult_stereo.h5ad')
        print(raw.shape)  # (8243, 22144)
        raw.obs["ground_truth"] = raw.obs['Annotation']
        print("Stereo标签类别数：", len(raw.obs['Annotation'].unique()))
        n_clusters = 16
        #######   尝试对Stereo-seq数据进行HE特征提取  #######
        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

        # adata.uns["spatial"] = dict()
        # library_id = 'Stereo'
        # adata.uns["spatial"][library_id] = dict()
        #
        # hires_image_file = '../../Dataset/Stereo/Stereo.png'
        # from matplotlib.image import imread
        # b = imread(hires_image_file)
        # adata.uns["spatial"][library_id]['images'] = dict()
        # adata.uns["spatial"][library_id]['images']['hires'] = b
        # adata.uns["spatial"][library_id]["use_quality"] = "hires"

    if dataset == 'SeqFish':
        raw = sq.datasets.seqfish()
        # print('Seqfish.shape',adata.shape)  # (19416, 351)
        raw.obs['ground_truth'] = raw.obs['celltype_mapped_refined']
        print("SeqFish标签类别数：", len(raw.obs['ground_truth'].unique()))
        n_clusters = 22

        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        # read the raw dataset
        # adata = sc.read(file_fold+"/STARmap_20180505_BY3_1k.h5ad") #(1207, 1020)
        # adata.var_names_make_unique()
        # df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt', sep='\t', index_col=0)
        # adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,'Annotation'].values
        raw = sc.read(file_fold + "/STARmap_1207_1020.h5ad")

        n_clusters = 16
        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]  #
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    return  raw,n_clusters



def load_graph_V1(dataset, adata,l):

    if dataset in ["Breast_cancer","Mouse_brain"]:
        x_array = adata.obs["array_row"].tolist()
        y_array=adata.obs["array_col"].tolist()
    if dataset=="PDAC":
        x_array = adata.obs["x_array"].tolist()
        y_array=adata.obs["y_array"].tolist()
    if dataset == "SeqFish":
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index  # spot命名
        coor.columns = ['imagerow', 'imagecol']
        adata.obs["x_array"] = coor['imagerow']
        adata.obs["y_array"] = coor['imagecol']
        x_array = adata.obs["x_array"].tolist()
        y_array = adata.obs["y_array"].tolist()
    if dataset=="STARmap":
        adata.obs["x_array"]=adata.obs["X"]
        adata.obs["y_array"]=adata.obs["Y"]
        x_array = adata.obs["x_array"].tolist()
        y_array = adata.obs["y_array"].tolist()


def calculate_clustering_matrix(pred, gt, sample):
    df = pd.DataFrame(columns=['Sample', 'ARI', 'NMI', 'AMI'])
    ari = adjusted_rand_score(pred, gt)
    nmi = normalized_mutual_info_score(pred, gt)
    ami = adjusted_mutual_info_score(pred, gt)
    df = df.append(pd.Series([sample, ari, nmi,ami],index=['Sample', 'ARI', 'NMI', 'AMI']), ignore_index=True)
    return df


