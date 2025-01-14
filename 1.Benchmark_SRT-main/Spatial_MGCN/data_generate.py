from __future__ import division
from __future__ import print_function

from utils import features_construct_graph, spatial_construct_graph1
import os
import argparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from config import Config
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    # adata.X=adata.X.toarray()
    return adata


def load_ST_file(dataset, highly_genes, k, radius, data_path, sq=None):

    if dataset.startswith('15'):  # DLPFC dataset
        print("load DLPFC dataset:")
        file_fold = f"{data_path}"
        raw = sc.read_visium(path=f'../../Dataset/DLPFC/{dataset}/', count_file='filtered_feature_bc_matrix.h5')
        Ann_df = pd.read_csv(f'../../Dataset/DLPFC/{dataset}/metadata.tsv', sep='\t')
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        raw.obs['ground_truth'] = Ann_df.loc[raw.obs_names, 'Ground Truth']
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

        n_clusters = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7

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
        raw.obs["x_pixel"] = spatial.loc[raw.obs_names, "X3"]
        raw.obs["y_pixel"] = spatial.loc[raw.obs_names, "X4"]
        x_array = raw.obs["x_array"].tolist()
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
        x_array = raw.obs['x_array']
        y_array = raw.obs['y_array']

        n_clusters = 4

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + '/Adult_stereo.h5ad')
        print(raw.shape)  # (8243, 22144)
        raw.obs["ground_truth"] = raw.obs['Annotation']
        n_clusters = 16
        #######   use for HE extraction  #######
        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]  #
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

        # adata.uns["spatial"] = dict()  #
        # library_id = 'Stereo'
        # adata.uns["spatial"][library_id] = dict()
        #
        # hires_image_file = '../../Dataset/Stereo/Stereo.png'
        # from matplotlib.image import imread
        # b = imread(hires_image_file)  # imread（）
        # adata.uns["spatial"][library_id]['images'] = dict()
        # adata.uns["spatial"][library_id]['images']['hires'] = b
        # adata.uns["spatial"][library_id]["use_quality"] = "hires"

    if dataset == 'SeqFish':
        raw = sq.datasets.seqfish()
        # print('Seqfish.shape',adata.shape)  # (19416, 351)
        raw.obs['ground_truth'] = raw.obs['celltype_mapped_refined']
        n_clusters = 22

        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    if dataset == "STARmap":
        file_fold = data_path + str(dataset)

        # adata = sc.read(file_fold+"/STARmap_20180505_BY3_1k.h5ad") #(1207, 1020)
        # adata.var_names_make_unique()
        # df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt', sep='\t', index_col=0)
        # adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,'Annotation'].values
        raw = sc.read(file_fold + "/STARmap_1207_1020.h5ad")

        n_clusters = 16
        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    adata=raw
    adata.var_names_make_unique()
    # adata.obs['ground'] = labels
    # adata.obs['ground_truth'] = labels
    adata.var_names_make_unique()
    adata = normalize(adata, highly_genes=highly_genes)
    fadj = features_construct_graph(adata.X, k=k) #K=15
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)

    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset2 = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    Dataset_test = ['151673']

    for dataset in dataset2:
        if dataset.startswith('15'):
            savepath = "../../Dataset/Spatial_MGCN_generate/DLPFC/" + dataset + "/"
            input=f'../../Dataset/DLPFC/{dataset}/'
            config_file = './config/DLPFC.ini'

        else:
            savepath = "../../Dataset/Spatial_MGCN_generate/" + dataset + "/"
            input = f'../../Dataset/{dataset}/'
            config_file = './config/' + dataset + '.ini'

        mk_dir(savepath)
        config = Config(config_file)
        adata = load_ST_file(dataset, config.fdim, config.k, config.radius,input)
        print("saving", adata.shape)

        adata.write(f'{savepath}Spatial_MGCN_{dataset}.h5ad')
        print("done!")






