import os
import torch
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn import metrics
import multiprocessing as mp
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score

import glob2
import matplotlib.pyplot as plt
import squidpy as sq
import time
import psutil
from matplotlib.backends.backend_pdf import PdfPages


def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

##它不用dropNA，输入原始的预测结果即可
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

#Dataset/Breast_cancer

def read_data(dataset, data_path='/media/test/Elements/备份/python工程文件/2023_Benchmark_ST_GNN/Dataset'):
    if dataset == "Mouse_brain":
        adata = sq.datasets.visium_hne_adata()
        adata.var_names_make_unique()
        adata.obs['ground_truth'] = adata.obs["cluster"]
        print("Mouse_brain的类别数", adata.obs['ground_truth'].value_counts())
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset)  # please replace 'file_fold' with the download path
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5',
                               load_images=True)
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'fine_annot_type'].values
        print("Breast_cancer的类别数", adata.obs['ground_truth'].value_counts())
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()
        # # 还有图片的数据进行处理，保存HE的CSV特征
        # from DeepST_V1.his_feat import image_feature, image_crop
        # library_id = list(adata.uns["spatial"].keys())[0]
        # scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
        # image_coor = adata.obsm["spatial"] * scale
        # adata.obs["imagecol"] = image_coor[:, 0]
        # adata.obs["imagerow"] = image_coor[:, 1]
        # adata.uns["spatial"][library_id]["use_quality"] = "hires"
        # from pathlib import Path
        # save_path_image_crop = Path(os.path.join("../../Output/DeepST_V1/temp/", 'Image_crop', f'{dataset}'))
        # save_path_image_crop.mkdir(parents=True, exist_ok=True)
        # adata = image_crop(adata, save_path=save_path_image_crop)
        # adata = image_feature(adata, pca_components=50, cnnType='ResNet50').extract_image_feat()
        # print("HE未经PCA前的维度：", adata.obsm['image_feat'].shape)

    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        # 读入原始数据
        # adata = sc.read(file_fold+"/STARmap_20180505_BY3_1k.h5ad") #(1207, 1020)
        # adata.var_names_make_unique()
        # df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt', sep='\t', index_col=0)
        # adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,'Annotation'].values
        adata = sc.read(file_fold + "/STARmap_1207_1020.h5ad")
        adata.obs['ground_truth'].value_counts().plot(kind='bar')
        plt.tight_layout()  # 调整画布在正中间
        plt.show()

    if dataset == "PDAC":
        file_fold = data_path +str(dataset)
        adata = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")  # (428, 19736)
        adata.obs['ground_truth'] = adata.obs['Ground Truth']
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset == 'Stereo':
        file_fold = data_path + "/Stereo"
        adata = sc.read(file_fold +'/Adult_stereo.h5ad')
        print(adata.shape)  # (8243, 22144)
        adata.obs["ground_truth"]=adata.obs['Annotation'] #后面需要obs['Annotation'] 格式
        print("标签类别数：", len(adata.obs['Annotation'].unique()))



    if dataset == 'SeqFish_Mouse_Embryos':
        adata = sq.datasets.seqfish()
        # print('Seqfish.shape',adata.shape)  # (19416, 351)
        adata.obs['ground_truth'] = adata.obs['celltype_mapped_refined']
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset in ["Mouse_olfactory", "MOB"]:
        # please replace 'file_fold' with the download path
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold + '/filtered_feature_bc_matrix.h5ad')  # (3739,36601)
        adata.var_names_make_unique()

    if dataset == "Mouse_hippocampus":
        adata = sq.datasets.slideseqv2()
        adata.var_names_make_unique()
    return adata

def run_STAGATE(adata, dataset, random_seed=np.random.randint(100),
                device=torch.device(
                    'cuda:0' if torch.cuda.is_available() else 'cpu'),
                save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/STAGATE/",
                n_clusters=None, rad_cutoff=150):
    import STAGATE_pyG as STAGATE
    start = time.time()
    start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    STAGATE.Stats_Spatial_Net(adata)
    adata = STAGATE.train_STAGATE(adata, device=device, random_seed=random_seed)
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)

    if ("ground_truth" in adata.obs.keys()):
        n_clusters = len(set(adata.obs["ground_truth"].dropna()))
    else:
        n_clusters = n_clusters
    adata = STAGATE.mclust_R(
        adata, used_obsm='STAGATE', num_cluster=n_clusters)

    obs_df = adata.obs.dropna()
    adata.obs["pred_label"] = adata.obs["mclust"]
    adata.obsm["embedding"] = adata.obsm["STAGATE"]

    res = {}
    if ("ground_truth" in adata.obs.keys()):
        ari, nmi, ami = eval_model(adata.obs['mclust'], adata.obs['ground_truth'])  # 因为结果都保存在domain中
        SC = silhouette_score(adata.obsm["embedding"],adata.obs['mclust'])
        SC_revise = silhouette_score(adata.obsm["embedding"], adata.obs['ground_truth'])
        end = time.time()
        end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  #
        used_memory = end_MB - start_MB

        res = {}
        res["dataset"] = dataset
        res["ari"] = ari
        res["nmi"] = nmi
        res["ami"] = ami
        res["sc"] = SC
        res["time"] = end - start
        res["Memo"] = used_memory
        res['SC_revise']=SC_revise

    # adata.write_h5ad(save_data_path+str(dataset)+".h5ad")
    return res, adata

# save_path = "/home/fangzy/project/st_cluster/code/compare/STAGATE/res"
# save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/STAGATE/"
# results = pd.DataFrame()
# for dataset in ["Breast_cancer", "Mouse_brain", "STARmap"]:
#     best_ari = 0
#     adata = read_data(dataset)
#     for i in range(10):
#         random_seed = np.random.randint(100)
#         res, adata_h5 = run_STAGATE(adata.copy(), dataset, random_seed=random_seed)
#         res["round"] = i
#         results = results._append(res, ignore_index=True)
#         results.to_csv(save_path +
#                     "/res_other.csv", header=True)
#         if res["ari_1"] > best_ari:
#             adata_h5.write_h5ad(save_data_path+str(dataset)+".h5ad")
# res_dataset_mean = results.groupby(["dataset"]).mean()
# res_dataset_mean.to_csv(save_path+"/other_data_mean.csv", header=True)




save_path = "/media/test/Elements/备份/python工程文件/2023_Benchmark_ST_GNN/Output/STAGATE"
results = pd.DataFrame()
## "Breast_cancer", "Mouse_brain,"STARmap","SeqFish","PDAC","Stereo"
for dataset in ["SeqFish_Mouse_Embryos"]:
    print("现在运行的数据是：",dataset)
    save_data_path = f'{save_path}{dataset}/'
    print(save_data_path)
    mk_dir(save_data_path)
    adata = read_data(dataset)
    print("读取的数据大小：",adata.shape)

    for i in range(1):
        random_seed = 0
        rad_cutoff=50
        res, adata_h5 = run_STAGATE(adata.copy(), dataset, random_seed=random_seed, rad_cutoff=rad_cutoff)
        res["round"] = i+1
        results = results._append(res, ignore_index=True)
        results.to_csv(save_data_path +"/result_"+dataset+".csv", header=True)
        print(results.head())
        adata_h5.write_h5ad(save_data_path+str(dataset)+".h5ad")

res_dataset_mean = results.groupby(["dataset"]).mean()
res_dataset_mean.to_csv(save_path+"/other_data_mean.csv", header=True)

res_dataset_std = results.groupby(["dataset"]).std()
res_dataset_std .to_csv(save_path+"/other_data_std.csv", header=True)

res_mean = res_dataset_mean.mean()
res_mean.to_csv(save_path+"/other_mean.csv", header=True)
res_median = res_dataset_mean.median()
res_median.to_csv(save_path+"/other_median.csv", header=True)
