import os
import torch
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
import squidpy as sq
import time,psutil,tracemalloc
#Be sure that R_HOME is included in the environment variant. Otherwise it needs to be defined here
os.environ["R_HOME"] = r"D:\R-4.3.1"
os.environ["PATH"]   = r"D:\R-4.3.1\bin\x64" + ";" + os.environ["PATH"]

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami=adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
    return  ari,nmi,ami


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
        file_fold = data_path + str(dataset)  # please replace 'file_fold' with the download path
        raw = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        raw.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
        raw.obs['ground_truth'] = df_meta.loc[raw.obs_names, 'fine_annot_type'].values
        n_clusters = 20

    if dataset == "Mouse_brain":
        raw = sq.datasets.visium_hne_adata()
        raw.var_names_make_unique()
        raw.obs['ground_truth'] = raw.obs["cluster"]

        n_clusters = 15

    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")
        raw.obs['ground_truth'] = raw.obs['Ground Truth']
        n_clusters = 4

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + '/Adult_stereo.h5ad')
        print(raw.shape)  # (8243, 22144)
        raw.obs["ground_truth"] = raw.obs['Annotation']
        n_clusters = 16

    if dataset == 'SeqFish':
        raw = sq.datasets.seqfish()
        raw.obs['ground_truth'] = raw.obs['celltype_mapped_refined']
        n_clusters = 22

        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/STARmap_1207_1020.h5ad")
        n_clusters = 16

    return raw, n_clusters

def run_STAGATE(adata, dataset, random_seed=np.random.randint(100),
                device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
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
    adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=n_clusters)

    obs_df = adata.obs.dropna()
    adata.obs["pred_label"] = adata.obs["mclust"]
    adata.obsm["embedding"] = adata.obsm["STAGATE"]

    res = {}
    if ("ground_truth" in adata.obs.keys()):
        ari, nmi, ami = eval_model(adata.obs['mclust'], adata.obs['ground_truth'])
        SC = silhouette_score(adata.obsm["embedding"],adata.obs['mclust'])

        used_adata = adata[adata.obs["ground_truth"].notna()]
        SC_revise = silhouette_score(used_adata.obsm["embedding"], used_adata.obs['ground_truth'])

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

import sys
sys.path.append('../')
import utils_for_all as usa
if __name__ == '__main__':

    # dataset1 = ["Stereo", "Breast_cancer", "Mouse_brain", "STARmap", "SeqFish", "STARmap"]
    Dataset_test = ['151673']
for dataset in Dataset_test:
    print(f"====================begin test on {dataset}======================================")
    if dataset.startswith('15'):
        save_path = f'../../Output/STAGATE/DLPFC/{dataset}/'
    else:
        save_path = f'../../Output/STAGATE/{dataset}/'
    mk_dir(save_path)

    adata, n_clusters = usa.get_adata(dataset, data_path='../../Dataset/')
    adata.var_names_make_unique()

    random_seed = 0
    rad_cutoff = 150
    results = pd.DataFrame()
    for i in range(1):
        num = i + 1
        print("===epoch:{}===".format(num))
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        res, adata_h5 = run_STAGATE(adata.copy(), dataset, random_seed=random_seed, rad_cutoff=rad_cutoff,n_clusters= n_clusters)

        end = time.time()
        end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        uesd_time = end - start
        used_memo = end_MB - start_MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = peak / 1024.0 / 1024.0 / 1024.0
        print(u'Current memory usage_end:：%.4f GB' % used_memo)
        print('time: {:.4f} s'.format(uesd_time))
        print('memory blocks peak:{:>10.4f} GB'.format(peak))
        tracemalloc.clear_traces()

        res["time"] = uesd_time
        res["Memo"] = used_memo
        res["Memo_peak"] = peak
        res["round"] = i + 1
        results = results._append(res, ignore_index=True)

    adata_h5.write_h5ad(save_path + str(dataset) + ".h5ad")
    results.set_index('dataset', inplace=True)
    results.to_csv(save_path +"/result_"+dataset+".csv", header=True)
    print(results.head())
    res_mean = results.mean()
    res_mean.to_csv(f'{save_path}{dataset}_mean.csv', header=True)
    res_std = results.std()
    res_std.to_csv(f'{save_path}{dataset}_std.csv', header=True)
    res_median = results.median()
    res_median.to_csv(f'{save_path}{dataset}_median.csv', header=True)

