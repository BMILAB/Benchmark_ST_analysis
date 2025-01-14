import os
import torch
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
import numpy as np
import psutil,time,tracemalloc
##Note: To call the mclust clustering algorithm in the R package, make sure that mclust is installed in R and that rpy2 can be connected successfully
os.environ["R_HOME"] = r"D:\R-4.3.1"
os.environ["PATH"]   = r"D:\R-4.3.1\bin\x64" + ";" + os.environ["PATH"]


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



def run_GraphST(adata, dataset, random_seed = np.random.randint(100), device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') ):

    from GraphST_v1 import GraphST
    from utils import clustering

    model = GraphST.GraphST(adata, device=device,epochs=600, random_seed=random_seed)
    adata = model.train()
    n_clusters = len(set(adata.obs["ground_truth"].dropna()))
    # set radius to specify the number of neighbors considered during refinement
    radius = 50 # ### default radius = 50
    adata.obsm["embedding"] = adata.obsm["emb"]
    clustering(adata, n_clusters, radius=radius, method="mclust",refinement=False)
    ari, nmi, ami  = eval_model(adata.obs['domain'], adata.obs['ground_truth'])
    SC = silhouette_score(adata.obsm["embedding"], adata.obs['domain'])
    used_adata = adata[adata.obs["ground_truth"].notna()]

    clustering(adata, n_clusters, radius=radius, method="mclust", refinement=True)
    ari_r, nmi_r, ami_r = eval_model(adata.obs['domain'], adata.obs['ground_truth'])
    SC_r = silhouette_score(adata.obsm["embedding"], adata.obs['domain'])

    used_adata = adata[adata.obs["ground_truth"].notna()]

    res = {}
    res["dataset"] = dataset
    res["ari"] = ari
    res["nmi"] = nmi
    res["ami"] = ami
    res["sc"] = SC

    res["nmi_r"] = nmi_r
    res["ari_r"] = ari_r
    res["ami_r"] = ami_r
    res["sc_r"] = SC_r
    adata.obs["pred_label"] = adata.obs['domain']
    return res, adata

import sys
sys.path.append('../')
import utils_for_all as usa  ##Call a unified data reading function to avoid repetitive data input in each method
if __name__ == '__main__':

  Dataset=["ST_Hippocampus_2",'SlideV2_mouse_embryo_E8.5','151673',"SeqFish","STARmap","Stereo","Mouse_brain","Breast_cancer","PDAC"]
  Dataset_test=['151673']
for dataset in Dataset_test:
    if dataset.startswith('15'):
        save_data_path = f'../../Output/GraphST/DLPFC/{dataset}/'
    else:
        save_data_path = f'../../Output/GraphST/{dataset}/'
    if not os.path.exists( save_data_path):
        os.makedirs(save_data_path)

    adata, _ = usa.get_adata(dataset, data_path='../../Dataset/')
    adata.var_names_make_unique()

    best_ari = 0
    results=pd.DataFrame()
    for i in range(1):
        random_seed = 0
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        res, adata_h5= run_GraphST(adata.copy(), dataset, random_seed=random_seed)

        end = time.time()
        end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        uesd_time = end - start
        used_memo = end_MB - start_MB
        current, peak = tracemalloc.get_traced_memory()  #
        tracemalloc.stop()
        peak = peak / 1024.0 / 1024.0 / 1024.0
        print(u'Current memory usage_end:：%.4f GB' % used_memo)
        print('time: {:.4f} s'.format(uesd_time))
        print('memory blocks peak:{:>10.4f} GB'.format(peak))
        tracemalloc.clear_traces()

        res["time"] = uesd_time
        res["Memo"] = used_memo
        res["Memo_peak"] = peak
        res["round"] = i+1

        results = results._append(res, ignore_index=True)

    print(results.head())
    results.to_csv(save_data_path + "/{}_result.csv".format(dataset), header=True)
    adata_h5.write_h5ad(save_data_path + str(dataset) + ".h5ad")
    results.to_csv(f'{save_data_path}{dataset}_result.csv', header=True)
    results.set_index('dataset', inplace=True)
    print(results.head())
    res_mean = results.mean()
    res_mean.to_csv(f'{save_data_path}{dataset}_mean.csv', header=True)
    res_std = results.std()
    res_std.to_csv(f'{save_data_path}{dataset}_std.csv', header=True)
    res_median = results.median()
    res_median.to_csv(f'{save_data_path}{dataset}_median.csv', header=True)