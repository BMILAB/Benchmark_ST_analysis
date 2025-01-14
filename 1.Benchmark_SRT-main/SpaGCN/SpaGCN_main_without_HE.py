
import os
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import psutil,tracemalloc
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
import time

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


def run_SpaGCN(adata, dataset, random_seed = np.random.randint(100),
                device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
                save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/SpaGCN/"):
    import numpy as np
    # import SpaGCN as spg
    import SpaGCN_raw as spg
    import random, torch
    import cv2


    ##### Spatial domain detection using SpaGCN
    spg.prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)


    ######  As there is no image, the distance is calculated directly from the coordinates#########################
    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()
    x_pixel=x_array
    y_pixel=y_array
    adj = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)

    p=0.5
    l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    n_clusters = len(set(adata.obs["ground_truth"].dropna()))
    r_seed=t_seed=n_seed=random_seed
    res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1,
                       tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed,
                       t_seed=t_seed, n_seed=n_seed)

    ### 4.3 Run SpaGCN
    clf=spg.SpaGCN()
    clf.set_l(l)
    # Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    clf.train(adata, adj, init_spa=True, init="louvain",
              res=res, tol=5e-3, lr=0.05, max_epochs=200)
    emb,y_pred, prob=clf.predict()

    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')
    adata.obsm["embedding"]=emb

    if dataset.startswith('15'):
        dataset='DLPFC'
    else:
        dataset=dataset
    refine_map = {"Breast_cancer": "hexagon", "Mouse_brain": "hexagon", "DLPFC": "hexagon", "PDAC": "square"}

    #Do cluster refinement(optional)
    # shape="hexagon" for Visium data, "square" for ST data.
    adj_2d=spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(),
                            pred=adata.obs["pred"].tolist(), dis=adj_2d,
                            shape=refine_map[dataset])
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')


    ari, nmi, ami = eval_model(adata.obs['pred'], adata.obs['ground_truth'])
    SC = silhouette_score(adata.obsm["embedding"], adata.obs['pred'])

    used_adata = adata[adata.obs["ground_truth"].notna()]
    SC_revise = silhouette_score(used_adata.obsm["embedding"], used_adata.obs['ground_truth'])


    ari_r, nmi_r, ami_r = eval_model(adata.obs['refined_pred'], adata.obs['ground_truth'])  # 因为结果都保存在domain中
    SC_r = silhouette_score(adata.obsm["embedding"], adata.obs['refined_pred'])
    SC_r_revise = silhouette_score(used_adata.obsm["embedding"], used_adata.obs['refined_pred'])
    res = {}
    res["dataset"] = dataset
    res["ari"] = ari
    res["nmi"] = nmi
    res["ami"] = ami
    res["sc"] = SC
    res['SC_revise'] = SC_revise

    res["nmi_r"] = nmi_r
    res["ari_r"] = ari_r
    res["ami_r"] = ami_r
    res["sc_r"] = SC_r
    res['SC_r_revise']=SC_r_revise
    return res, adata


import sys
sys.path.append('../')
import utils_for_all as usa
if __name__ == '__main__':

 Dataset= ["Mouse_brain", "Breast_cancer", "PDAC", "SeqFish", "Stereo", "STARmap", '151507', '151508']
Dataset_test = ['151673']

for dataset in Dataset_test:
    if dataset.startswith('15'):
        save_path=f'../../Output/SpaGCN/DLPFC/{dataset}/Has_HE/'
    else:
        save_path = f'../../Output/SpaGCN/{dataset}/Has_HE/'
    mk_dir(save_path)

    adata, n_cluster = usa.get_adata(dataset, data_path='../../Dataset/')
    adata.var_names_make_unique()
    random_seed = np.random.randint(100)

    results = pd.DataFrame()
    for i in range(1):
        num = i + 1
        print("===Training epoch:{}====".format(num))
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        res, adata_h5 = run_SpaGCN(adata.copy(), dataset, random_seed=random_seed)

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
    results.to_csv(save_path +"/res_result.csv", header=True)
    results.set_index('dataset', inplace=True)
    print(results.head())
    res_mean = results.mean()
    res_mean.to_csv(f'{save_path}{dataset}_mean.csv', header=True)
    res_std = results.std()
    res_std.to_csv(f'{save_path}{dataset}_std.csv', header=True)
    res_median = results.median()
    res_median.to_csv(f'{save_path}{dataset}_median.csv', header=True)