# https://github.com/jianhuupenn/SpaGCN

import os
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import psutil,tracemalloc
from sklearn import metrics
import multiprocessing as mp
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
import glob2
import matplotlib.pyplot as plt
import squidpy as sq
import time
from matplotlib.backends.backend_pdf import PdfPages
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
        file_fold = f"{data_path}DLPFC/{dataset}/"
        raw = sc.read_visium(path=f'../../Dataset/DLPFC/{dataset}/', count_file='filtered_feature_bc_matrix.h5')
        # raw = sc.read_visium(path=file_fold, count_file=dataset + '_filtered_feature_bc_matrix.h5')
        raw.var_names_make_unique()
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
        n_clusters = 20

    if dataset == "Mouse_brain":
        raw = sq.datasets.visium_hne_adata()
        raw.var_names_make_unique()
        raw.obs['ground_truth'] = raw.obs["cluster"]
        print( f"Mouse_brain shape：{raw.shape}，clustering type：{len(raw.obs['ground_truth'].unique())}")

        x_array = raw.obs['array_row']
        y_array = raw.obs['array_col']
        raw.obs["x_array"] = x_array
        raw.obs["y_array"] = y_array
        n_clusters = 15

    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")  # (428, 19736)
        raw.obs['ground_truth'] = raw.obs['Ground Truth']
        print("PDAC clustering types", raw.obs['ground_truth'].unique())
        x_array = raw.obs['x_array']
        y_array = raw.obs['y_array']

        n_clusters = 4

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + '/Adult_stereo.h5ad')
        print(raw.shape)  # (8243, 22144)
        raw.obs["ground_truth"] = raw.obs['Annotation']
        print("Stereo clustering type：", len(raw.obs['Annotation'].unique()))
        n_clusters = 16
        ####### Complete Stereo-seq data for easy utilization of HE information  #######
        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

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
        # raw dataset
        # adata = sc.read(file_fold+"/STARmap_20180505_BY3_1k.h5ad") #(1207, 1020)
        # adata.var_names_make_unique()
        # df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt', sep='\t', index_col=0)
        # adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,'Annotation'].values
        raw = sc.read(file_fold + "/STARmap_1207_1020.h5ad")
        print(f"STARmap shape：{raw.shape}，Clustering types：{len(raw.obs['ground_truth'].unique())}")

        n_clusters = 16
        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    return  raw,n_clusters


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

    img = None
    if dataset == "Mouse_brain":
        img = cv2.imread(f"../../Dataset/Mouse_brain/spatial/tissue_hires_image.png")
    elif dataset == "Breast_cancer":
        img = cv2.imread(f"../../Dataset/Breast_cancer/spatial/tissue_hires_image.png")
    elif dataset == "Stereo":
        img = cv2.imread(f"../../Dataset/Stereo/Adult.tif")
    elif dataset.startswith('15'):
        img=cv2.imread(f'../../Dataset/DLPFC/{dataset}/spatial/tissue_hires_image.png')


    if img is not None:
        img_new = img.copy()
    else:
        pass

    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()
    x_pixel=x_array
    y_pixel=y_array
    # Test coordinates on the image
    for i in range(len(x_pixel)):
        x = x_pixel[i]
        y = y_pixel[i]
        img_new[int(x - 20):int(x + 20), int(y - 20):int(y + 20), :] = 0

    b = 49
    a = 1
    adj = spg.calculate_adj_matrix(x=x_array, y=y_array, x_pixel=x_pixel, y_pixel=y_pixel,
                                   image=img, beta=b, alpha=a, histology=True)  ##  histology=True
    p=0.5
    l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    n_clusters = len(set(adata.obs["ground_truth"].dropna()))
    r_seed=t_seed=n_seed=random_seed
    res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1,
                       tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed,
                       t_seed=t_seed, n_seed=n_seed)

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
    refine_map = {"Breast_cancer": "hexagon", "Mouse_brain": "hexagon", "DLPFC": "hexagon",
                  "STARmap": "square", "MOB_without_label": "hexagon", "PDAC": "square","SeqFish": "square","Stereo": "hexagon"}

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


    ari_r, nmi_r, ami_r = eval_model(adata.obs['refined_pred'], adata.obs['ground_truth'])
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



dataset1 = ["Stereo", "Breast_cancer", "Mouse_brain"] #"STARmap","PDAC","SeqFish" has no HE.
dataset2 = ['151507', '151508', '151509', '151510','151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
Dataset_test = ["Mouse_brain", "Breast_cancer", "PDAC", "SeqFish", "Stereo", "STARmap", '151507', '151508']

for dataset in dataset2:
    if dataset.startswith('15'):
        save_path=f'../../Output/SpaGCN/DLPFC/{dataset}/Has_HE/'
    else:
        save_path = f'../../Output/SpaGCN/{dataset}/Has_HE/'
    mk_dir(save_path)


    adata,n_cluster = get_adata(dataset)
    print("adate.shape：", adata.shape)
    random_seed = np.random.randint(100)

    results = pd.DataFrame()
    for i in range(10):
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