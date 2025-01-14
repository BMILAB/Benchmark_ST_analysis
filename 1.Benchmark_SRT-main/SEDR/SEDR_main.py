import os
import torch
import argparse
import warnings
import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_visium_sge
from src.SEDR_train import SEDR_Train
from util import get_adata,mk_dir,eval_model
from sklearn.metrics import silhouette_score
import time
import psutil
warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import psutil,time,tracemalloc
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res


def run_SEDR(adata_h5,save_path,epochs):
    # ################ Parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                        help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs to train.') #default=200
    parser.add_argument('--cell_feat_dim', type=int, default=200, help='Dim of PCA')
    parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
    parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
    parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
    parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
    parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
    parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
    parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
    parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
    parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
    parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
    parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
    parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
    parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
    parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
    # ______________ Eval clustering Setting _________
    parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
    parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')

    params = parser.parse_args()
    params.device = device

    params.cell_num = adata_h5.shape[0]
    params.save_path = mk_dir(save_path)

    start = time.time()
    start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

    adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)
    graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], params)
    params.cell_num = adata_h5.shape[0]
    params.save_path = mk_dir(save_path)


    print('==== Graph Construction Finished')
    # ################## Model training
    sedr_net = SEDR_Train(adata_X, graph_dict, params)
    if params.using_dec:
        sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()
    sedr_feat, _, _, _ = sedr_net.process()

    # ################## Result plot
    adata_sedr = anndata.AnnData(sedr_feat)
    if dataset in ["Mouse_brain","Breast_cancer"]:
        adata_sedr.uns['spatial'] = adata_h5.uns['spatial']
        adata_sedr.obsm['spatial'] = adata_h5.obsm['spatial']
    else:
         print("no spatial information")

    adata_sedr.obs['ground_truth'] = adata_h5.obs['ground_truth'].values
    sc.pp.neighbors(adata_sedr, n_neighbors=params.eval_graph_n)
    sc.tl.umap(adata_sedr)
    eval_resolution = res_search_fixed_clus(adata_sedr, n_cluster)
    sc.tl.leiden(adata_sedr, key_added="SEDR_leiden", resolution=eval_resolution)

    adata_sedr.write_h5ad(f'{save_path}{dataset}.h5ad')

    if dataset in ['Mouse_brain,Breast_cancer']:
        sc.pl.spatial(adata_sedr, img_key="hires", color=['SEDR_leiden'], show=False)
        plt.savefig(os.path.join(params.save_path, "SEDR_leiden_plot.pdf"), bbox_inches='tight', dpi=150)
        plt.show()


    # ---------- Load manually annotation ---------------
    ari, nmi, ami = eval_model(adata_sedr.obs['SEDR_leiden'], adata_sedr.obs['ground_truth'])
    SC = silhouette_score(adata_sedr.X, adata_sedr.obs['SEDR_leiden'])
    used_adata = adata_sedr[adata_sedr.obs["ground_truth"].notna()]
    res = {}
    res["dataset"] = dataset
    res["ari"] = ari
    res["nmi"] = nmi
    res["ami"] = ami
    res["sc"] = SC
    return res


import sys
sys.path.append('../')
import utils_for_all as usa
if __name__ == '__main__':
 # Dataset1 = ['151507', '151508', '151509', '151510', '151669', '151670','151671', '151672', '151673', '151674', '151675', '151676',"STARmap","ST_Hippocampus_2",'SlideV2_mouse_embryo_E8.5','151673',"SeqFish","Stereo"]
  Dataset_test=['151673']

for dataset in Dataset_test:
    data_root = os.path.join("../../Dataset/", dataset)
    save_path = os.path.join("../../Output/SEDR/", dataset)
    adata_h5, n_cluster = usa.get_adata(dataset, data_path='../../Dataset/')
    adata_h5.var_names_make_unique()
    results = pd.DataFrame()
    for i in range(1):
        num = i + 1
        print("===Training epoch:{}====".format(num))
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        res=run_SEDR(adata_h5,save_path,epochs=200)

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
        res["round"] = num

        results = results._append(res, ignore_index=True)
    results.set_index('dataset', inplace=True)
    print(results.head())
    results.to_csv(os.path.join(save_path, "result_scores.csv"))

    res_mean = results.mean()
    res_mean.to_csv(f'{save_path}/{dataset}_mean.csv', header=True)
    res_std = results.std()
    res_std.to_csv(f'{save_path}/{dataset}_std.csv', header=True)
    res_median = results.median()
    res_median.to_csv(f'{save_path}/{dataset}_median.csv', header=True)





