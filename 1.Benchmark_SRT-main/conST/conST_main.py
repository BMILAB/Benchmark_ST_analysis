import os
import torch
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,silhouette_samples
from sklearn.metrics import silhouette_score
from src.utils_func import plot_clustering
import psutil,time,tracemalloc

def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami=adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
    return  ari,nmi,ami

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# set seed before every run
def seed_torch(seed):
    import random
    import numpy as np
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def refine_function(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print(
            "Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values(ascending=False)
        nbs = dis_tmp[0:num_nbs+1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs/2) and (np.max(v_c) > num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


def run_conST(adata_h5, dataset,
              device=torch.device(
                  'cuda:0' if torch.cuda.is_available() else 'cpu'),
              save_data_path="../../Output/conST/",
              n_clusters=6):
    import sys
    # sys.path.append("/home/fangzy/project/st_cluster/code/methods/conST-main")
    from src.graph_func import graph_construction
    from src.utils_func import mk_dir, adata_preprocess, load_ST_file, res_search_fixed_clus, plot_clustering
    from src.training import conST_training
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10,
                        help='parameter k in spatial graph')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                        help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.') #200
    parser.add_argument('--cell_feat_dim', type=int,
                        default=300, help='Dim of PCA')
    parser.add_argument('--feat_hidden1', type=int,
                        default=100, help='Dim of DNN hidden 1-layer.')
    parser.add_argument('--feat_hidden2', type=int,
                        default=20, help='Dim of DNN hidden 2-layer.')
    parser.add_argument('--gcn_hidden1', type=int, default=32,
                        help='Dim of GCN hidden 1-layer.')
    parser.add_argument('--gcn_hidden2', type=int, default=8,
                        help='Dim of GCN hidden 2-layer.')
    parser.add_argument('--p_drop', type=float,
                        default=0.2, help='Dropout rate.')
    parser.add_argument('--use_img', type=bool,
                        default=False, help='Use histology images.')
    parser.add_argument('--img_w', type=float, default=0.1,
                        help='Weight of image features.')
    parser.add_argument('--use_pretrained', type=bool,
                        default=True, help='Use pretrained weights.')
    parser.add_argument('--using_mask', type=bool,
                        default=False, help='Using mask for multi-dataset.')
    parser.add_argument('--feat_w', type=float, default=10,
                        help='Weight of DNN loss.')
    parser.add_argument('--gcn_w', type=float, default=0.1,
                        help='Weight of GCN loss.')
    parser.add_argument('--dec_kl_w', type=float,
                        default=10, help='Weight of DEC loss.')
    parser.add_argument('--gcn_lr', type=float, default=0.01,
                        help='Initial GNN learning rate.')
    parser.add_argument('--gcn_decay', type=float,
                        default=0.01, help='Initial decay rate.')
    parser.add_argument('--dec_cluster_n', type=int,
                        default=10, help='DEC cluster number.')
    parser.add_argument('--dec_interval', type=int,
                        default=20, help='DEC interval nnumber.')
    parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--beta', type=float, default=100,
                        help='beta value for l2c')
    parser.add_argument('--cont_l2l', type=float, default=0.3,
                        help='Weight of local contrastive learning loss.')
    parser.add_argument('--cont_l2c', type=float, default=0.1,
                        help='Weight of context contrastive learning loss.')
    parser.add_argument('--cont_l2g', type=float, default=0.1,
                        help='Weight of global contrastive learning loss.')

    parser.add_argument('--edge_drop_p1', type=float, default=0.1,
                        help='drop rate of adjacent matrix of the first view')
    parser.add_argument('--edge_drop_p2', type=float, default=0.1,
                        help='drop rate of adjacent matrix of the second view')
    parser.add_argument('--node_drop_p1', type=float, default=0.2,
                        help='drop rate of node features of the first view')
    parser.add_argument('--node_drop_p2', type=float, default=0.3,
                        help='drop rate of node features of the second view')

    # ______________ Eval clustering Setting ______________
    parser.add_argument('--eval_resolution', type=int,
                        default=1, help='Eval cluster number.')
    parser.add_argument('--eval_graph_n', type=int,
                        default=20, help='Eval graph kN tol.')

    params = parser.parse_args(args=['--k', '10', '--knn_distanceType',
                               'euclidean', '--epochs', '200', '--use_pretrained', 'False'])


    params.device = device
    params.save_path = mk_dir(f'{save_data_path}/{dataset}')
    # start = time.time()
    # start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim) #300
    graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], params)
    params.cell_num = adata_h5.shape[0]

    if ("ground_truth" in adata_h5.obs.keys()):
        n_clusters = len(set(adata_h5.obs["ground_truth"].dropna()))
    else:
        n_clusters = n_clusters

    print("===Dataset:{},n_clusters:{}====".format(dataset,n_clusters))

    if params.use_img:
        img_transformed = np.load('./MAE-pytorch/extracted_feature.npy')
        img_transformed = (img_transformed - img_transformed.mean()) / \
            img_transformed.std() * adata_X.std() + adata_X.mean()
        conST_net = conST_training(
            adata_X, graph_dict, params, n_clusters, img_transformed)
    else:
        conST_net = conST_training(adata_X, graph_dict, params, n_clusters)

    conST_net.pretraining()
    conST_net.major_training()

    conST_embedding = conST_net.get_embedding()

    # np.save(f'{params.save_path}/conST_result.npy', conST_embedding)
    adata_h5.obsm["embedding"] = conST_embedding
    sc.pp.neighbors(adata_h5, n_neighbors=params.eval_graph_n,use_rep='embedding')
    eval_resolution = res_search_fixed_clus(adata_h5, n_clusters)
    print("resolution:",eval_resolution)
    cluster_key = "conST_leiden"
    sc.tl.leiden(adata_h5, key_added=cluster_key, resolution=eval_resolution)

    index = np.arange(start=0, stop=adata_X.shape[0]).tolist()
    index = [str(x) for x in index]

    dis = graph_dict['adj_norm'].to_dense().numpy(
    ) + np.eye(graph_dict['adj_norm'].shape[0])

    if dataset.startswith('15'):
        dataset='DLPFC'
    refine_map = {"Breast_cancer": "hexagon", "Mouse_brain": "hexagon",
                  "STARmap": "square", "Mouse_olfactory": "hexagon","DLPFC":"hexagon" }
    refine = refine_function(sample_id=index, shape=refine_map[dataset],
                             pred=adata_h5.obs['leiden'].tolist(), dis=dis)
    # end = time.time()
    # end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  #
    # used_memory = end_MB - start_MB

    res = {}
    adata_h5.obs['refine'] = refine
    if ("ground_truth" in adata_h5.obs.keys()):
        ari_r,nmi_r,ami_r = eval_model(adata_h5.obs['refine'], adata_h5.obs['ground_truth'])
        ari,nmi,ami= eval_model(adata_h5.obs['conST_leiden'], adata_h5.obs['ground_truth'])
        print("adata_h5.obsm[embedding].shape",adata_h5.obsm["embedding"].shape)
        print("adata_h5.obs['conST_leiden']",adata_h5.obs['conST_leiden'].value_counts())
        SC = silhouette_score(adata_h5.obsm["embedding"], adata_h5.obs['conST_leiden'])
        SC_r = silhouette_score(adata_h5.obsm["embedding"], adata_h5.obs['refine'])

        used_adata = adata_h5[adata_h5.obs["ground_truth"].notna()] #因为ground_truth中可能Na,所以筛选去除
        SC_revise = silhouette_score(used_adata.obsm["embedding"], used_adata.obs['ground_truth'])
        print("SC_revise:", SC_revise)


        res["dataset"] = dataset
        res["ari"] = ari
        res["nmi"] = nmi
        res["ami"] = ami
        res["sc"] = SC
        res["SC_revise"] =SC_revise

        res["ari_1"] = ari_r
        res["nmi_1"] = nmi_r
        res["ami_1"] = ami_r
        res["sc_1"] = SC_r
    adata_h5.obs["pred_label"] = refine

    return res, adata_h5


import sys
sys.path.append('../')
import utils_for_all as usa
if __name__ == '__main__':


    dataset2 = ['151507', '151508', '151509', '151510','151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    Dataset_test = ['151673']

for dataset in Dataset_test:
    best_ari = 0
    if dataset.startswith('15'):
        save_path =f'../../Output/conST/DLPFC/{dataset}/'
    else:
        save_path = f'../../Output/conST/{dataset}/'
    mk_dir(save_path)

    adata, n_clusters = usa.get_adata(dataset, data_path='../../Dataset/')
    adata.var_names_make_unique()
    print(adata.shape)


    results = pd.DataFrame()
    for i in range(1):
        num=i+1
        print("===Training epoch:{}====".format(num))
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        res, adata_h5 = run_conST(adata.copy(), dataset,n_clusters=n_clusters)

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
        res["time"] = end - start
        res["Memo"] =  used_memo
        res["Memo_peak"] = peak
        res["round"] = i+1
        results = results._append(res, ignore_index=True)
        results.to_csv(save_path + "result.csv", header=True)
        adata_h5.write_h5ad(save_path+str(dataset)+".h5ad")

        if dataset in ["Breast_cancer", "Mouse_brain","Stereo"]:
            key='leiden'
            savepath = f'{save_path}/conST_leiden_plot.jpg'
            plot_clustering(adata_h5, key, savepath=savepath)
            cluster_key = 'refine'
            savepath = f'{save_path}/conST_leiden_plot_refined.jpg'
            plot_clustering(adata_h5, cluster_key, savepath=savepath)

    results.set_index('dataset', inplace=True)
    print(results.head())
    res_mean = results.mean()
    res_mean.to_csv(f'{save_path}{dataset}_mean.csv', header=True)
    res_std = results.std()
    res_std.to_csv(f'{save_path}{dataset}_std.csv', header=True)
    res_median = results.median()
    res_median.to_csv(f'{save_path}{dataset}_median.csv', header=True) #

