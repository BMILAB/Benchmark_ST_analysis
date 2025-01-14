import os
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
import time
import psutil
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
        # vm = v_measure_score(label_df["True"], label_df["Pred"])
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami=adjusted_mutual_info_score(label_df["True"], label_df["Pred"])

    return ari, nmi,ami


def run_CCST(data_name, n_clusters,read_data_path,save_data_path):
    import os
    import sys
    # sys.path.append("/home/fangzy/project/st_cluster/code/methods/CCST-main")
    import matplotlib
    matplotlib.use('Agg')
    from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
    rootPath = os.path.dirname(sys.path[0]) # sys.path[0] 返回当前路径，os.path.dirname当前路径的父路劲
    os.chdir(rootPath+'/CCST') #在来到子路径

    import argparse
    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    parser.add_argument('--data_type', default='nsc', help='"sc" or "nsc", \
        refers to single cell resolution datasets(e.g. MERFISH) and \
        non single cell resolution data(e.g. ST) respectively')
    # =========================== args ===============================
    parser.add_argument('--data_name', type=str, default=data_name,
                        help="'MERFISH' or 'V1_Breast_Cancer_Block_A_Section_1")
    # 0.8 on MERFISH, 0.3 on ST
    parser.add_argument('--lambda_I', type=float, default=0.3)
    parser.add_argument('--data_path', type=str,
                        default=read_data_path, help='data path')
    parser.add_argument('--save_path', type=str,
                        default=save_data_path, help='data path')

    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--embedding_data_path', type=str,
                        default='Embedding_data')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--DGI', type=int, default=1,
                        help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
    parser.add_argument('--load', type=int, default=0,
                        help='Load pretrained DGI model')
    parser.add_argument('--num_epoch', type=int, default=5000,
                        help='numebr of epoch in training DGI') #5000
    parser.add_argument('--hidden', type=int, default=256,
                        help='hidden channels in DGI')
    parser.add_argument('--PCA', type=int, default=1, help='run PCA or not')
    parser.add_argument('--cluster', type=int, default=1,
                        help='run cluster or not')
    parser.add_argument('--n_clusters', type=int, default=n_clusters,
                        help='number of clusters in Kmeans, when ground truth label is not avalible.')  # 5 on MERFISH, 20 on Breast
    parser.add_argument('--draw_map', type=int,
                        default=1, help='run drawing map')
    parser.add_argument('--diff_gene', type=int, default=0,
                        help='Run differential gene expression analysis')
    args = parser.parse_args(args=['--data_type', "nsc",
                                   '--data_path', '../../Dataset/',
                                   '--model_path', '../../Output/',
                                   '--embedding_data_path', '../../Output/',
                                   '--result_path', '../../Output/',
                                   ])
    args.num_epoch = 5
    if dataset in ["Mouse_brain","Breast_cancer","PDAC"]:
        args.data_type = 'nsc'
        args.lambda_I = 0.3
    elif dataset in ["Stereo","STARmap","SeqFish"]:
        args.data_type = 'sc'
        args.lambda_I = 0.8


    args.data_path =read_data_path
    save_path = save_data_path
    mk_dir(save_path)
    args.result_path = save_path
    args.model_path = save_path
    args.embedding_data_path = save_path
    args.result_path=save_path

    if not os.path.exists(args.embedding_data_path):
        os.makedirs(args.embedding_data_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print('------------------------Model and Training Details--------------------------')
    print(args)

    if args.data_type == 'sc':  # should input a single cell resolution dataset, e.g. MERFISH
        from CCST_merfish_utils import CCST_on_MERFISH
        sc_score=CCST_on_MERFISH(args)
    elif args.data_type == 'nsc':  # should input a non-single cell resolution dataset, e.g. V1_Breast_Cancer_Block_A_Section_1
        from CCST_ST_utils import CCST_on_ST
        sc_score=CCST_on_ST(args)
    else:
        print('Data type not specified')
    return sc_score


n_clusters_map = {"Stereo": 16, "STARmap": 16, "SeqFish": 22,"DLPFC": '5-7', "Breast_cancer": 20, "Mouse_brain": 15,"PDAC": 4}

if __name__ == '__main__':

    # Dataset1 = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    # Dataset2=["Mouse_brain","Breast_cancer","PDAC","SeqFish","Stereo","STARmap"]
    Dataset_test = ['151673']

    for dataset in Dataset_test:
        print(f"==============================The data running now is：{dataset}=============================")
        if dataset.startswith('15'): #if Dataset is DLPFC
            read_data_path=f'../../Dataset/CCST_generate_dataset/DLPFC/{dataset}/'
            save_data_path = f'../../Output/CCST/DLPFC/{dataset}/'
            cluster_num = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7
        else:
            read_data_path = f'../../Dataset/CCST_generate_dataset/{dataset}/'
            save_data_path = f'../../Output/CCST/{dataset}/'
            cluster_num = n_clusters_map[dataset]

        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)

        results = pd.DataFrame()
        best_ari = 0
        for i in range(1):
            start = time.time()
            start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  #
            SC=run_CCST(dataset, cluster_num,read_data_path,save_data_path)
            end = time.time()
            end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  #
            usd_time=end-start
            used_memory=end_MB-start_MB
            print("used time and memory：",  usd_time, used_memory)

            pred = pd.read_csv(f'{save_data_path}predict_types.csv')
            print("pred.shape:",pred.shape)
            ari, nmi,ami = eval_model(pred.iloc[:,2],pred.iloc[:,1])
            res = {}

            res["dataset"] = dataset
            res["ari"] = ari
            res["nmi"] = nmi
            res["ami"]=ami
            res["sc"]=SC
            res["time"] = usd_time
            res["memory"]=used_memory
            res["round"] = i

            results = results._append(res, ignore_index=True)

        results.to_csv(f'{save_data_path}{dataset}_result.csv', header=True)
        results.set_index('dataset', inplace=True)
        print(results.head())
        res_mean = results.mean()
        res_mean.to_csv(f'{save_data_path}{dataset}_mean.csv', header=True)
        res_std = results.std()
        res_std.to_csv(f'{save_data_path}{dataset}_std.csv', header=True)
        res_median = results.median()
        res_median.to_csv(f'{save_data_path}{dataset}_median.csv', header=True) #



