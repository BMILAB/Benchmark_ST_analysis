from __future__ import division
from __future__ import print_function

import torch.optim as optim
from utils import *
from models import Spatial_MGCN
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import random
import time,psutil,tracemalloc
from scipy.sparse import issparse

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def load_DLPFC_data(dataset):
    print("load DLPFC dataset:")
    path = "../../Dataset/Spatial_MGCN_generate/DLPFC/" + dataset + "/Spatial_MGCN.h5ad"
    adata = sc.read_h5ad(path)

    adata.X=adata.X.todense() if issparse(adata.X) else adata.X
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground_truth']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg


def load_data(dataset):
    if dataset.startswith('15'):
        path = "../../Dataset/Spatial_MGCN_generate/DLPFC/" + dataset + "/"
    else:
        path = "../../Dataset/Spatial_MGCN_generate/" + dataset + "/"


    adata = sc.read_h5ad(f'{path}Spatial_MGCN.h5ad')
    features = torch.FloatTensor(adata.X)
    if dataset=='Stereo':
        adata.obs['ground_truth']=adata.obs['Annotation']
    labels = adata.obs['ground_truth']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nsadj, nfadj, graph_nei, graph_neg


def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, con_loss, total_loss


if __name__ == "__main__":
 parse = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


Dataset_test= [ '151673']
for dataset in Dataset_test:
    if dataset.startswith('15'):
        config_file = './config/DLPFC.ini'
        savepath = f'../../Output/Spatial_MGCN/DLPFC/{dataset}/'
        adata, features, labels, sadj, fadj, graph_nei, graph_neg = load_DLPFC_data(dataset)
    else:
        config_file = './config/' + dataset + '.ini'
        savepath = f'../../Output/Spatial_MGCN/{dataset}/'
        adata, features, labels, sadj, fadj, graph_nei, graph_neg = load_data(dataset)
    mk_dir(savepath)


    config = Config(config_file)
    cuda = not config.no_cuda and torch.cuda.is_available()
    use_seed = not config.no_seed

    _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
    ground = torch.LongTensor(ground)
    config.n = len(ground)
    config.class_num = len(ground.unique())

    results = pd.DataFrame()
    for i in range(1):
        num = i + 1
        print("===Train epoch{}====".format(num))
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        config.epochs = config.epochs + 1

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)
        model = Spatial_MGCN(nfeat=config.fdim,
                             nhid1=config.nhid1,
                             nhid2=config.nhid2,
                             dropout=config.dropout)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)
        epoch_max = 0
        ari_max = 0
        idx_max = []
        mean_max = []
        emb_max = []
        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, con_loss, total_loss = train()
            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss), ' con_loss = {:.2f}'.format(con_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_

            if dataset.startswith('15'):
                df = pd.DataFrame({'labels': labels, 'idx': idx})
                df_cleaned = df.dropna()
                ground_filter = df_cleaned['labels'].values
                pred_filter = df_cleaned['idx'].values
                ari_res = metrics.adjusted_rand_score(ground_filter, pred_filter)
            else:
                ari_res = metrics.adjusted_rand_score(labels, idx)

            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb

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
        label_df = pd.DataFrame({"True": labels, "Pred": idx,"pred_max":idx_max}).dropna()


        ari,nmi,ami= eval_model(label_df["Pred"],label_df["True"])
        SC = silhouette_score(emb, idx)
        res = {}
        res["dataset"] = dataset
        res["ari"] = ari
        res["nmi"] = nmi
        res["ami"] = ami
        res["sc"] = SC

        ari_max = adjusted_rand_score(label_df["pred_max"],label_df["True"])
        nmi_max = normalized_mutual_info_score(label_df["pred_max"],label_df["True"])
        ami_max = adjusted_mutual_info_score(label_df["pred_max"],label_df["True"])
        res['ari_max'] = ari_max
        res['nmi_max'] = nmi_max
        res['ami_max'] = ami_max
        res['SC_max'] = silhouette_score(emb_max, idx_max)
        res["time"] = uesd_time
        res["Memo"] = used_memo
        res["Memo_peak"] = peak
        res["round"] = i + 1
        results = results._append(res, ignore_index=True)

    results.set_index('dataset', inplace=True)
    print(results.head())
    results.to_csv(os.path.join(savepath, "result_scores.csv"))

    res_mean = results.mean()
    res_mean.to_csv(f'{savepath}{dataset}_mean.csv', header=True)
    res_std = results.std()
    res_std.to_csv(f'{savepath}{dataset}_std.csv', header=True)
    res_median = results.median()
    res_median.to_csv(f'{savepath}{dataset}_median.csv', header=True)

    adata.obs['idx'] = idx_max.astype(str)
    adata.obsm['emb'] = emb_max
    adata.obsm['mean'] = mean_max
    if config.gamma == 0:
        title = 'Spatial_MGCN-w/o'
        pd.DataFrame(emb_max).to_csv(savepath + 'Spatial_MGCN_no_emb.csv', header=None, index=None)
        pd.DataFrame(idx_max).to_csv(savepath + 'Spatial_MGCN_no_idx.csv', header=None, index=None)
        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig(savepath + 'Spatial_MGCN_no.jpg', bbox_inches='tight', dpi=600)
        plt.show()
    else:
        title = 'Spatial_MGCN'
        pd.DataFrame(emb_max).to_csv(savepath + 'Spatial_MGCN_emb.csv', header=None, index=None)
        pd.DataFrame(idx_max).to_csv(savepath + 'Spatial_MGCN_idx.csv', header=None, index=None)
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        adata.write(f'{savepath}Spatial_MGCN_{dataset}.h5ad' )
        adata.write(savepath + 'Spatial_MGCN_SeqFish.h5ad')


