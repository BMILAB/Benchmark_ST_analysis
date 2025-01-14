# -*- coding:utf-8 -*-
import time,psutil,tracemalloc
from loss import target_distribution, kl_loss
from util import *
import argparse
from models import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score,adjusted_rand_score
from sklearn.metrics import silhouette_score
import psutil,tracemalloc
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def train_MSpaGCN(opts):
    start = time.time()
    start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

    if opts.dataset.startswith('15'):
        features_adata,features,labels = load_data(opts.dataset,opts.npca)
    else:

        features_adata, features, labels = load_other_data(opts.dataset, opts.input_root,opts.npca)

    adj1, adj2 = load_graph_V1(opts.dataset, features_adata,opts.l)


    model =STMGCN(nfeat=features.shape[1],
                    nhid1=opts.nhid1,
                    nclass=opts.n_cluster
                    )
    if opts.cuda:
        model.cuda()
        features = features.cuda()
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()

    optimizer = optim.Adam(model.parameters(),lr=opts.lr, weight_decay=opts.weight_decay)
    emb = model.mgcn(features,adj1,adj2) #(3639,32)


    if opts.initcluster == "kmeans":
        print("Initializing cluster centers with kmeans, n_clusters known")
        n_clusters=opts.n_cluster
        kmeans = KMeans(n_clusters,n_init=20)
        y_pred = kmeans.fit_predict(emb.detach().cpu().numpy())

    elif opts.initcluster == "louvain":
        print("Initializing cluster centers with louvain,resolution=",opts.res)
        adata=sc.AnnData(emb.detach().cpu().numpy())
        sc.pp.neighbors(adata, n_neighbors=opts.n_neighbors)
        sc.tl.louvain(adata,resolution=opts.res)
        y_pred=adata.obs['louvain'].astype(int).to_numpy()
        n=len(np.unique(y_pred))



    emb=pd.DataFrame(emb.detach().cpu().numpy(),index=np.arange(0,emb.shape[0]))
    Group=pd.Series(y_pred,index=np.arange(0,emb.shape[0]),name="Group")
    Mergefeature=pd.concat([emb,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())

    y_pred_last = y_pred
    with torch.no_grad():
        model.cluster_layer.copy_(torch.tensor(cluster_centers))

    model.train()
    for epoch in range(opts.max_epochs):

        if epoch % opts.update_interval == 0:
            _, tem_q = model(features,adj1,adj2)
            tem_q = tem_q.detach() #calculate q
            p = target_distribution(tem_q) #calculate p

            y_pred = torch.argmax(tem_q, dim=1).cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            y = labels #【-1，6】

            nmi = normalized_mutual_info_score(y, y_pred)
            ari = adjusted_mutual_info_score(y, y_pred)

            print('Iter {}'.format(epoch), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch>0 and delta_label < opts.tol:
                print('delta_label ', delta_label, '< tol ', opts.tol)
                print("Reach tolerance threshold. Stopping training.")
                break

        optimizer.zero_grad()
        x,q = model(features,adj1,adj2)
        loss = kl_loss(q.log(), p)
        loss.backward()
        optimizer.step()


    #save emnddings
    key_added = "STMGCN"
    embeddings = pd.DataFrame(x.detach().cpu().numpy())
    embeddings.index = features_adata.obs_names

    features_adata.obsm[key_added] = embeddings.loc[features_adata.obs_names,].values
    features_adata.obs['pred']=y_pred_last


    end = time.time()
    end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  #
    used_memory = end_MB - start_MB

    ari, nmi, ami = eval_model(y_pred_last, features_adata.obs['Ground Truth'])
    SC = silhouette_score(embeddings, y_pred_last)

    used_adata = features_adata[features_adata.obs["Ground Truth"].notna()]
    SC_revise = silhouette_score(used_adata.obsm["STMGCN"], used_adata.obs['Ground Truth'])

    print(f"sc{SC:5f}")
    results_df = pd.DataFrame()
    res = {}
    res["dataset"] = dataset
    res["ari"] = ari
    res["nmi"] = nmi
    res["ami"] = ami
    res["sc"] = SC
    res["time"] = end-start
    res['Memo']=used_memory
    res["sc_revise"]=SC_revise
    return features_adata,res

def parser_set(n_cluster,dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--nhid1',type=int,default=32)
    parser.add_argument('--n_cluster',default=n_cluster,type=int)
    parser.add_argument('--max_epochs',default=2000,type=int) #2000
    parser.add_argument('--update_interval',default= 3,type=int)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--weight_decay',default=0.001,type=float)
    parser.add_argument('--dataset', type=str, default=dataset)
    # parser.add_argument('--sicle', default="151673", type=str)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--l', default=1, type=float)
    parser.add_argument('--npca', default=50, type=int)
    parser.add_argument('--n_neighbors',type=int,default=10)
    parser.add_argument('--initcluster', default="kmeans", type=str)
    parser.add_argument('--input_root', default=f'../../Dataset/', type=str)
    parser.add_argument('--output_root', default=f'../../Output/', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    return args

n_clusters_map = {"Mouse_hippocampus": 10, "Mouse_olfactory_slide_seqv2": 9, "MOB_without_label": 7,
                  "PDAC": 4, "Breast_cancer": 20, "Mouse_brain": 15,
                  "SeqFish": 22, "STARmap": 16,"Stereo":16}

if __name__ == "__main__":
    dataset = ["Mouse_brain", "Breast_cancer", "PDAC", "Stereo", "STARmap"]
    dataset2 = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674','151675', '151676']
    Dataset_test=['151673']
for dataset in Dataset_test:
    print(f"====================begin test on {dataset}======================================")
    if dataset.startswith('15'):
        n_domains = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7
        save_data_path = f'../../Output/STMGCN/DLPFC/{dataset}/'
    else:
        n_domains = n_clusters_map[dataset]
        save_data_path = f'../../Output/STMGCN/{dataset}/'
    mk_dir(save_data_path)
    print(f"{dataset} has {n_domains} cluster type!")
    opts = parser_set(n_domains,dataset)
    print(opts)

    results_df=pd.DataFrame()
    for i in range(1):
        random_seed = 0
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        adata,res=train_MSpaGCN(opts)
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
        results_df = results_df._append(res, ignore_index=True)


    print( results_df.head())

    results_df.to_csv(save_data_path + "/{}_result.csv".format(dataset), header=True)
    adata.write(f'{save_data_path}/STMGCN_{dataset}.h5ad')


    results_df.set_index('dataset', inplace=True)
    res_mean = results_df.mean()
    res_mean.to_csv(f'{save_data_path}{dataset}_mean.csv', header=True)
    res_std =results_df.std()
    res_std.to_csv(f'{save_data_path}{dataset}_std.csv', header=True)
    res_median = results_df.median()
    res_median.to_csv(f'{save_data_path}{dataset}_median.csv', header=True)  #

























































