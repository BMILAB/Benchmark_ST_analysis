
from DeepST import run
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
import os
import psutil,time,tracemalloc

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



if __name__ == '__main__':
 method = 'DeepST'
n_clusters_map= {"Stereo": 16, "STARmap": 16, "SeqFish": 22, "Breast_cancer": 20, "Mouse_brain": 15,"PDAC": 4}
# dataset1= ["Stereo","Breast_cancer", "Mouse_brain"]
# dataset2 = ['151507', '151508', '151509', '151510','151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
dataset=['151673']

for data_name in dataset:
    if data_name.startswith('15'):
        data_path = '../../Dataset/DLPFC/'
        save_root =f'../../Output/DeepST/DLPFC/{data_name}/'
    else:
        data_path = '../../Dataset/'
        save_root = f'../../Output/DeepST/{data_name}/'
    os.makedirs(save_root, exist_ok=True)

    #get the DLPFC Clustering number
    if  data_name.startswith('15'):
        n_domains= 5 if data_name in ['151669', '151670', '151671', '151672'] else 7
    else:
        n_domains  = n_clusters_map[data_name]
    print(f"{data_name} has {n_domains} cluster type!")

    results = pd.DataFrame()
    for i in range(1):
        num=i+1
        print("===training epoch:{}====".format(num))
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        deepen = run(
            save_path = save_root,
            task = "Identify_Domain", #### DeepST includes two tasks, one is "Identify_Domain" and the other is "Integration"
            pre_epochs = 800, #### pre_epochs = 800,  choose the number of training
            epochs = 1000, #### epochs = 1000,choose the number of training
            use_gpu = True)

        ###### (1)read adata
        if  data_name.startswith('15') or data_name in ["Breast_cancer", "Mouse_brain"]:
            adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name)
        elif  data_name =='PDAC':
            adata = deepen._get_adata(platform="ST", data_path=data_path, data_name=data_name)
        else:
            adata = deepen._get_adata(platform="StereoSeq", data_path=data_path, data_name=data_name)

        ###### (2) Segment the Morphological Image
        adata = deepen._get_image_crop(adata, data_name=data_name) #未经PCA的结果保存在：adata.obsm['image_feat']

        ###### (3)ata augmentation. spatial_type includes three kinds of "KDTree", "BallTree" and "LinearRegress", among which "LinearRegress"
        adata = deepen._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)

        ###### (4)Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
        graph_dict = deepen._get_graph(adata.obsm["spatial"], distType = "BallTree")

        ###### (5)Enhanced data preprocessing
        data = deepen._data_process(adata, pca_n_comps = 200)  #图像特征用RSNET50 处理完后，保存在这     adata.obsm["X_morphology"] = pca.transform(feature_df.transpose().to_numpy())

        ###### (6)Training models
        deepst_embed = deepen._fit(
            data = data,
            graph_dict = graph_dict,
        )

        adata.obsm["DeepST_embed"] = deepst_embed
        ###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
        adata = deepen._get_cluster_data(adata, n_domains=n_domains, priori = True) #The refine result save in：'DeepST_refine_domain'
        adata.obs['DeepST'] = adata.obs['DeepST_refine_domain']

        ###### Spatial localization map of the spatial domain
        # sc.pl.spatial(adata, color='DeepST_refine_domain', frameon = False, spot_size=15)
        # plt.savefig(os.path.join(save_root, f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=300)
        ###（7） Calculating outcome indicators
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

        ari, nmi, ami = eval_model(adata.obs['DeepST'], adata.obs['ground_truth'])
        SC = silhouette_score(adata.obsm["DeepST_embed"], adata.obs['DeepST'])
        used_adata = adata[adata.obs["ground_truth"].notna()]  # ccc

        res = {}
        res["dataset"] = data_name
        res["ari"] = ari
        res["nmi"] = nmi
        res["ami"] = ami
        res["sc"] = SC
        res["time"] = uesd_time
        res["Memo"] = used_memo
        res["Memo_peak"] = peak
        res["round"] = i + 1
        results = results._append(res, ignore_index=True)
    print(results.head())
    results.to_csv(f'{save_root}/{data_name}_result.csv')

    results.set_index('dataset', inplace=True)
    res_mean = results.mean()
    res_mean.to_csv(f'{save_root}{data_name}_mean.csv', header=True)
    res_std = results.std()
    res_std.to_csv(f'{save_root}{data_name}_std.csv', header=True)
    res_median = results.median()
    res_median.to_csv(f'{save_root}{data_name}_median.csv', header=True)  #

    adata.write(f'{save_root}/DeepST_{data_name}.h5ad')
