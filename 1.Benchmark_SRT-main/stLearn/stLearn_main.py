import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc
import stlearn as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import time,psutil,tracemalloc
import squidpy as sq
import seaborn as sns

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def plot_clustering(adata, dataset,colors, savepath=None):

    adata.obs['x_pixel'] = adata.obsm['spatial'][:, 0]
    adata.obs['y_pixel'] = adata.obsm['spatial'][:, 1]

    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    ax1.axes.invert_yaxis()
    sc.pl.scatter(adata, alpha=1, x="x_pixel", y="y_pixel", color=colors, title=f'Spatial distribution of the {dataset}',
                  palette=sns.color_palette('plasma', 7), show=False, ax=ax1)
    ax1.set_aspect('equal', 'box')
    ax1.axis('off')
    plt.tight_layout()
    plt.savefig(f'{savepath}/{dataset}_Spatial_plot.jpg',dpi=400,bbox_inches='tight')
    plt.show()


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

        library_id = dataset
        quality = 'hires'
        raw.uns["spatial"][library_id]["use_quality"] = quality  # quality='hires'
        raw.var_names_make_unique()
        image_coor = raw.obsm["spatial"]
        raw.obs["imagecol"] = image_coor[:, 0]
        raw.obs["imagerow"] = image_coor[:, 1]


    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset)  # please replace 'file_fold' with the download path
        raw = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        raw.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
        raw.obs['ground_truth'] = df_meta.loc[raw.obs_names, 'fine_annot_type'].values
        n_clusters = 20

        library_id = 'V1_Breast_Cancer_Block_A_Section_1'
        quality = 'hires'
        raw.uns["spatial"][library_id]["use_quality"] = quality  # quality='hires'
        raw.var_names_make_unique()

        image_coor = raw.obsm["spatial"]
        raw.obs["imagecol"] = image_coor[:, 0]
        raw.obs["imagerow"] = image_coor[:, 1]



    if dataset == "Mouse_brain":
        raw = sq.datasets.visium_hne_adata()
        raw.var_names_make_unique()
        raw.obs['ground_truth'] = raw.obs["cluster"]

        library_id = 'V1_Adult_Mouse_Brain'
        quality = 'hires'
        raw.uns["spatial"][library_id]["use_quality"] = quality  # quality='hires'
        raw.var_names_make_unique()
        raw.obs['ground_truth'] = raw.obs["cluster"]

        image_coor = raw.obsm["spatial"]
        raw.obs["imagecol"] = image_coor[:, 0]
        raw.obs["imagerow"] = image_coor[:, 1]

        n_clusters = 15


    if dataset == "PDAC":
        file_fold = data_path + str(dataset)

        raw = sc.read(file_fold + "/ST_PDAC_B.h5ad")
        n_clusters = len(raw.obs['ground_truth'].unique())

        raw.uns["spatial"] = dict()
        library_id = 'PDAC'
        raw.uns["spatial"][library_id] = dict()
        quality = 'hires'
        raw.uns["spatial"][library_id]['use_quality'] = dict()
        raw.uns["spatial"][library_id]["use_quality"] = quality  # quality='hires'
        raw.var_names_make_unique()
        hires_image_file =  file_fold+'/V1/PDAC-B-ST1-HE.jpg'
        from matplotlib.image import imread
        raw.uns["spatial"][library_id]['images'] = dict()
        raw.uns["spatial"][library_id]['images']['hires'] =  imread(hires_image_file)
        raw.uns["spatial"][library_id]["use_quality"] = "hires"

        image_coor = raw.obsm["spatial"]
        raw.obs["imagecol"] = image_coor[:, 0]
        raw.obs["imagerow"] = image_coor[:, 1]

        raw.obs["array_col"] = image_coor[:, 0]
        raw.obs["array_row"] = image_coor[:, 1]

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + '/Stereo_DPI.h5ad')
        print(raw.shape)  # (8243, 22144)
        n_clusters = len(raw.obs['ground_truth'].unique())


        ####### Complete Stereo-seq data for HE feature extraction #######
        image_coor = raw.obsm["spatial"]
        raw.obs["imagecol"] = image_coor[:, 0]
        raw.obs["imagerow"] = image_coor[:, 1]
        raw.obs["array_col"] = image_coor[:, 0]
        raw.obs["array_row"] = image_coor[:, 1]

        raw.uns["spatial"] = dict()
        library_id = 'Stereo'
        raw.uns["spatial"][library_id] = dict()

        hires_image_file = f'{file_fold}/Stereo_2DPI.png'
        from matplotlib.image import imread
        raw.uns["spatial"][library_id]['images'] = dict()
        raw.uns["spatial"][library_id]['images']['hires'] =imread(hires_image_file)
        raw.uns["spatial"][library_id]["use_quality"] = "hires"


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


if __name__ == '__main__':
    dataset1 = ["Stereo", "Breast_cancer", "Mouse_brain", ]
    dataset2 = ['151507', '151508', '151509', '151510','151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    Dataset_test = ['Stereo']
for sample in Dataset_test:
    print(f"====================begin test on {sample}======================================")
    if sample.startswith('15'):
        save_path = f'../../Output/stLearn/DLPFC/{sample}/'
        TILE_PATH = Path('../../Output/stLearn/DLPFC/{}/tile/'.format(sample))

    else:
        save_path = f'../../Output/stLearn/{sample}/'
        TILE_PATH = Path('../../Output/stLearn/{}/tile/'.format(sample))
    mk_dir(save_path)
    mk_dir(TILE_PATH)

    data,n_clusters = read_data(sample)
    le = LabelEncoder()
    ground_truth_le = le.fit_transform(list(data.obs['ground_truth']))

    start = time.time()
    tracemalloc.start()
    start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

    # pre-processing for gene count table
    st.pp.filter_genes(data,min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data) #(3639,21842)

    # run PCA for gene expression data
    st.em.run_pca(data,n_comps=15)

    # pre-processing for spot image
    st.pp.tiling(data, TILE_PATH) # 切割成上spot

    st.pp.extract_feature(data)
    HE_feature=pd.DataFrame(data=data.obsm["X_tile_feature"])
    print("HE_feature.shape",HE_feature.shape)
    HE_feature.to_csv(f'{save_path}/mouse_brain_HE.csv' )


    # stSME
    st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
    data_ = data.copy()
    data_.X = data_.obsm['raw_SME_normalized']
    st.pp.scale(data_)
    st.em.run_pca(data_,n_comps=30)
    st.tl.clustering.kmeans(data_, n_clusters=n_clusters, use_data="X_pca", key_added="X_pca_kmeans")

    methods_ = "stSME_disk"

    from utils import eval_model
    ari, nmi, ami = eval_model(data_.obs["X_pca_kmeans"], ground_truth_le)
    SC = silhouette_score(  data_.obsm["X_pca"], data_.obs["X_pca_kmeans"])

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

    res = {}
    res['dataset'] = sample
    res["ari"] = ari
    res["nmi"] = nmi
    res["ami"] = ami
    res["sc"] = SC
    res["time"] = uesd_time
    res["Memo"] = used_memo
    res["Memo_peak"] = peak

    results = pd.DataFrame()
    results = results._append(res, ignore_index=True)
    print(results.head())

    results.to_csv(f'{save_path}stlearn_result.csv')

    plt.savefig(f'{save_path}cluster.png')
    data_.obs.to_csv(f'{save_path}metadata.tsv', sep='\t', index=False)

    df_PCA = pd.DataFrame(data = data_.obsm['X_pca'], index = data_.obs.index)
    df_PCA.to_csv(f'{save_path}PCs.tsv', sep='\t')
    data_.write(f"{save_path}/stLearn_{sample}_kmeans.h5ad")
