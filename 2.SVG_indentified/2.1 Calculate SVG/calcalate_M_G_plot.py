<<<<<<< HEAD

import numpy as np
import scanpy as sc
from SpaGCN.calculate_moran_I import Moran_I  # No module named 'SpaGCN.calculate_moran_I'; 'SpaGCN' is not a package
from SpaGCN.calculate_moran_I import Geary_C #从site-package
from SpaGCN.calculate_adj import calculate_adj_matrix
import os
import SpaGCN as spg
import pandas as pd
from scipy.sparse import issparse

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def get_pred_adata(method):

   DF=pd.read_csv('../Pred_label/151673_pred_label.csv',index_col=0)
   label=DF[method].astype(int).values
   return label



import sys
sys.path.append('../../1.Benchmark_SRT-main/')
import utils_for_all as usa

if __name__ == '__main__':

    dataset='151673'  #'STARmap','Stereo','SeqFish','151673'

if dataset.startswith('15'):
    out_dir = f'../SVG_indentified_Output/DLPFC/{dataset}/'
    M_G_output = f"../result_M_G_score/DLPFC/{dataset}/"
else:
    out_dir = f'../SVG_indentified_Output/{dataset}/'
    M_G_output = f"../result_M_G_score/{dataset}/"

mk_dir(out_dir)
mk_dir(M_G_output)


raw,n_clusters=usa.get_adata(dataset, data_path='../../Dataset/')
raw.var_names_make_unique()
raw.X = raw.X.A if issparse(raw.X) else raw.X
print(f"{dataset}'s shape:{raw.shape}")
sc.pp.log1p(raw)
x_array = raw.obs["x_array"]
y_array = raw.obs["y_array"]


methotd_set=["Seurat","BayesSpace","STAGATE","conST","SEDR","stLearn","SpaceFlow","SCGDL","GraphST","CCST","Spatial_MGCN","SpaGCN","DeepST","STMGCN"] #全部的14个方法
pred_dict={}
for method in methotd_set:
    print( f"*****************************************The {method}  is now being calculated************************************")
    raw.obs['pred']=get_pred_adata(method)
    adata=raw

    # Set filtering criterials
    min_in_group_fraction = 0.8
    min_in_out_group_ratio = 1
    min_fold_change = 1.5

    adj_2d =calculate_adj_matrix(x=x_array, y=y_array, histology=False) # Calculate the Euclidean distance between the spot
    start, end = np.quantile(adj_2d[adj_2d != 0], q=0.001), np.quantile(adj_2d[adj_2d != 0], q=0.1)

    ###calculate the distribution of predicted label class
    cluster_num = dict()
    for i in adata.obs['pred']:
        cluster_num[i] = cluster_num.get(i, 0) + 1
    cluster_num_backup = cluster_num
    cluster_num_backup = [(k, v) for k, v in
                          cluster_num_backup.items()]
    cluster_num_backup.sort(key=lambda x: -x[0])

    cluster_num_backup = [item for item in cluster_num_backup if item[1] != 1]  # Filtering out domains with only one spot,which easily report errors
    print(f'{method}  has {len(cluster_num)} class type：', "\n", cluster_num_backup)
    cluster_type_coverd = sorted([tup[0] for tup in cluster_num_backup])


    all_SVG = pd.DataFrame()
    for target in cluster_type_coverd: #7
        print("==============Now is calcalating the domain:",target,"===============================")
        #Find suitable radius for each target domain in preparation for finding neighbors
        r = spg.search_radius(
            target_cluster=target,
            cell_id=adata.obs.index.tolist(),
            x=x_array,
            y=y_array,
            pred=adata.obs["pred"].tolist(),
            start=start,
            end=end,
            num_min=10,
            num_max=14,
            max_run=100,
        )
        if r == None:
            r = 3.162277660168379
            print(f"***********{dataset}:{method}:{target}******r==none,reset r=={r}**************************")

        nbr_domians = spg.find_neighbor_clusters(
            target_cluster=target,
            cell_id=adata.obs.index.tolist(),
            x=adata.obs["x_array"].tolist(),
            y=adata.obs["y_array"].tolist(),
            pred=adata.obs["pred"].tolist(),  # usd cluster result for finding SVG
            radius=r,
            ratio=1/2,
        )
        print(f"target {target} has {len(nbr_domians)} neighbor domian：{nbr_domians}")
        nbr_domians = nbr_domians[0:3]  #get the top3 domian
        de_genes_info = spg.rank_genes_groups(
            input_adata=adata,
            target_cluster=target,
            nbr_list=nbr_domians,
            label_col="pred",
            adj_nbr=True,
            log=True,
        )

        de_genes_info = de_genes_info[(de_genes_info["pvals_adj"] < 0.05)]
        print("1.Left after pvals_adj < 0.05 filtering：",de_genes_info.shape)
        filtered_info = de_genes_info
        filtered_info = filtered_info[
            (filtered_info["pvals_adj"] < 0.05)
            & (filtered_info["in_out_group_ratio"] > min_in_out_group_ratio)
            & (filtered_info["in_group_fraction"] > min_in_group_fraction)
            & (filtered_info["fold_change"] > min_fold_change)
        ]
        print("2.After multiple screenings there are still：", filtered_info.shape)

        filtered_info = filtered_info.sort_values(by="in_group_fraction", ascending=False)
        filtered_info["target_dmain"] = target
        filtered_info["neighbors"] = str(nbr_domians)
        SVG_num=len(filtered_info["genes"].tolist())
        print(f"target{target} find {SVG_num}个SVG，namely: {filtered_info['genes'].tolist()} " )
        all_SVG = pd.concat([all_SVG, filtered_info], ignore_index=True)  # 另外，需要把每次for得到的SVG拼接起来，最终统一返回


    SVG_filter=all_SVG.loc[:, ~all_SVG.columns.duplicated()]
    SVG_filter=SVG_filter.drop_duplicates()
    print("{} after filter：SVG shape:{}".format(method,SVG_filter.shape))
    all_SVG.to_csv(f'{out_dir}{method}_{dataset}_SVG_identified_{all_SVG.shape[0]}.csv')

    ##  After obtaining SVG, then calculate the moran and Geary scores
    SVG=all_SVG['genes'].values
    # Matching the gene expression corresponding to these SVGs from raw gene expression
    adata_DF=adata.to_df()
    SVG_count=adata_DF.loc[:,SVG]
    SVG_count=SVG_count.loc[:, ~SVG_count.columns.duplicated()]

    Moran_score=Moran_I(SVG_count,adata.obs["x_array"].values,adata.obs["y_array"].values)
    Geary_score=Geary_C(SVG_count,adata.obs["x_array"],adata.obs["y_array"])
    # print(f'Moran_score.mean: {Moran_score.mean():.5f} '+f'\nGeary_score.mean: {Geary_score.mean():.5f}')
    ## Save the Moran's I and Geary's C
    M_G_score=Moran_score.to_frame(name="Moran")
    M_G_score['Geary']=Geary_score.values
    M_G_score['Geary_revise'] = 1-Geary_score.values
    M_G_score.to_csv(f'{ M_G_output}{method}_M_G_score.csv')
=======

import numpy as np
import scanpy as sc
from SpaGCN.calculate_moran_I import Moran_I  # No module named 'SpaGCN.calculate_moran_I'; 'SpaGCN' is not a package
from SpaGCN.calculate_moran_I import Geary_C #从site-package
from SpaGCN.calculate_adj import calculate_adj_matrix
import os
import SpaGCN as spg
import pandas as pd
from scipy.sparse import issparse

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def get_pred_adata(method):

   DF=pd.read_csv('../Pred_label/151673_pred_label.csv',index_col=0)
   label=DF[method].astype(int).values
   return label



import sys
sys.path.append('../../1.Benchmark_SRT-main/')
import utils_for_all as usa

if __name__ == '__main__':

    dataset='151673'  #'STARmap','Stereo','SeqFish','151673'

if dataset.startswith('15'):
    out_dir = f'../SVG_indentified_Output/DLPFC/{dataset}/'
    M_G_output = f"../result_M_G_score/DLPFC/{dataset}/"
else:
    out_dir = f'../SVG_indentified_Output/{dataset}/'
    M_G_output = f"../result_M_G_score/{dataset}/"

mk_dir(out_dir)
mk_dir(M_G_output)


raw,n_clusters=usa.get_adata(dataset, data_path='../../Dataset/')
raw.var_names_make_unique()
raw.X = raw.X.A if issparse(raw.X) else raw.X
print(f"{dataset}'s shape:{raw.shape}")
sc.pp.log1p(raw)
x_array = raw.obs["x_array"]
y_array = raw.obs["y_array"]


methotd_set=["Seurat","BayesSpace","STAGATE","conST","SEDR","stLearn","SpaceFlow","SCGDL","GraphST","CCST","Spatial_MGCN","SpaGCN","DeepST","STMGCN"] #全部的14个方法
pred_dict={}
for method in methotd_set:
    print( f"*****************************************The {method}  is now being calculated************************************")
    raw.obs['pred']=get_pred_adata(method)
    adata=raw

    # Set filtering criterials
    min_in_group_fraction = 0.8
    min_in_out_group_ratio = 1
    min_fold_change = 1.5

    adj_2d =calculate_adj_matrix(x=x_array, y=y_array, histology=False) # Calculate the Euclidean distance between the spot
    start, end = np.quantile(adj_2d[adj_2d != 0], q=0.001), np.quantile(adj_2d[adj_2d != 0], q=0.1)

    ###calculate the distribution of predicted label class
    cluster_num = dict()
    for i in adata.obs['pred']:
        cluster_num[i] = cluster_num.get(i, 0) + 1
    cluster_num_backup = cluster_num
    cluster_num_backup = [(k, v) for k, v in
                          cluster_num_backup.items()]
    cluster_num_backup.sort(key=lambda x: -x[0])

    cluster_num_backup = [item for item in cluster_num_backup if item[1] != 1]  # Filtering out domains with only one spot,which easily report errors
    print(f'{method}  has {len(cluster_num)} class type：', "\n", cluster_num_backup)
    cluster_type_coverd = sorted([tup[0] for tup in cluster_num_backup])


    all_SVG = pd.DataFrame()
    for target in cluster_type_coverd: #7
        print("==============Now is calcalating the domain:",target,"===============================")
        #Find suitable radius for each target domain in preparation for finding neighbors
        r = spg.search_radius(
            target_cluster=target,
            cell_id=adata.obs.index.tolist(),
            x=x_array,
            y=y_array,
            pred=adata.obs["pred"].tolist(),
            start=start,
            end=end,
            num_min=10,
            num_max=14,
            max_run=100,
        )
        if r == None:
            r = 3.162277660168379
            print(f"***********{dataset}:{method}:{target}******r==none,reset r=={r}**************************")

        nbr_domians = spg.find_neighbor_clusters(
            target_cluster=target,
            cell_id=adata.obs.index.tolist(),
            x=adata.obs["x_array"].tolist(),
            y=adata.obs["y_array"].tolist(),
            pred=adata.obs["pred"].tolist(),  # usd cluster result for finding SVG
            radius=r,
            ratio=1/2,
        )
        print(f"target {target} has {len(nbr_domians)} neighbor domian：{nbr_domians}")
        nbr_domians = nbr_domians[0:3]  #get the top3 domian
        de_genes_info = spg.rank_genes_groups(
            input_adata=adata,
            target_cluster=target,
            nbr_list=nbr_domians,
            label_col="pred",
            adj_nbr=True,
            log=True,
        )

        de_genes_info = de_genes_info[(de_genes_info["pvals_adj"] < 0.05)]
        print("1.Left after pvals_adj < 0.05 filtering：",de_genes_info.shape)
        filtered_info = de_genes_info
        filtered_info = filtered_info[
            (filtered_info["pvals_adj"] < 0.05)
            & (filtered_info["in_out_group_ratio"] > min_in_out_group_ratio)
            & (filtered_info["in_group_fraction"] > min_in_group_fraction)
            & (filtered_info["fold_change"] > min_fold_change)
        ]
        print("2.After multiple screenings there are still：", filtered_info.shape)

        filtered_info = filtered_info.sort_values(by="in_group_fraction", ascending=False)
        filtered_info["target_dmain"] = target
        filtered_info["neighbors"] = str(nbr_domians)
        SVG_num=len(filtered_info["genes"].tolist())
        print(f"target{target} find {SVG_num}个SVG，namely: {filtered_info['genes'].tolist()} " )
        all_SVG = pd.concat([all_SVG, filtered_info], ignore_index=True)  # 另外，需要把每次for得到的SVG拼接起来，最终统一返回


    SVG_filter=all_SVG.loc[:, ~all_SVG.columns.duplicated()]
    SVG_filter=SVG_filter.drop_duplicates()
    print("{} after filter：SVG shape:{}".format(method,SVG_filter.shape))
    all_SVG.to_csv(f'{out_dir}{method}_{dataset}_SVG_identified_{all_SVG.shape[0]}.csv')

    ##  After obtaining SVG, then calculate the moran and Geary scores
    SVG=all_SVG['genes'].values
    # Matching the gene expression corresponding to these SVGs from raw gene expression
    adata_DF=adata.to_df()
    SVG_count=adata_DF.loc[:,SVG]
    SVG_count=SVG_count.loc[:, ~SVG_count.columns.duplicated()]

    Moran_score=Moran_I(SVG_count,adata.obs["x_array"].values,adata.obs["y_array"].values)
    Geary_score=Geary_C(SVG_count,adata.obs["x_array"],adata.obs["y_array"])
    # print(f'Moran_score.mean: {Moran_score.mean():.5f} '+f'\nGeary_score.mean: {Geary_score.mean():.5f}')
    ## Save the Moran's I and Geary's C
    M_G_score=Moran_score.to_frame(name="Moran")
    M_G_score['Geary']=Geary_score.values
    M_G_score['Geary_revise'] = 1-Geary_score.values
    M_G_score.to_csv(f'{ M_G_output}{method}_M_G_score.csv')
>>>>>>> fd993021549337095c3af2b26e3476a95b99a611
