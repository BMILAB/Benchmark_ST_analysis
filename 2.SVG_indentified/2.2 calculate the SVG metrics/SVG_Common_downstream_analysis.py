<<<<<<< HEAD
## 1.从每个数据下读入3个传统识别的SVG，取交集
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import scanpy as sc
import squidpy as sq
import random

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def read_data(dataset, data_path='../../adata/Raw_adata/'):
    if dataset.startswith("SlideV2"):
        file_fold = f'{data_path}slideV2/{dataset}.h5ad'
        adata = sc.read(file_fold)
        n_clusters = len(adata.obs['ground_truth'].value_counts())
        print(f"slideV2的{dataset}类别数:{n_clusters},shape:{adata.shape}")
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset.startswith("STARmap"):
        file_fold = f'{data_path}STARmap/{dataset}.h5ad'
        adata = sc.read(file_fold)
        n_clusters = len(adata.obs['ground_truth'].value_counts())
        print(f"STARmap的{dataset}数据集类别数:{n_clusters},shape:{adata.shape}")
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset.startswith("ST_"):
        file_fold = f'{data_path}ST/{dataset}.h5ad'
        adata = sc.read(file_fold)
        n_clusters = len(adata.obs['ground_truth'].value_counts())
        print(f"ST的{dataset}数据集类别数:{n_clusters},shape:{adata.shape}")

    if dataset.startswith('15'):  # DLPFC dataset
        print("load DLPFC dataset:")
        file_path = f'{data_path}/DLPFC/{dataset}/'  # f'../../../Dataset/DLPFC/{dataset}/'

        adata = sc.read_visium(path=file_path, count_file=f'{dataset}_filtered_feature_bc_matrix.h5')  #
        adata.var_names_make_unique()
        # 读入真实标签
        Ann_df = pd.read_csv(f'{file_path}/metadata.tsv', sep='\t')
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']


    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        # 读入原始数据
        # adata = sc.read(file_fold+"/STARmap_20180505_BY3_1k.h5ad") #(1207, 1020)
        # adata.var_names_make_unique()
        # df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt', sep='\t', index_col=0)
        # adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,'Annotation'].values
        adata = sc.read(file_fold + "/STARmap.h5ad")
        image_coor =  adata.obsm["spatial"]  # 直接获取像素点位置
        adata.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        adata.obs["y_array"] = image_coor[:, 1]  #


    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold + '/Adult_stereo.h5ad')
        print(adata.shape)  # (8243, 22144)
        adata.obs["ground_truth"] = adata.obs['Annotation']  # 后面需要obs['Annotation'] 格式
        print("标签类别数：", len(adata.obs['Annotation'].unique()))
        image_coor = adata.obsm["spatial"]  # 直接获取像素点位置
        adata.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        adata.obs["y_array"] = image_coor[:, 1]

    if dataset == 'SeqFish':
            adata = sq.datasets.seqfish()
            # print('Seqfish.shape',adata.shape)  # (19416, 351)
            adata.obs['ground_truth'] = adata.obs['celltype_mapped_refined']
            adata.obs['ground_truth'].value_counts().plot(kind='bar')
            image_coor = adata.obsm["spatial"]  # 直接获取像素点位置
            adata.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
            adata.obs["y_array"] = image_coor[:, 1]

    if dataset == "Mouse_brain":
            adata = sq.datasets.visium_hne_adata()
            adata.var_names_make_unique()
            adata.obs['ground_truth'] = adata.obs["cluster"]
            print("Mouse_brain的类别数", adata.obs['ground_truth'].value_counts())
            image_coor = adata.obsm["spatial"]  # 直接获取像素点位置
            adata.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
            adata.obs["y_array"] = image_coor[:, 1]

    if dataset == "Breast_cancer":
            file_fold = data_path + str(dataset)  # please replace 'file_fold' with the download path
            adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5',
                                   load_images=True)
            adata.var_names_make_unique()
            df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
            adata.obs['ground_truth'] = df_meta.loc[
                adata.obs_names, 'fine_annot_type'].values  # fine_annot_type代替annot_type
            print("Breast_cancer的类别数", adata.obs['ground_truth'].value_counts())
            adata.obs['ground_truth'].value_counts().plot(kind='bar')
            # plt.tight_layout()  # 调整画布在正中间
            # plt.show()
    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")  # (428, 19736)
        adata.obs['ground_truth'] = adata.obs['Ground Truth']
        adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    return adata



import sys
sys.path.append('../../1.Benchmark_SRT-main/')
import utils_for_all as usa


if __name__ == '__main__':

    Dataset_test=['151673']#['151673','STARmap','Stereo','SeqFish']
for dataset in  Dataset_test:
    raw_adata, n_clusters = usa.get_adata(dataset, data_path='../../Dataset/')
    raw_adata.var_names_make_unique()
    gene_all=raw_adata.var_names.values

    com_SVG=pd.read_csv('Ref_Consense_SVG.csv', index_col=0)
    com_SVG=com_SVG[dataset][com_SVG[dataset].notna()]
    SVG_num=com_SVG.shape[0]
    print(f"{dataset} has {SVG_num} consesus SVG")
    print(f'===============================Now is calculating dataset:{dataset}========================')

    AUPR_df = pd.DataFrame()
    AUPR_dict_list = []
    dict_list = []
    df = pd.DataFrame()
    for num in [SVG_num]:
        SVG_ref=com_SVG[:num].values

        if dataset.startswith('15'):
            data_path = f"../SVG_indentified_Output/DLPFC/{dataset}"
            moran_path=f'../result_M_G_score/DLPFC/{dataset}/'
        else:
            data_path=f"../SVG_indentified_Output/{dataset}/"
            moran_path = f'../result_M_G_score/{dataset}/'

        FP_adjust_type = 'FP_SVG_Moran_mean' # 0.3
        for root, dirs, files in os.walk(data_path):
            files = [file for file in files if file.endswith('.csv')]
            for i,file in enumerate(files):
                method=file.split('_')[0]
                if method=='Spatial':
                    method='Spatial_MGCN'
                print(f'============now is running the {i} method：{method}================')
                file_path = os.path.join(root, file)
                SVG_DF = pd.read_csv(file_path, index_col=0)
                # get the  gene name and p-value of the SVG indentified
                SVG_DF[method] =  SVG_DF['genes'].values
                SVG_DF['pvals_adj'] =  SVG_DF['pvals_adj']

                # and get the  Moran's score of the SVG indentified
                moran_df = pd.read_csv(f'{moran_path}{method}_M_G_score.csv', index_col=0)
                SVG_name=list(SVG_DF['genes'].values)
                SVG_DF['Moran']=moran_df.loc[SVG_name,'Moran'].values
                if SVG_DF.shape[0] == 0:
                    continue

                ##Whether the identified SVG can be found in the reference SVG, 1 if it can be found, 0 otherwise
                SVG_DF['Ground'] = SVG_DF[method].apply(lambda x: 1 if x in SVG_ref else 0).values
                print(f"{method} detects {SVG_DF.shape[0]} SVG,the number of TP,FP were : { SVG_DF['Ground'].value_counts()}")

                dataset_specific_moran_average = SVG_DF['Moran'].mean()
                print(f"{dataset} dataset has mean Moran's score:{dataset_specific_moran_average}")


                # Re-recruitment of true SVG by recalculating FP's Moran score
                FP_init = SVG_DF[SVG_DF['Ground'] == 0][method].values #get FP gene's name

                FP_morans=moran_df.loc[FP_init, 'Moran'].apply(lambda x: f"{x:.3f}") #get the match moran's score
                FP_morans=FP_morans.values

                if FP_adjust_type=='FP_SVG_Moran_mean':
                    FP_gene_biaozhun=dataset_specific_moran_average
                else:
                    FP_gene_biaozhun=0.3

                FP_Moran_lists = list(zip(FP_init, FP_morans))
                # recuite the real SVG for FP
                filtered_FP_Moran_lists = [t for t in FP_Moran_lists if float(t[1]) > FP_gene_biaozhun]
                filtered_FP_gene = [t[0] for t in filtered_FP_Moran_lists]

                tp_plus=0
                # 遍历合并后的列表
                for FP_gene, gene_moran in FP_Moran_lists:
                    if float(gene_moran) >= FP_gene_biaozhun:  # dataset_specific_moran_average
                        tp_plus = tp_plus + 1

                # reclassfy the TP,FP
                TP = SVG_DF[SVG_DF['Ground'] == 1][method].values
                TP_num=len(TP)+tp_plus

                print(f"after the fine-tune，{method} can find {TP_num} SVG")
                FP = SVG_DF[SVG_DF['Ground'] == 0][method].values
                FP_num=len(FP)-tp_plus
                print(f"after the fine-tune，{method} finally predicit {FP_num} FP genes:")

                # Calculate rankings based on pvalue and consequently AUPR
                results = -np.log10( SVG_DF['pvals_adj'])
                SVG_DF['Pred'] = (-results).rank(na_option='keep')
                # Calculte ROC
                fpr, tpr, thresholds = roc_curve(SVG_DF['Ground'], SVG_DF['Pred'])  # 注意先FPR,再TPR
                roc_auc = auc(fpr, tpr)
                roc_auc = roc_auc if not np.isnan(roc_auc) else 0

                precision, recall, _ = precision_recall_curve(SVG_DF['Ground'], SVG_DF['Pred'])
                aupr = auc(recall, precision)


                ### calculate the ACC,Recall,etc metrics
                cutoff = [0.05]
                for thresholds in cutoff:
                    df=SVG_DF
                    df['tag'] = df[method].apply(lambda x: 1 if x in SVG_ref else 0).values
                    pred = df[df['pvals_adj'] < thresholds]
                    TP=TP_num
                    FP=FP_num
                    indentified_SVG=pred[method].values
                    #Remove the ones already predicted to be SVG and then randomly select the negative pole
                    pred_other = set(gene_all) - set(pred[method].values)  #
                    pred_other=list(pred_other)[:3000]

                    FN= len(set(pred_other) & set(SVG_ref))
                    TN=len(pred_other)-FN

                    TPR=TP/(TP+FN+0.01)
                    recall=TPR
                    FPR=FP/(FP+TN+0.01)
                    FDR=FP/(FP+TP+0.01)
                    # TNR=TN/(TN+FP) ##
                    precision=TP/(TP+FP+0.01)
                    F1=2*precision*recall/(precision+recall+0.01)

                    if thresholds==0.05:
                        AUPR_dict_list.append({
                            'dataset': dataset,
                            'method': method, 'REF_SVG': num, 'SVG_identified': df.shape[0], 'threshold': thresholds,
                            "AUPR":aupr,"ROC":roc_auc,'precision': precision, 'recall': recall, 'F1': F1,
                            'TPR': TPR, 'FPR': FPR, 'FDR': FDR,
                            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
                            'TP_plus':tp_plus,'Moran_mean':dataset_specific_moran_average
                        }) #"FP_init":FP_init,"FP_morans":FP_morans
                        print("AUPR_dict_list:",AUPR_dict_list)

         #save the result
        result_df = pd.DataFrame(AUPR_dict_list)
        result_save_file=f'./SVG_calculate_result/'
        mk_dir(result_save_file)

        result_df.to_csv(f'{result_save_file}/{dataset}_AUPR_ROC_result.csv')
















=======
## 1.从每个数据下读入3个传统识别的SVG，取交集
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import scanpy as sc
import squidpy as sq
import random

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def read_data(dataset, data_path='../../adata/Raw_adata/'):
    if dataset.startswith("SlideV2"):
        file_fold = f'{data_path}slideV2/{dataset}.h5ad'
        adata = sc.read(file_fold)
        n_clusters = len(adata.obs['ground_truth'].value_counts())
        print(f"slideV2的{dataset}类别数:{n_clusters},shape:{adata.shape}")
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset.startswith("STARmap"):
        file_fold = f'{data_path}STARmap/{dataset}.h5ad'
        adata = sc.read(file_fold)
        n_clusters = len(adata.obs['ground_truth'].value_counts())
        print(f"STARmap的{dataset}数据集类别数:{n_clusters},shape:{adata.shape}")
        # adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    if dataset.startswith("ST_"):
        file_fold = f'{data_path}ST/{dataset}.h5ad'
        adata = sc.read(file_fold)
        n_clusters = len(adata.obs['ground_truth'].value_counts())
        print(f"ST的{dataset}数据集类别数:{n_clusters},shape:{adata.shape}")

    if dataset.startswith('15'):  # DLPFC dataset
        print("load DLPFC dataset:")
        file_path = f'{data_path}/DLPFC/{dataset}/'  # f'../../../Dataset/DLPFC/{dataset}/'

        adata = sc.read_visium(path=file_path, count_file=f'{dataset}_filtered_feature_bc_matrix.h5')  #
        adata.var_names_make_unique()
        # 读入真实标签
        Ann_df = pd.read_csv(f'{file_path}/metadata.tsv', sep='\t')
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']


    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        # 读入原始数据
        # adata = sc.read(file_fold+"/STARmap_20180505_BY3_1k.h5ad") #(1207, 1020)
        # adata.var_names_make_unique()
        # df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt', sep='\t', index_col=0)
        # adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,'Annotation'].values
        adata = sc.read(file_fold + "/STARmap.h5ad")
        image_coor =  adata.obsm["spatial"]  # 直接获取像素点位置
        adata.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        adata.obs["y_array"] = image_coor[:, 1]  #


    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold + '/Adult_stereo.h5ad')
        print(adata.shape)  # (8243, 22144)
        adata.obs["ground_truth"] = adata.obs['Annotation']  # 后面需要obs['Annotation'] 格式
        print("标签类别数：", len(adata.obs['Annotation'].unique()))
        image_coor = adata.obsm["spatial"]  # 直接获取像素点位置
        adata.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        adata.obs["y_array"] = image_coor[:, 1]

    if dataset == 'SeqFish':
            adata = sq.datasets.seqfish()
            # print('Seqfish.shape',adata.shape)  # (19416, 351)
            adata.obs['ground_truth'] = adata.obs['celltype_mapped_refined']
            adata.obs['ground_truth'].value_counts().plot(kind='bar')
            image_coor = adata.obsm["spatial"]  # 直接获取像素点位置
            adata.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
            adata.obs["y_array"] = image_coor[:, 1]

    if dataset == "Mouse_brain":
            adata = sq.datasets.visium_hne_adata()
            adata.var_names_make_unique()
            adata.obs['ground_truth'] = adata.obs["cluster"]
            print("Mouse_brain的类别数", adata.obs['ground_truth'].value_counts())
            image_coor = adata.obsm["spatial"]  # 直接获取像素点位置
            adata.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
            adata.obs["y_array"] = image_coor[:, 1]

    if dataset == "Breast_cancer":
            file_fold = data_path + str(dataset)  # please replace 'file_fold' with the download path
            adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5',
                                   load_images=True)
            adata.var_names_make_unique()
            df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
            adata.obs['ground_truth'] = df_meta.loc[
                adata.obs_names, 'fine_annot_type'].values  # fine_annot_type代替annot_type
            print("Breast_cancer的类别数", adata.obs['ground_truth'].value_counts())
            adata.obs['ground_truth'].value_counts().plot(kind='bar')
            # plt.tight_layout()  # 调整画布在正中间
            # plt.show()
    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")  # (428, 19736)
        adata.obs['ground_truth'] = adata.obs['Ground Truth']
        adata.obs['ground_truth'].value_counts().plot(kind='bar')
        # plt.tight_layout()  # 调整画布在正中间
        # plt.show()

    return adata



import sys
sys.path.append('../../1.Benchmark_SRT-main/')
import utils_for_all as usa


if __name__ == '__main__':

    Dataset_test=['151673']#['151673','STARmap','Stereo','SeqFish']
for dataset in  Dataset_test:
    raw_adata, n_clusters = usa.get_adata(dataset, data_path='../../Dataset/')
    raw_adata.var_names_make_unique()
    gene_all=raw_adata.var_names.values

    com_SVG=pd.read_csv('Ref_Consense_SVG.csv', index_col=0)
    com_SVG=com_SVG[dataset][com_SVG[dataset].notna()]
    SVG_num=com_SVG.shape[0]
    print(f"{dataset} has {SVG_num} consesus SVG")
    print(f'===============================Now is calculating dataset:{dataset}========================')

    AUPR_df = pd.DataFrame()
    AUPR_dict_list = []
    dict_list = []
    df = pd.DataFrame()
    for num in [SVG_num]:
        SVG_ref=com_SVG[:num].values

        if dataset.startswith('15'):
            data_path = f"../SVG_indentified_Output/DLPFC/{dataset}"
            moran_path=f'../result_M_G_score/DLPFC/{dataset}/'
        else:
            data_path=f"../SVG_indentified_Output/{dataset}/"
            moran_path = f'../result_M_G_score/{dataset}/'

        FP_adjust_type = 'FP_SVG_Moran_mean' # 0.3
        for root, dirs, files in os.walk(data_path):
            files = [file for file in files if file.endswith('.csv')]
            for i,file in enumerate(files):
                method=file.split('_')[0]
                if method=='Spatial':
                    method='Spatial_MGCN'
                print(f'============now is running the {i} method：{method}================')
                file_path = os.path.join(root, file)
                SVG_DF = pd.read_csv(file_path, index_col=0)
                # get the  gene name and p-value of the SVG indentified
                SVG_DF[method] =  SVG_DF['genes'].values
                SVG_DF['pvals_adj'] =  SVG_DF['pvals_adj']

                # and get the  Moran's score of the SVG indentified
                moran_df = pd.read_csv(f'{moran_path}{method}_M_G_score.csv', index_col=0)
                SVG_name=list(SVG_DF['genes'].values)
                SVG_DF['Moran']=moran_df.loc[SVG_name,'Moran'].values
                if SVG_DF.shape[0] == 0:
                    continue

                ##Whether the identified SVG can be found in the reference SVG, 1 if it can be found, 0 otherwise
                SVG_DF['Ground'] = SVG_DF[method].apply(lambda x: 1 if x in SVG_ref else 0).values
                print(f"{method} detects {SVG_DF.shape[0]} SVG,the number of TP,FP were : { SVG_DF['Ground'].value_counts()}")

                dataset_specific_moran_average = SVG_DF['Moran'].mean()
                print(f"{dataset} dataset has mean Moran's score:{dataset_specific_moran_average}")


                # Re-recruitment of true SVG by recalculating FP's Moran score
                FP_init = SVG_DF[SVG_DF['Ground'] == 0][method].values #get FP gene's name

                FP_morans=moran_df.loc[FP_init, 'Moran'].apply(lambda x: f"{x:.3f}") #get the match moran's score
                FP_morans=FP_morans.values

                if FP_adjust_type=='FP_SVG_Moran_mean':
                    FP_gene_biaozhun=dataset_specific_moran_average
                else:
                    FP_gene_biaozhun=0.3

                FP_Moran_lists = list(zip(FP_init, FP_morans))
                # recuite the real SVG for FP
                filtered_FP_Moran_lists = [t for t in FP_Moran_lists if float(t[1]) > FP_gene_biaozhun]
                filtered_FP_gene = [t[0] for t in filtered_FP_Moran_lists]

                tp_plus=0
                # 遍历合并后的列表
                for FP_gene, gene_moran in FP_Moran_lists:
                    if float(gene_moran) >= FP_gene_biaozhun:  # dataset_specific_moran_average
                        tp_plus = tp_plus + 1

                # reclassfy the TP,FP
                TP = SVG_DF[SVG_DF['Ground'] == 1][method].values
                TP_num=len(TP)+tp_plus

                print(f"after the fine-tune，{method} can find {TP_num} SVG")
                FP = SVG_DF[SVG_DF['Ground'] == 0][method].values
                FP_num=len(FP)-tp_plus
                print(f"after the fine-tune，{method} finally predicit {FP_num} FP genes:")

                # Calculate rankings based on pvalue and consequently AUPR
                results = -np.log10( SVG_DF['pvals_adj'])
                SVG_DF['Pred'] = (-results).rank(na_option='keep')
                # Calculte ROC
                fpr, tpr, thresholds = roc_curve(SVG_DF['Ground'], SVG_DF['Pred'])  # 注意先FPR,再TPR
                roc_auc = auc(fpr, tpr)
                roc_auc = roc_auc if not np.isnan(roc_auc) else 0

                precision, recall, _ = precision_recall_curve(SVG_DF['Ground'], SVG_DF['Pred'])
                aupr = auc(recall, precision)


                ### calculate the ACC,Recall,etc metrics
                cutoff = [0.05]
                for thresholds in cutoff:
                    df=SVG_DF
                    df['tag'] = df[method].apply(lambda x: 1 if x in SVG_ref else 0).values
                    pred = df[df['pvals_adj'] < thresholds]
                    TP=TP_num
                    FP=FP_num
                    indentified_SVG=pred[method].values
                    #Remove the ones already predicted to be SVG and then randomly select the negative pole
                    pred_other = set(gene_all) - set(pred[method].values)  #
                    pred_other=list(pred_other)[:3000]

                    FN= len(set(pred_other) & set(SVG_ref))
                    TN=len(pred_other)-FN

                    TPR=TP/(TP+FN+0.01)
                    recall=TPR
                    FPR=FP/(FP+TN+0.01)
                    FDR=FP/(FP+TP+0.01)
                    # TNR=TN/(TN+FP) ##
                    precision=TP/(TP+FP+0.01)
                    F1=2*precision*recall/(precision+recall+0.01)

                    if thresholds==0.05:
                        AUPR_dict_list.append({
                            'dataset': dataset,
                            'method': method, 'REF_SVG': num, 'SVG_identified': df.shape[0], 'threshold': thresholds,
                            "AUPR":aupr,"ROC":roc_auc,'precision': precision, 'recall': recall, 'F1': F1,
                            'TPR': TPR, 'FPR': FPR, 'FDR': FDR,
                            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
                            'TP_plus':tp_plus,'Moran_mean':dataset_specific_moran_average
                        }) #"FP_init":FP_init,"FP_morans":FP_morans
                        print("AUPR_dict_list:",AUPR_dict_list)

         #save the result
        result_df = pd.DataFrame(AUPR_dict_list)
        result_save_file=f'./SVG_calculate_result/'
        mk_dir(result_save_file)

        result_df.to_csv(f'{result_save_file}/{dataset}_AUPR_ROC_result.csv')
















>>>>>>> fd993021549337095c3af2b26e3476a95b99a611
