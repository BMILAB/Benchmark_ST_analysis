import squidpy as sq
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

platform_map = {"Mouse_hippocampus": "slideseq", "Mouse_olfactory_slide_seqv2": "slideseq", "MOB_without_label": "stereoseq",
                "PDAC": "ST", "DLPFC": '10 X', "Breast_cancer": '10 X', "Mouse_brain": '10 X',
                "SeqFish": "Seqfish", "STARmap": "STARmap"
                }
n_clusters_map = {"Mouse_hippocampus": 10, "Mouse_olfactory_slide_seqv2": 9, "MOB_without_label": 7,
                  "PDAC": 4, "DLPFC": '5-7', "Breast_cancer": 20, "Mouse_brain": 15,
                  "SeqFish": 22, "STARmap": 16}

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



def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def get_adata(dataset, data_path='../../Dataset/'):
    if dataset.startswith('15'):  # DLPFC dataset
        print("load DLPFC dataset:")
        file_fold = f"{data_path}DLPFC/{dataset}/"
        # 读入count
        raw = sc.read_visium(path=f'../../Dataset/DLPFC/{dataset}/', count_file='filtered_feature_bc_matrix.h5')
        # raw = sc.read_visium(path=file_fold, count_file=dataset + '_filtered_feature_bc_matrix.h5')
        raw.var_names_make_unique()
        # 读入真实标签
        Ann_df = pd.read_csv(f'../../Dataset/DLPFC/{dataset}/metadata.tsv', sep='\t')
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        raw.obs['ground_truth'] = Ann_df.loc[raw.obs_names, 'Ground Truth']

        n_clusters = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7

    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset)  # please replace 'file_fold' with the download path
        raw = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        raw.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
        raw.obs['ground_truth'] = df_meta.loc[raw.obs_names, 'fine_annot_type'].values  # fine_annot_type代替annot_type
        print("Breast_cancer的类别数", len(raw.obs['ground_truth'].unique()))

        n_clusters = 20

    if dataset == "Mouse_brain":
        raw = sq.datasets.visium_hne_adata()
        raw.var_names_make_unique()
        raw.obs['ground_truth'] = raw.obs["cluster"]
        print(
            f"Mouse_brain数据大小：{raw.shape}，类别数：{len(raw.obs['ground_truth'].unique())}")  # ,raw.obs['ground_truth'].unique())

        n_clusters = 15

    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")  # (428, 19736)
        raw.obs['ground_truth'] = raw.obs['Ground Truth']
        print("PDAC的类别数", raw.obs['ground_truth'].unique())
        n_clusters = 4

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + '/Adult_stereo.h5ad')
        print(raw.shape)  # (8243, 22144)
        raw.obs["ground_truth"] = raw.obs['Annotation']  # 后面需要obs['Annotation'] 格式
        print("Stereo标签类别数：", len(raw.obs['Annotation'].unique()))
        n_clusters = 16

    if dataset == 'SeqFish':
        raw = sq.datasets.seqfish()
        # print('Seqfish.shape',adata.shape)  # (19416, 351)
        raw.obs['ground_truth'] = raw.obs['celltype_mapped_refined']
        print("SeqFish标签类别数：", len(raw.obs['ground_truth'].unique()))
        n_clusters = 22

        image_coor = raw.obsm["spatial"]  # 直接获取像素点位置
        x_array = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]  # 由uns中存储的放缩比，推测像素点 imagecol,imagerow的位置
        raw.obs["y_array"] = image_coor[:, 1]

    if dataset == "STARmap":
        file_fold = data_path + str(dataset)

        raw = sc.read(file_fold + "/STARmap_1207_1020.h5ad")
        print(
            f"STARmap数据大小：{raw.shape}，类别数：{len(raw.obs['ground_truth'].unique())}")  # ,raw.obs['ground_truth'].unique())
        n_clusters = 16

    return raw, n_clusters