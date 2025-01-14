import numpy as np
import scanpy as sc
from calculate_moran_I import Moran_I
from calculate_moran_I import Geary_C
from scipy.sparse import issparse
import SpaGCN as spg
import pandas as pd
import matplotlib.colors as clr
import matplotlib.pyplot as plt

adata = sc.read("./SpaGCN_emb_cluster.h5ad")

x_array = adata.obs["x_array"].tolist()  # 存储所有X横坐标位置
y_array = adata.obs["y_array"].tolist()
x_pixel = adata.obs["x_pixel"].tolist()
y_pixel = adata.obs["y_pixel"].tolist()

raw = sc.read("../../dataset/151673/151673_filtered_feature_bc_matrix.h5")
raw.var_names_make_unique()
raw.obs["pred"] = adata.obs["pred"].astype("category")
raw.obs["x_array"] = raw.obs["x2"]
raw.obs["y_array"] = raw.obs["x3"]
raw.obs["x_pixel"] = raw.obs["x4"]
raw.obs["y_pixel"] = raw.obs["x5"]
# Convert sparse matrix to non-sparse
raw.X = raw.X.A if issparse(raw.X) else raw.X
raw.raw = raw
sc.pp.log1p(raw)

# Set filtering criterials
min_in_group_fraction = 0.8
min_in_out_group_ratio = 1
min_fold_change = 1.5
# 在靶标域中找包含大约10个邻居的半径
adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False) # 计算数据间的欧式距离
start, end = np.quantile(adj_2d[adj_2d != 0], q=0.001), np.quantile(
    adj_2d[adj_2d != 0], q=0.1
)
## 7.1 将所有识别的SVG保存为DF
all_SVG = pd.DataFrame()
for target in range(2): #7
    print("正在计算targt:",target)
    #为每个靶标域找合适的半径，为找邻居做准备
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
    # 在半径域内，找邻居空间域列表，用于识别SVG

    nbr_domians = spg.find_neighbor_clusters(
        target_cluster=target,
        cell_id=raw.obs.index.tolist(),
        x=raw.obs["x_array"].tolist(),
        y=raw.obs["y_array"].tolist(),
        pred=raw.obs["pred"].tolist(),  # 这里有用到之前的聚类信息
        radius=r,
        ratio=1 / 2,
    )
    print("target {}的邻居域{}".format(target,nbr_domians))  # 返回靶标域的邻居域，空间域1有邻居域3，6。因为SVG是基于邻居域得到的。

    nbr_domians = nbr_domians[0:3]  # 只取前3个邻居域的
    ##返回（33539，8），即每个基因的差异分析，普通方法直接就发现SVG。但spaGCN中要设定每个靶标域，得到每个域的的svg，然后再相加得到所有的SVG.
    ##输入原始基因表达，靶标域，还有邻居
    de_genes_info = spg.rank_genes_groups(
        input_adata=raw,   #（3639，33538）
        target_cluster=target,  # 2
        nbr_list=nbr_domians,   #（1，5，6）
        label_col="pred", #将spaGCN预测的结果传入
        adj_nbr=True,
        log=True,
    )
    #de_genes （33538，8），完成spot-gene到gene-指标的计算，即行列转化
    # de_genes_info (33538,8)，计算完每个基因的8项指标
    # 过滤基因，找出p值小于0.05的，之后只有(404, 8)
    de_genes_info = de_genes_info[
        (de_genes_info["pvals_adj"] < 0.05)
    ]  # （4468，8）,选出其中符合的4468个基因

    filtered_info = de_genes_info
    filtered_info = filtered_info[
        (filtered_info["pvals_adj"] < 0.05)
        & (filtered_info["in_out_group_ratio"] > min_in_out_group_ratio)
        & (filtered_info["in_group_fraction"] > min_in_group_fraction)
        & (filtered_info["fold_change"] > min_fold_change)
    ]
    # 过滤完之后就只剩下一个,得到该靶标域的SVG，要
    filtered_info = filtered_info.sort_values(
        by="in_group_fraction", ascending=False
    )  # 对找出的差异基因排序
    filtered_info["target_dmain"] = target  # 得到的结果中新列一列，记录target
    filtered_info["neighbors"] = str(nbr_domians) # 得到的结果中新列一列，记录demain
    print("domain{} for SVG ".format( str(target),filtered_info["genes"].tolist()))

    filtered_info  # 得到每个靶标域的SVG
    all_SVG = pd.concat([all_SVG, filtered_info], ignore_index=True) #将每个靶标域的SVG进行拼接

print("得到汇总的SVG",all_SVG.shape) # (76, 10)
# print(all_SVG.head())
SVG_filter=all_SVG.loc[:, ~all_SVG.columns.duplicated()] #对重复的列过滤
SVG_filter=SVG_filter.drop_duplicates()
print("得到过滤后汇总的SVG",SVG_filter.shape) # (76, 10)，76个SVG对应的10个指标
# SVG_filter.to_csv("151673_spaGCN_SVG_65.csv")

## 7.2 得到SVG的DF后，用于计算moran 和 Geary得分
SVG=SVG_filter['genes'].values #获得靶标基因的名称

# 从原始基因表达中匹配这些SVG对应的基因表达
# raw[:,SVG].X 虽然也可以返回(3639, 65)，没有index,var
raw_DF=raw.to_df() # 把X变成DF,为了获得它的index,var
SVG_count=raw_DF.loc[:,SVG]
print("raw_DF.loc[:,SVG]",SVG_count.shape)

##输入SVG的基因表达矩阵(要有它的spot，var名)，还有X,Y坐标的值，返回series，即每个SVG对应的Morans得分
Moran_score=Moran_I(SVG_count,raw.obs["x_array"].values,raw.obs["y_array"].values)
Geary_score=Geary_C(SVG_count,raw.obs["x_array"],raw.obs["y_array"])
print(Moran_score,Geary_score)
## 7.3 绘制每个SVG的空间表达
# Plot refinedspatial domains，自己生成颜色
color_self = clr.LinearSegmentedColormap.from_list(
    "pink_green", ["#3AB370", "#EAE7CC", "#FD1593"], N=256
)
###为每个svg的表达，画图
for g in filtered_info["genes"].tolist():  # 变量靶标为0的SVG,然后再画出来
    raw.obs["exp"] = raw.X[:, raw.var.index == g]
    # 选取Raw中x_pixel,y_pixel，所以有3639个坐标
    ax = sc.pl.scatter(
        raw,
        alpha=1,
        x="y_pixel",
        y="x_pixel",
        color="exp", #颜色由exp这一列区分
        title=g,
        color_map=color_self,
        show=True,
        size=100000 / raw.shape[0],
    )

    # ax.set_aspect('equal', 'box')
  #  ax.axes.invert_yaxis()
  #  plt.savefig("./sample_results/" + g + ".png", dpi=600)
    plt.show()