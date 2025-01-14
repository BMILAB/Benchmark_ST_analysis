import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import os
from anndata import AnnData,read_csv,read_text,read_mtx
from scipy.sparse import issparse
import random
import torch
from . SpaGCN import SpaGCN
from . calculate_adj import calculate_adj_matrix
def prefilter_cells(adata,min_counts=None,max_counts=None,min_genes=200,max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[0],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_genes=min_genes)[0]) if min_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_genes=max_genes)[0]) if max_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw=sc.pp.log1p(adata,copy=True) #check the rowname 
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:",adata.raw.var_names.is_unique)
   

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)


def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)


def calculate_p(adj, l):
    adj_exp=np.exp(-1*(adj**2)/(2*(l**2)))
    return np.mean(np.sum(adj_exp,1))-1

def test_l(adj, list_l):
    for l in list_l:
        print("l is ",str(l),"Percentage of total expression contributed by neighborhoods:",calculate_p(adj, l))

def find_l(p, adj, start=0.5, end=20,sep=0.01, tol=0.01): #end=2，报错：l not found, try bigger range or smaller sep!
    for l in np.arange(start, end, sep):
        q=calculate_p(adj, l)
        print("L=", str(l), "P=", str(round(q,5)))
        if np.abs(p-q)<=tol:
            return l
    print("l not found, try bigger range or smaller sep!")

def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    run=0
    p_low=calculate_p(adj, start)
    p_high=calculate_p(adj, end)
    if p_low>p+tol:
        print("l not found, try smaller start point.")
        return None
    elif p_high<p-tol:
        print("l not found, try bigger end point.")
        return None
    elif  np.abs(p_low-p) <=tol:
        print("recommended l = ", str(start))
        return start
    elif  np.abs(p_high-p) <=tol:
        print("recommended l = ", str(end))
        return end
    while (p_low+tol)<p<(p_high-tol):
        run+=1
       # print("Run "+str(run)+": l ["+str(start)+", "+str(end)+"], p ["+str(p_low)+", "+str(p_high)+"]")
        if run >max_run:
            print("Exact l not found, closest values are:\n"+"l="+str(start)+": "+"p="+str(p_low)+"\nl="+str(end)+": "+"p="+str(p_high))
            return None
        mid=(start+end)/2
        p_mid=calculate_p(adj, mid)
        if np.abs(p_mid-p)<=tol:
            print("recommended l = ", str(mid))
            return mid
        if p_mid<=p:
            start=mid
            p_low=p_mid
        else:
            end=mid
            p_high=p_mid
#，其目的是计算在给定半径内，特定类别（由 target_cluster 指定）的细胞周围邻居细胞的平均数量
def count_nbr(target_cluster,cell_id, x, y, pred, radius):
    adj_2d=calculate_adj_matrix(x=x,y=y, histology=False)
    cluster_num = dict()
    df = {'cell_id': cell_id, 'x': x, "y":y, "pred":pred}
    df = pd.DataFrame(data=df)
    df.index=df['cell_id']
    target_df=df[df["pred"]==target_cluster]
    row_index=0
    num_nbr=[]
    for index, row in target_df.iterrows():
        x=row["x"]
        y=row["y"]
        tmp_nbr=df[((df["x"]-x)**2+(df["y"]-y)**2)<=(radius**2)] #欧式距离小于r范围内的
        num_nbr.append(tmp_nbr.shape[0])
    return np.mean(num_nbr) #邻居的平均个数

#找合适b
def search_radius(target_cluster,cell_id, x, y, pred, start, end, num_min=8, num_max=15,  max_run=100):
    run=0
    num_low=count_nbr(target_cluster,cell_id, x, y, pred, start)
    num_high=count_nbr(target_cluster,cell_id, x, y, pred, end)
    print(f'target {target_cluster}最小的邻居个数：{num_low},最大的邻居个数：{num_high}')
    if num_min<=num_low<=num_max: #8<nub_low<15
        print("recommended radius = ", str(start))
        return start
    elif num_min<=num_high<=num_max:
        print("recommended radius = ", str(end))
        return end
    elif num_low>num_max:
        print(f"当前半径范围{start}内，最小的邻居数都超过15，范围None,所以Try smaller start.")
        # return None
        new_start=start-0.2*(start)
        print(f'手工减少到原来4/5，所以设置start={new_start}')
        return new_start

    elif num_high<num_min:
        print(f"当前半径范围{end}内，最大的邻居数都不到8，,所以Try bigger end.")
        #return None
        new_end = end + 0.2 * (end)
        print(f'手工增加到原来6/5，所以设置end={new_end}')
        return new_end

    while (num_low<num_min) and (num_high>num_min):#如果计算的neigh小于最小的，或者大于最大的，run+1
        run+=1
        print(f'如果计算的neigh小于最小的，或者大于最大的，run++,run变成{run}')
       # print("Run "+str(run)+": radius ["+str(start)+", "+str(end)+"], num_nbr ["+str(num_low)+", "+str(num_high)+"]")
        if run >max_run:
            print("如果尝试多次,Exact radius not found, closest values are:\n"+"radius="+str(start)+": "+"num_nbr="+str(num_low)+"\nradius="+str(end)+": "+"num_nbr="+str(num_high))
            potention_r = end
            print("没找到r,使用推荐半径",potention_r)
            return potention_r
            # return start  #当SCGDL、CCST时、Muse报错，就用最近似的SVG.
        mid=(start+end)/2
        num_mid=count_nbr(target_cluster,cell_id, x, y, pred, mid)
        if num_min<=num_mid<=num_max:
            print("recommended radius = ", round(mid,3), "邻居个数=",round(num_mid,3))
            return mid
        if num_mid<num_min:
            start=mid
            num_low=num_mid
        elif num_mid>num_max:
            end=mid
            num_high=num_mid
#尝试在满足特定条件（即目标域内的spot数量至少是邻居域的一半）的情况下，找到目标聚类的邻居域。
#如果没有找到任何邻居域，它会返回spot数量最多的聚类作为潜在的邻居域，并给出相应的提示信息。
def find_neighbor_clusters(target_cluster,cell_id, x, y, pred,radius, ratio=1/2): #ratio=1/2
    cluster_num = dict()
    for i in pred:
        cluster_num[i] = cluster_num.get(i, 0) + 1 #统计预测结果中类别的分布
    # cluster_num_backup=cluster_num
    # cluster_num_backup = [(k, v) for k, v in  cluster_num_backup.items()]  # 域内spot要大于该数据集的类别分布 数的一半            #字典变成元组：得到邻居空间域，与对应的spot点数
    # cluster_num_backup.sort(key=lambda x: -x[0])
    # print(f"该数据集预测类别数目：{len(cluster_num)}，分布如下：\n", cluster_num_backup)

    df = {'cell_id': cell_id, 'x': x, "y":y, "pred":pred} #df中存储每个spot的id名，x,y还有所属label
    df = pd.DataFrame(data=df) #（3639，4），每个cell_id的坐标，以及它预测的类别
    df.index=df['cell_id']
    target_df=df[df["pred"]==target_cluster] #（605，4）靶标域为2的cell_id的所有spot信息
    nbr_num={}
    row_index=0
    num_nbr=[]
    for index, row in target_df.iterrows(): #一行行遍历这个DF
        x=row["x"]
        y=row["y"]
        #  rint("type(x),type(y),type(df['x']",type(x),type(y),type(df['x']),type(radius))  CCST在计算target=4时，radius=None
        # tmp_nbr=df[((df["x"]-x)**2+(df["y"]-y)**2)<=(radius**2)]
        tmp_nbr=df[(df["x"]<x+radius) & (df["x"]>x-radius) & (df["y"]<y+radius) & (df["y"]>y-radius)] #遍历这个target的域内spot,计算每个spot的邻居
        num_nbr.append(tmp_nbr.shape[0])#统计这个target域内，每个spot的邻居个数。num_nbr：邻居个数，nbr_num更详细，邻居的类别分布
        for p in tmp_nbr["pred"]:
            nbr_num[p]=nbr_num.get(p,0)+1 #字段，表示这个target内所有spot，的邻居分别来自哪些区域
    if target_cluster in nbr_num:
        del nbr_num[target_cluster]  # 删除靶标区域，就全领域的
    else:
        print("靶标域没在预测的邻居中")
    nbr_num_back=nbr_num.copy() #Backup,后面不在使用
    nbr_num=[(k, v)  for k, v in nbr_num.items() if v>(ratio*cluster_num[k])] #域内spot要大于该数据集的类别分布 数的一半            #字典变成元组：得到邻居空间域，与对应的spot点数
    nbr_num.sort(key=lambda x: -x[1]) #邻居的个数从大到小排列
    print(f"target{target_cluster}域的邻居类别个数：{len(nbr_num)},具体的邻居类别分布如下：\n",nbr_num)
    #print("radius=", radius, "average number of neighbors for each spot is", np.mean(num_nbr))
    # radius= 2.8726212047040462 average number of neighbors for each spot is 12.816528925619835
   
    # for t in nbr_num:
    #     print("Dmain ", t[0], ": ",t[1])
    ret=[t[0] for t in nbr_num] # ret为领域列表
    if len(ret)==0:
        nbr_num_back=[(k, v)  for k, v in nbr_num_back.items()]
        nbr_num_back.sort(key=lambda x: -x[1]) #对nbr_num_back进行排序，以便找到spot数量最多的聚类。
        ret=[nbr_num_back[0][0]] # 将spot数量最多的聚类的类别索引作为潜在的邻居域添加到ret列表中。
        print("No neighbor domain found, only return one potential neighbor domain:",ret)
        print("Try bigger radius or smaller ratio.")
    return ret #返回空间域列表

#input_adata（3639，33538），进行svg识别
def rank_genes_groups(input_adata, target_cluster,nbr_list, label_col, adj_nbr=True, log=False):
    if adj_nbr:
        nbr_list=nbr_list+[target_cluster] #得到邻居域与靶标域，获得列表
        adata=input_adata[input_adata.obs[label_col].isin(nbr_list)] #(1516,33538) #（2475，33538），即邻居域与靶标域所有的spot基因信息
    else:
        adata=input_adata.copy()
    adata.var_names_make_unique() #（1674，33538），去重
    adata.obs["target"]=((adata.obs[label_col]==target_cluster)*1).astype('category') #新增target列，将预测label为2的标注出来，1/0
    #基因差异分析的函数, groupby是定义组别，这里将target与其他邻域进行对比，用wilcoxon进行统计，结果将存储在adata.uns字典中，可以通过adata.uns[key_added]来访问
    sc.tl.rank_genes_groups(adata, groupby="target",reference="rest", n_genes=adata.shape[1],method='wilcoxon') #基于wilcoxon对靶标域与邻域差异分析，得到SVG.
    pvals_adj=[i[0] for i in adata.uns['rank_genes_groups']["pvals_adj"]] #获得33538的基因的p值，经前面一步计算得到
    genes=[i[1] for i in adata.uns['rank_genes_groups']["names"]]
    if issparse(adata.X):
        obs_tidy=pd.DataFrame(adata.X.A)
    else:
        obs_tidy=pd.DataFrame(adata.X) #（2475，33538）
    obs_tidy.index=adata.obs["target"].tolist() #索引为target，0/1，即用0，1来分类
    obs_tidy.columns=adata.var.index.tolist() #列为名字
    obs_tidy=obs_tidy.loc[:,genes] #DF,其中index为0，1表领域还是靶标域
    # 1. compute mean value
    ####从这里开始不同，组内/组外平均表达值、折叠变化值这三个指标不同
    mean_obs = obs_tidy.groupby(level=0).mean() #（2，n)表示该基因在每个聚类中基因的平均表达水平。
    # 2. compute fraction of cells having value >0
    obs_bool = obs_tidy.astype(bool)
    #fraction_obs 计算每个聚类中表达该基因的细胞比例
    fraction_obs = obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()  #(2,n),第一行为0(领域)，第二行为1（靶标域）
###注意：mean-obs与fraction_obs都有0，1两行，1的为target域在所在基因表达上的，而0为邻居域在所有基因上的表达
    # compute fold change.
    if log: #The adata already logged
        fold_change=np.exp((mean_obs.loc[1] - mean_obs.loc[0]).values) #如果数据已经对数化（log=True），则直接计算平均表达值的指数差
    else:
        fold_change = (mean_obs.loc[1] / (mean_obs.loc[0]+ 1e-9)).values #如果没有，则计算目标聚类的平均表达与邻近聚类平均表达的比值，并处理除以零的情况。
        #将每个Gene,及指标又进行了重排，所以从列变成了行
    df = {'genes': genes, 'in_group_fraction': fraction_obs.loc[1].tolist(), "out_group_fraction":fraction_obs.loc[0].tolist(),"in_out_group_ratio":(fraction_obs.loc[1]/fraction_obs.loc[0]).tolist(),
          "in_group_mean_exp": mean_obs.loc[1].tolist(), "out_group_mean_exp": mean_obs.loc[0].tolist(),"fold_change":fold_change.tolist(), "pvals_adj":pvals_adj}
    df = pd.DataFrame(data=df) 
    #print("对靶标域与邻居域中所有spot，约2400+个，计算8个指标",df.shape) #(33538, 8)
    return df 

def relative_func(expres):
    #expres: an array counts expression for a gene
    maxd = np.max(expres) - np.min(expres)
    min_exp=np.min(expres)
    rexpr = (expres - min_exp)/maxd
    return rexpr

def plot_relative_exp(input_adata, gene, x_name, y_name,color,use_raw=False, spot_size=200000):
    adata=input_adata.copy()
    if use_raw:
        X=adata.raw.X
    else:
        X=adata.X
    if issparse(X):
        X=pd.DataFrame(X.A)
    else:
        X=pd.DataFrame(X)
    X.index=adata.obs.index
    X.columns=adata.var.index
    rexpr=relative_func(X.loc[:,gene])
    adata.obs["rexpr"]=rexpr
    fig=sc.pl.scatter(adata,x=x_name,y=y_name,color="rexpr",title=gene+"_rexpr",color_map=color,show=False,size=spot_size/adata.shape[0])
    return fig

def plot_log_exp(input_adata, gene, x_name, y_name,color,use_raw=False):
    adata=input_adata.copy()
    if use_raw:
        X=adata.X
    else:
        X=adata.raw.X
    if issparse(X):
        X=pd.DataFrame(X.A)
    else:
        X=pd.DataFrame(X)
    X.index=adata.obs.index
    X.columns=adata.var.index
    adata.obs["log"]=np.log((X.loc[:,gene]+1).tolist())
    fig=sc.pl.scatter(adata,x=x_name,y=y_name,color="log",title=gene+"_log",color_map=color,show=False,size=200000/adata.shape[0])
    return fig

def detect_subclusters(cell_id, x, y, pred, target_cluster, radius=3, res=0.2):
    df = {'cell_id': cell_id, 'x': x, "y":y, "pred":pred}
    df = pd.DataFrame(data=df)
    df.index=df['cell_id']
    target_df=df[df["pred"]==target_cluster]
    nbr=np.zeros([target_df.shape[0],len(set(df["pred"]))],dtype=int)
    num_nbr=[]
    row_index=0
    for index, row in target_df.iterrows():
        x=row["x"]
        y=row["y"]
        tmp_nbr=df[(df["x"]<x+radius) & (df["x"]>x-radius) & (df["y"]<y+radius) & (df["y"]>y-radius)]
        num_nbr.append(tmp_nbr.shape[0])
        for p in tmp_nbr["pred"]:
            nbr[row_index,int(p)]+=1
        row_index+=1
    #Minus out the cell itself
    nbr[:,target_cluster]=nbr[:,target_cluster]-1
    nbr=sc.AnnData(nbr)
    sc.pp.neighbors(nbr, n_neighbors=10)
    sc.tl.louvain(nbr,resolution=res)
    sub_cluster=nbr.obs['louvain'].astype(int).to_numpy()
    target_df["sub_cluster"]=sub_cluster
    target_df["sub_cluster"]=target_df["sub_cluster"].astype('category')
    tmp=[]
    for j in df.index:
        if j in target_df.index:
            tmp.append(target_df.loc[j,"sub_cluster"])
        else:
            tmp.append("-1")
    #ret = {'cell_id': cell_id, 'sub_cluster_'+str(target_cluster): tmp}
    #ret = pd.DataFrame(data=ret)
    #ret.index=ret['cell_id']
    ret=tmp
    return ret

def find_meta_gene(input_adata,
                    pred,
                    target_domain,
                    start_gene,
                    mean_diff=0,
                    early_stop=True,
                    max_iter=5,
                    use_raw=False):
    meta_name=start_gene
    adata=input_adata.copy()
    adata.obs["meta"]=adata.X[:,adata.var.index==start_gene]
    adata.obs["pred"]=pred
    num_non_target=adata.shape[0]
    for i in range(max_iter):
        #Select cells
        tmp=adata[((adata.obs["meta"]>np.mean(adata.obs[adata.obs["pred"]==target_domain]["meta"]))|(adata.obs["pred"]==target_domain))]
        tmp.obs["target"]=((tmp.obs["pred"]==target_domain)*1).astype('category').copy()
        if (len(set(tmp.obs["target"]))<2) or (np.min(tmp.obs["target"].value_counts().values)<5):
            print("Meta gene is: ", meta_name)
            return meta_name, adata.obs["meta"].tolist()
        #DE
        sc.tl.rank_genes_groups(tmp, groupby="target",reference="rest", n_genes=1,method='wilcoxon')
        adj_g=tmp.uns['rank_genes_groups']["names"][0][0]
        add_g=tmp.uns['rank_genes_groups']["names"][0][1]
        meta_name_cur=meta_name+"+"+add_g+"-"+adj_g
        print("Add gene: ", add_g)
        print("Minus gene: ", adj_g)
        #Meta gene
        adata.obs[add_g]=adata.X[:,adata.var.index==add_g]
        adata.obs[adj_g]=adata.X[:,adata.var.index==adj_g]
        adata.obs["meta_cur"]=(adata.obs["meta"]+adata.obs[add_g]-adata.obs[adj_g])
        adata.obs["meta_cur"]=adata.obs["meta_cur"]-np.min(adata.obs["meta_cur"])
        mean_diff_cur=np.mean(adata.obs["meta_cur"][adata.obs["pred"]==target_domain])-np.mean(adata.obs["meta_cur"][adata.obs["pred"]!=target_domain])
        num_non_target_cur=np.sum(tmp.obs["target"]==0)
        if (early_stop==False) | ((num_non_target>=num_non_target_cur) & (mean_diff<=mean_diff_cur)):
            num_non_target=num_non_target_cur
            mean_diff=mean_diff_cur
            print("Absolute mean change:", mean_diff)
            print("Number of non-target spots reduced to:",num_non_target)
        else:
            print("Stopped!", "Previous Number of non-target spots",num_non_target, num_non_target_cur, mean_diff,mean_diff_cur)
            print("Previous Number of non-target spots",num_non_target, num_non_target_cur, mean_diff,mean_diff_cur)
            print("Previous Number of non-target spots",num_non_target)
            print("Current Number of non-target spots",num_non_target_cur)
            print("Absolute mean change", mean_diff)
            print("===========================================================================")
            print("Meta gene: ", meta_name)
            print("===========================================================================")
            return meta_name, adata.obs["meta"].tolist()
        meta_name=meta_name_cur
        adata.obs["meta"]=adata.obs["meta_cur"]
        print("===========================================================================")
        print("Meta gene is: ", meta_name)
        print("===========================================================================")
    return meta_name, adata.obs["meta"].tolist()


def search_res(adata, adj, l, target_num, start=0.4, step=0.1, tol=5e-3, lr=0.05, max_epochs=10, r_seed=100, t_seed=100, n_seed=100, max_run=10):
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    res=start
    print("Start at res = ", res, "step = ", step)
    clf=SpaGCN()
    clf.set_l(l)
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=tol, lr=lr, max_epochs=max_epochs)
    _,y_pred, _=clf.predict()
    old_num=len(set(y_pred))
    print("Res = ", res, "Num of clusters = ", old_num)
    run=0
    while old_num!=target_num:
        random.seed(r_seed)
        torch.manual_seed(t_seed)
        np.random.seed(n_seed)
        old_sign=1 if (old_num<target_num) else -1
        clf=SpaGCN()
        clf.set_l(l)
        clf.train(adata,adj,init_spa=True,init="louvain",res=res+step*old_sign, tol=tol, lr=lr, max_epochs=max_epochs)
        _,y_pred, _=clf.predict()
        new_num=len(set(y_pred))
        print("Res = ", res+step*old_sign, "Num of clusters = ", new_num)
        if new_num==target_num:
            res=res+step*old_sign
            print("recommended res = ", str(res))
            return res
        new_sign=1 if (new_num<target_num) else -1
        if new_sign==old_sign:
            res=res+step*old_sign
            print("Res changed to", res)
            old_num=new_num
        else:
            step=step/2
            print("Step changed to", step)
        if run >max_run:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res
        run+=1
    print("recommended res = ", str(res))
    return res

###完成功能：检测spot检查每个点及其周围点的域分配。对于给定的点，如果其周围点的一半以上被分配到不同的域，则该点将被重新标记到与其周围点的主标签相同的域。
def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon": #如果是六边型，有6个邻居，
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]#一个个遍历行，即spot
        dis_tmp=dis_df.loc[index, :].sort_values() #得到这个spot与其他spot的距离
        nbs=dis_tmp[0:num_nbs+1] #得到距离最小的前7个邻居
        nbs_pred=pred.loc[nbs.index, "pred"] #得到这6个邻居的index,即spot
        self_pred=pred.loc[index, "pred"] #得到这个spot预测的label
        v_c=nbs_pred.value_counts() #得到邻居的邻居情况（2，3）（1，3）（5，1）得到邻居的相同个数
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):#如果这个spot，它的邻居label有一半不是这个spot的label; 
                                                                     #并且这个spot的邻居中某个标签很多，则需要重新赋label
           # print("i为XX时，需要refine_pred",i)
            refined_pred.append(v_c.idxmax()) #如果一半以上的邻居都属于其他label，则把它与邻居的最大label相同
        else:           
            refined_pred.append(self_pred)
    return refined_pred


