# some of the code is from https://github.com/JinmiaoChenLab/SEDR
import os

import networkx as nx
import scanpy as sc
import pandas as pd
from pathlib import Path
import torch
import scipy as sp
from scipy import stats
import scipy.sparse as sp
from scipy.spatial import distance
from torch_sparse import SparseTensor
from scanpy.readwrite import read_visium
from scanpy._utils import check_presence_download
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance


# edgeList to edgeDict
class graph():
    def __init__(self, data, k, rad_cutoff=150, distType='BallTree', ):
        super(graph, self).__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff

    def graph_computing(self):
        """
        Input: -adata.obsm['spatial']
               -distanceType:
                    -if get more information, https://docs.scipy.org/doc/scipy/reference/generated/scipy.
                     spatial.distance.cdist.html#scipy.spatial.distance.cdist
               -k: number of neighbors
        Return: graphList
        """
        dist_list = ["euclidean", "braycurtis", "canberra", "mahalanobis", "chebyshev", "cosine",
                     "jensenshannon", "mahalanobis", "minkowski", "seuclidean", "sqeuclidean", "hamming",
                     "jaccard", "jensenshannon", "kulsinski", "mahalanobis", "matching", "minkowski",
                     "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                     "sqeuclidean", "wminkowski", "yule"]

        if self.distType == 'spearmanr':
            SpearA, _ = stats.spearmanr(self.data, axis=1)
            graphList = []
            for node_idx in range(self.data.shape[0]):
                tmp = SpearA[node_idx, :].reshape(1, -1)
                res = tmp.argsort()[0][-(self.k + 1):]
                for j in np.arange(0, self.k):
                    graphList.append((node_idx, res[j]))
            print('%.4f neighbors per cell on average.' % (len(graphList) / self.data.shape[0]))

        elif self.distType == "BallTree":
            from sklearn.neighbors import BallTree
            tree = BallTree(self.data)
            dist, ind = tree.query(self.data, k=self.k + 1)
            indices = ind[:, 1:]
            graphList = []
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))
            print('%.4f neighbors per cell on average.' % (len(graphList) / self.data.shape[0]))

        elif self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k + 1)
            indices = ind[:, 1:]
            graphList = []
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))
            print('%.4f neighbors per cell on average.' % (len(graphList) / self.data.shape[0]))

        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity', include_self=False)
            A = A.toarray()
            graphList = []
            for node_idx in range(self.data.shape[0]):
                indices = np.where(A[node_idx] == 1)[0]
                for j in np.arange(0, len(indices)):
                    graphList.append((node_idx, indices[j]))
            print('%.4f neighbors per cell on average.' % (len(graphList) / self.data.shape[0]))

        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius=self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList = []
            for node_idx in range(indices.shape[0]):
                for j in range(indices[node_idx].shape[0]):
                    if distances[node_idx][j] > 0:
                        graphList.append((node_idx, indices[node_idx][j]))
            print('%.4f neighbors per cell on average.' % (len(graphList) / self.data.shape[0]))

        elif self.distType in dist_list:
            graphList = []
            for node_idx in range(self.data.shape[0]):
                tmp = self.data[node_idx, :].reshape(1, -1)
                distMat = distance.cdist(tmp, self.data, self.distType)
                res = distMat.argsort()[:self.k + 1]
                tmpdist = distMat[0, res[0][1:self.k + 1]]
                boundary = np.mean(tmpdist) + np.std(tmpdist)
                for j in np.arange(1, self.k + 1):
                    if distMat[0, res[0][j]] <= boundary:
                        graphList.append((node_idx, res[0][j]))
                    else:
                        pass
                print('%.4f neighbors per cell on average.' % (len(graphList) / self.data.shape[0]))

        else:
            raise ValueError(
                f"""\
                {self.distType!r} does not support. Disttype must in {dist_list} """)

        return graphList

    def List2Dict(self, graphList):
        """
        Return dict: eg {0: [0, 3542, 2329, 1059, 397, 2121, 485, 3099, 904, 3602],
                     1: [1, 692, 2334, 1617, 1502, 1885, 3106, 586, 3363, 101],
                     2: [2, 1849, 3024, 2280, 580, 1714, 3311, 255, 993, 2629],...}
        """
        graphdict = {}
        tdict = {}
        for graph in graphList:
            end1 = graph[0]
            end2 = graph[1]
            tdict[end1] = ""
            tdict[end2] = ""
            if end1 in graphdict:
                tmplist = graphdict[end1]
            else:
                tmplist = []
            tmplist.append(end2)
            graphdict[end1] = tmplist

        for i in range(self.data.shape[0]):
            if i not in tdict:
                graphdict[i] = []

        return graphdict

    def mx2SparseTensor(self, mx):

        """Convert a scipy sparse matrix to a torch SparseTensor."""
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).to(torch.long)
        col = torch.from_numpy(mx.col).to(torch.long)
        values = torch.from_numpy(mx.data)
        adj = SparseTensor(row=row, col=col, \
                           value=values, sparse_sizes=mx.shape)
        adj_ = adj.t()
        return adj_


    def pre_graph(self, adj):

        """ Graph preprocessing."""
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

        # a= adj_normalized
        # b=torch.FloatTensor(adj_normalized)
        # print("type(a),type(b)",type(a),type(b))

        return self.mx2SparseTensor(adj_normalized) #返回稀疏的sparse_tensor

    def main(self):
        adj_mtx = self.graph_computing()
        graphdict = self.List2Dict(adj_mtx)
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

        """ Store original adjacency matrix (without diagonal entries) for later """
        adj_pre = adj_org
        adj_pre = adj_pre - sp.dia_matrix((adj_pre.diagonal()[np.newaxis, :], [0]), shape=adj_pre.shape)
        adj_pre.eliminate_zeros()

        """ Some preprocessing."""
        adj_norm = self.pre_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)

        graph_dict = {
            "adj_norm": adj_norm,
            "adj_label": adj_label,
            "norm_value": norm}

        return graph_dict




def plot_clustering(adata, colors, savepath = None):
    my_colors = ["#E377C2","#8C564B","#9467BD","#D62728","#2CA02C","#FF7F0E","#1F77B4"]

    adata.obs['x_pixel'] = adata.obsm['spatial'][:, 0]
    adata.obs['y_pixel'] = adata.obsm['spatial'][:, 1]

    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    sc.pl.scatter(adata, alpha=1, x="x_pixel", y="y_pixel", color=colors, title='Clustering of 151673 slice',
                  palette=sns.color_palette('plasma', 7), show=False, ax=ax1) #color=colors
    ax1.set_aspect('equal', 'box')
    ax1.axis('off')
    ax1.axes.invert_yaxis()
    #
    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.spatial(adata, color=colors, title='Const_leiden',show=False)
    # if savepath is not None:
    #     plt.savefig(savepath, bbox_inches='tight')
    #


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.01, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res


def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

## 补充图h5ad理代码 filter(min=3)->normalize_total->scale->pca(300)
def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    # print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X
##补充图像处理代码
def _get_graph(
        # self,
        data,
        distType="BallTree",  #k=12
        k=20,  #KNN =20
        rad_cutoff=150,
):
    graph_dict = graph(data, distType=distType, k=k, rad_cutoff=rad_cutoff).main()
    print("Step 2: Graph computing is Done!")
    return graph_dict




def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()

    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)

    print('adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5


# from scanpy
def _download_visium_dataset(
        sample_id: str,
        spaceranger_version: str,
        base_dir='./data/',
):
    import tarfile

    url_prefix = f'https://cf.10xgenomics.com/samples/spatial-exp/{spaceranger_version}/{sample_id}/'

    sample_dir = Path(mk_dir(os.path.join(base_dir, sample_id)))

    # Download spatial data
    tar_filename = f"{sample_id}_spatial.tar.gz"
    tar_pth = Path(os.path.join(sample_dir, tar_filename))
    check_presence_download(filename=tar_pth, backup_url=url_prefix + tar_filename)
    with tarfile.open(tar_pth) as f:
        for el in f:
            if not (sample_dir / el.name).exists():
                f.extract(el, sample_dir)

    # Download counts
    check_presence_download(
        filename=sample_dir / "filtered_feature_bc_matrix.h5",
        backup_url=url_prefix + f"{sample_id}_filtered_feature_bc_matrix.h5",
    )


def load_visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_1', save_path='./data/'):
    if "V1_" in sample_id:
        spaceranger_version = "1.1.0"
    else:
        spaceranger_version = "1.2.0"
    _download_visium_dataset(sample_id, spaceranger_version, base_dir=save_path)
    adata = read_visium(os.path.join(save_path, sample_id))

    print('adata: (' + str(adata.shape[0]) + ', ' + str(adata.shape[1]) + ')')
    return adata
