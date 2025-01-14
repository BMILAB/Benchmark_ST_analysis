import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from . layers import GraphConvolution


class simple_GC_DEC(nn.Module):
    def __init__(self, nfeat, nhid, alpha=0.2):
        super(simple_GC_DEC, self).__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid=nhid
        #self.mu determined by the init method
        self.alpha=alpha

    def forward(self, x, adj): #进行卷积运算，
        x=self.gc(x, adj) #X(3639,50)  ###q可理解为预测的，具体来说，通过soft assignment获得的概率
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8) #（3639，7）
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True) #(3639,7)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X,adj,  lr=0.001, max_epochs=5000, update_interval=3, trajectory_interval=50,weight_decay=5e-4,opt="sgd",init="louvain",n_neighbors=10,res=0.4,n_clusters=10,init_spa=True,tol=1e-3):
        self.trajectory=[]
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)
        ##将pca降维后的50，与标准化的adj矩阵，输入到GCN中，得到卷积后的features(3639,50)
        features= self.gc(torch.FloatTensor(X),torch.FloatTensor(adj))
        #----------------------------------------------------------------        
        if init=="kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                #------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                #------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
        elif init=="louvain": #用louvain初始化聚类中心
            print("Initializing cluster centers with louvain, resolution = ", res)
            if init_spa:
                adata=sc.AnnData(features.detach().numpy())
            else:
                adata=sc.AnnData(X)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata,resolution=res) # 对初步卷积后的（3639，50）进行louvain聚类
            y_pred=adata.obs['louvain'].astype(int).to_numpy() #初始化后，得到3639个spot的标签[0,6]
            self.n_clusters=len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1) #（3639，51）前50个为嵌入，后一个为聚类label
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean()) #(7,50),第类聚类中心，它的向量取平均
        
        self.mu.data.copy_(torch.Tensor(cluster_centers))
           
        self.train() #用KL损失训练n个epoch，将GCN得到的概率
        for epoch in range(max_epochs): #200
            if epoch%update_interval == 0:
                _, q = self.forward(X,adj) #_,(3639,7)，经过卷积后，获得的(3639,7)的概率得分
                p = self.target_distribution(q).data #(3639,7) #p是靶标的分布，有用到前面louvain初始化，得到的label，然后计算每组label的聚类中心
                # print("type(self.target_distribution(q))",type(self.target_distribution(q)))
                # print("type(self.target_distribution(q).data):",type(self.target_distribution(q).data))
            if epoch%10==0:
                print("Epoch ", epoch) 
            optimizer.zero_grad()
            z,q = self(X, adj) #输入(3639,50)，（3639，3639），输出(3639,50)，(3639,7)，然后p,q之间通过KL损失，获得损失得分
            loss = self.loss_function(p, q) #kld损失，用训练与预测的结果
            loss.backward()
            optimizer.step()

            if epoch%trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            #Check stop criterion
            # if epoch==19:
            #     print(type(q),q.shape) #q是否为最终的嵌入 torch.Size([3639, 7])
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break



    def fit_with_init(self, X,adj, init_y, lr=0.001, max_epochs=5000, update_interval=1, weight_decay=5e-4,opt="sgd"):
        print("Initializing cluster centers with kmeans.")
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        features, _ = self.forward(X,adj)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(init_y,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
                p = self.target_distribution(q).data
            X=torch.FloatTensor(X)
            adj=torch.FloatTensor(adj)
            optimizer.zero_grad()
            z,q = self(X, adj) ###这个Z就是经GCN整合后的嵌入，之后基于它求出q后，z就没有再用
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self, X, adj):
        z,q = self(torch.FloatTensor(X),torch.FloatTensor(adj))
        return z, q




class GC_DEC(nn.Module):
    def __init__(self, nfeat, nhid1,nhid2, n_clusters=None, dropout=0.5,alpha=0.2):
        super(GC_DEC, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.mu = Parameter(torch.Tensor(n_clusters, nhid2))
        self.n_clusters=n_clusters
        self.alpha=alpha

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=True)
        x = self.gc2(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-6)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X,adj, lr=0.001, max_epochs=10, update_interval=5, weight_decay=5e-4,opt="sgd",init="louvain",n_neighbors=10,res=0.4):
        self.trajectory=[]
        print("Initializing cluster centers with kmeans.")
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

        features, _ = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
        #----------------------------------------------------------------
        
        if init=="kmeans":
            #Kmeans only use exp info, no spatial
            #kmeans = KMeans(self.n_clusters, n_init=20)
            #y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
            #Kmeans use exp and spatial
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().numpy())
        elif init=="louvain":
            adata=sc.AnnData(features.detach().numpy())
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata,resolution=res)
            y_pred=adata.obs['louvain'].astype(int).to_numpy()
        #----------------------------------------------------------------
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(X,adj)
                p = self.target_distribution(q).data
            if epoch%100==0:
                print("Epoch ", epoch) 
            optimizer.zero_grad()
            z,q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

    def fit_with_init(self, X,adj, init_y, lr=0.001, max_epochs=10, update_interval=1, weight_decay=5e-4,opt="sgd"):
        print("Initializing cluster centers with kmeans.")
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        features, _ = self.forward(X,adj)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(init_y,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
                p = self.target_distribution(q).data
            X=torch.FloatTensor(X)
            adj=torch.FloatTensor(adj)
            optimizer.zero_grad()
            z,q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self, X, adj):
        z,q = self(torch.FloatTensor(X),torch.FloatTensor(adj))
        return z, q


