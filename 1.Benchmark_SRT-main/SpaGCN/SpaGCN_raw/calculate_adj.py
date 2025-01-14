import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import numba
##传入两点spot的索引，由索引找到对应的3D,计算3D欧式距离。
@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
	sum=0
	for i in range(t1.shape[0]): #t1.shape=3,因为为3维
		sum+=(t1[i]-t2[i])**2 #i为0-2，分别计算第一维，第二维，第三维，
	return np.sqrt(sum)

#自己写，传入两个三维坐标，计算欧式距离
# def distance_3d(coords1, coords2):
# #Calculates the euclidean distance between 2 lists of coordinates.
#     dist = 0
#     for (x, y) in zip(coords1, coords2): #遍历三次，第一维,第二维...
#         dist += (x - y)**2
#     return dist**0.5

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
	n=X.shape[0]
	adj=np.empty((n, n), dtype=np.float32) #（3639，3639）
	for i in numba.prange(n):
		for j in numba.prange(n):
			adj[i][j]=euclid_dist(X[i], X[j])
	return adj 

def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49):
	#beta to control the range of neighbourhood when calculate grey vale for one spot
	beta_half=round(beta/2)
	g=[]
	for i in range(len(x_pixel)):
		max_x=image.shape[0]
		max_y=image.shape[1]
		nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
		g.append(np.mean(np.mean(nbs,axis=0),axis=0))
	c0, c1, c2=[], [], []
	for i in g:
		c0.append(i[0])
		c1.append(i[1])
		c2.append(i[2])
	c0=np.array(c0)
	c1=np.array(c1)
	c2=np.array(c2)
	c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
	return c3

def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=False):
	#x,y,x_pixel, y_pixel are lists
	if histology:
		assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
		assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
		print("Calculateing adj matrix using histology image...")
		#beta to control the range of neighbourhood when calculate grey vale for one spot,即beta控制邻居范围 
		#alpha to control the color scale，即alpha控制颜色尺度
		beta_half=round(beta/2) #49/2=24
		g=[]
		for i in range(len(x_pixel)): #3639  #遍历每个spot，获得周围附近的像素点，取平均
			max_x=image.shape[0] #13332
			max_y=image.shape[1] #13332
			# 每遍历一个基因，获得三级的nbs坐标，如(49,49,3)，其实是获得50*50范围内，用平均代替某个点，更平均。
			a=max(0,x_pixel[i]-beta_half)
			b=min(max_x,x_pixel[i]+beta_half+1)
			c=max(0,y_pixel[i]-beta_half)
			d=min(max_y,y_pixel[i]+beta_half+1)
			f=image[a:b,c:d] #shape(49,49,3)
			nbs=image[a:b,c:d]
			A=np.mean(nbs,axis=0) #按列相加，（49，49，3）降成2维，得到（49，3），axis=0，变的是0维; 再进行一次再变为1维，先将二维矩阵按列相加，得到3个相加结果，然后取均值，得到一个值。
			g.append(np.mean(np.mean(nbs,axis=0),axis=0)) #3639个list，也就是将nbs取平均。即对每个spot附近的RGB图像中X,Y,Z，进行取平均。遍历完一个spot，得到一个三元组，即spot方圆内像素平均值。
		c0=[]
		c1=[]
		c2=[]
		for i in g:
			c0.append(i[0])
			c1.append(i[1])
			c2.append(i[2])
		c0=np.array(c0)
		c1=np.array(c1)
		c2=np.array(c2)
		#遍历3639个spot，得到X,Y,Z的方差
		#print("R,B,G的方差 Var of c0,c1,c2 = ", np.var(c0),np.var(c1),np.var(c2)) #R,B,G的方差
		c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
		c4=(c3-np.mean(c3))/np.std(c3) #得到ZV
		z_scale=np.max([np.std(x), np.std(y)])*alpha
		z=c4*z_scale #用一个值ZV*来表示组织学图像特征
		z=z.tolist()
		#print("spot三元组坐标(x,y,z)方差,其中Z由R,G,B方差放缩得到： ", np.var(x),np.var(y),np.var(z))
		X=np.array([x, y, z]).T.astype(np.float32) #X (3639,3),x,y为spot点像素点位置，新增了Z，由像素点位置的R,G,B获得的Z,将2D变成3D
		##X(3639,3),得到location(X,Y),还有组织学图像的Z
		
	else:
		#print("Calculateing adj matrix using xy only...")
		X=np.array([x, y]).T.astype(np.float32)
		
	return pairwise_distance(X) #计算X,Y,Z的欧式距离，当成adj里的值,所以adj理解为矩阵，并没用进行TOP选择构图

###下面是实际的adj矩阵构建，distance（）函数没写，是用来处理权重的
# def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
# 	#x,y,x_pixel, y_pixel are lists
# 	adj=np.zeros((len(x),len(x)))
# 	if histology:
# 		assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
# 		assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
# 		print("Calculateing adj matrix using histology image...")
# 		#beta to control the range of neighbourhood when calculate grey vale for one spot
# 		#alpha to control the color scale
# 		beta_half=round(beta/2)
# 		g=[]
# 		for i in range(len(x_pixel)):
# 			max_x=image.shape[0]
# 			max_y=image.shape[1]
# 			nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
# 			g.append(np.mean(np.mean(nbs,axis=0),axis=0))
# 		c0, c1, c2=[], [], []
# 		for i in g:
# 			c0.append(i[0])
# 			c1.append(i[1])
# 			c2.append(i[2])
# 		c0=np.array(c0)
# 		c1=np.array(c1)
# 		c2=np.array(c2)
# 		print("Var of c0,c1,c2 = ", np.var(c0),np.var(c1),np.var(c2))
# 		c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
# 		c4=(c3-np.mean(c3))/np.std(c3)
# 		z_scale=np.max([np.std(x), np.std(y)])*alpha
# 		z=c4*z_scale
# 		z=z.tolist()
# 		print("Var of x,y,z = ", np.var(x),np.var(y),np.var(z))
# 		for i in range(len(x)):
# 			if i%50==0:
# 				print("Calculating spot ", i)
# 			for j in range(len(x)):
# 				x1,y1,z1,x2,y2,z2=x[i],y[i],z[i],x[j],y[j],z[j]
# 				adj[i][j]=distance((x1,y1,z1),(x2,y2,z2))
#
# 	else:
# 		print("Calculateing adj matrix using xy only...")
# 		for i in range(len(x)):
# 			if i%50==0:
# 				print("Calculating spot", i)
# 			for j in range(len(x)):
# 				x1,y1,x2,y2=x[i],y[i],x[j],y[j]
# 				adj[i][j]=distance((x1,y1),(x2,y2))
# 	return adj
# #