import torch
import numpy as np
from torch.nn.functional import grid_sample
import anndata as ad
import cv2
from scipy.ndimage import gaussian_filter
import squidpy as sq
from sklearn.neighbors import NearestNeighbors,KNeighborsRegressor
import hnswlib
import networkx as nx
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sn
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.metrics import r2_score
from scipy.spatial import distance_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# set default dtype
torch.set_default_dtype(torch.float64)
# set defauly device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#%% Data pre-processing
def process_Merfish_data(adata,min_counts = 10,max_counts = 4000,
                         min_volume = 50,max_volume = 3000,
                         n_top_genes = 6000):
    ###
    ### Filtering
    ###
    sc.pp.calculate_qc_metrics(adata, expr_type='counts', var_type='genes',
                               qc_vars=(), percent_top=(50, 100), 
                               log1p= False, inplace = True)  
    adata = adata[
        (adata.obs['total_counts'] >= min_counts) &
        (adata.obs['total_counts'] <= max_counts) &
        (adata.obs['volume'] >= min_volume) &
        (adata.obs['volume'] <= max_volume)
    ].copy()
    ###
    ### Normalization and log1p
    ###
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata

def process_Visium_data(adata, n_top_genes=5000,min_counts = 5000,max_counts = 35000,min_cells = 10):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, max_counts=max_counts)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata

def process_adata(adata, n_top_genes=5000):
    adata.var_names_make_unique()
    # adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # sc.pp.filter_cells(adata, min_counts=100)
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    # sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    return adata

def compute_svg_by_group(adatas, sample_groups, svg_mode="moran", top_k=5000, coord_type="grid"):
    group_svg_summary = {}

    for i, group_samples in enumerate(sample_groups):
        gene_sets = []

        for sample in group_samples:
            ads = adatas[sample]
            sq.gr.spatial_neighbors(ads, coord_type=coord_type)
            sq.gr.spatial_autocorr(ads, mode=svg_mode)

            svg_df = ads.uns[svg_mode + "I"].sort_values("I", ascending=False)
            top_genes = svg_df.index[:top_k]
            gene_sets.append(set(top_genes))

        # 求交集与并集
        intersection = set.intersection(*gene_sets)
        union = set.union(*gene_sets)

        group_svg_summary[i+1] = {
            "intersection": intersection,
            "union": union,
            "samples": group_samples
        }

    return group_svg_summary

#%% helper functions
def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val

def normalize(arr, t_min=0, t_max=1):
    diff = t_max - t_min
    diff_arr = np.max(arr) - np.min(arr)
    min_ = np.min(arr)
        
    norm_arr = ((arr - min_)/diff_arr * diff) + t_min
    
    return norm_arr

def partial_cut(adata, percentage, axis='x', is_lower=True):
    assert 0 < percentage < 1, "Percentage must be between 0 and 1"
    assert axis in ['x', 'y'], "Axis must be 'x' or 'y'"

    coord = 0 if axis == 'x' else 1
    values = adata.obsm['spatial'][:, coord]

    if is_lower:
        threshold = np.percentile(values, percentage * 100)
        selected_indices = np.where(values <= threshold)[0]
    else:
        threshold = np.percentile(values, 100 - percentage * 100)
        selected_indices = np.where(values >= threshold)[0]

    filtered_data = adata[selected_indices]
    filtered_anndata = ad.AnnData(
        X=filtered_data.X,
        obs=filtered_data.obs.copy(),
        var=filtered_data.var.copy(),
        uns=filtered_data.uns.copy(),
    )
    filtered_anndata.obsm['spatial'] = adata.obsm['spatial'][selected_indices]

    return filtered_anndata

def top_p_genes(adata, shared_gene_names, p =3):
    data_knn = adata.copy()
    X_knn = data_knn.obsm["spatial"]
    Y_knn = np.array(data_knn.X.todense())
    Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
    knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
    preds = knn.predict(X_knn)
    r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

    top_gene_indices = np.argsort(-r2_vals)[:p]
    top_gene_names = shared_gene_names[top_gene_indices]
    
    return(top_gene_names)
    
    


#%% Global and local deformation 
def computeA(angle,sx,sy,tx,ty,xI,xJ, centroid = True):
    xI = [torch.as_tensor(x) for x in xI]
    xJ = [torch.as_tensor(x) for x in xJ]
    # Rotation matrix 
    angle_rad = torch.tensor(angle * np.pi / 180)
    R = torch.tensor([
        [torch.cos(angle_rad), -torch.sin(angle_rad)],
        [torch.sin(angle_rad),  torch.cos(angle_rad)]
    ])
    
    # Scale matrix
    S = torch.diag(torch.tensor([sx, sy]))
    L = R@S
    
    if centroid:
        # Move the centroid of I to the centroid of J
        cI = torch.tensor([torch.mean(x) for x in xI])
        cJ = torch.tensor([torch.mean(x) for x in xJ])
        Tc = cJ - L@cI 
        T = torch.tensor([tx,ty]) + Tc
    else :
        T = torch.tensor([tx,ty])
    
    # make A
    O = torch.tensor([0.,0.,1.])
    A = torch.cat((torch.cat((L,T[:,None]),1),O[None]))
    return A

def interp(x,I,phii,**kwargs):
    # convert to tensor
    x = [torch.as_tensor(xi) for xi in x]
    I = torch.as_tensor(I)
    
    if isinstance(phii,torch.Tensor):
        phii = torch.clone(phii)
    else:
        phii = torch.tensor(phii)

    for i in range(2):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    phii *= 2.0
    phii -= 1.0

    out = grid_sample(I[None],phii.flip(0).permute((1,2,0))[None],align_corners=True,**kwargs)
    return out[0]

def transform_image_with_A(A,xJ,xI,I):
    A = torch.as_tensor(A,device = I.device)
    
    I = torch.as_tensor(I)
    xI = [torch.as_tensor(x) for x in xI]
    xJ = [torch.as_tensor(x) for x in xJ]
    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    
    Ai = torch.linalg.inv(A)
    # transform sample points
    Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]    
    # # there is no need to diffeo as
    # for t in range(nt-1,-1,-1):
    #     Xs = Xs + utils.interp(xv,-v[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
    # transform image
    AI = interp(xI,I,Xs.permute(2,0,1),padding_mode="border")
    
    return AI

def build_transform(xv,v,A,XJ,forward = False):
    # tensor
    A = torch.as_tensor(A,device = v.device)
    # if it is already in meshgrid form we just need to make sure it is a tensor
    # need meshgrid
    XJ = XJ.clone().detach().to(device=v.device)
    
    if forward :
        Xs = torch.clone(XJ)
        nt = v.shape[0]
        for t in range(nt):
            Xs = Xs + interp(xv,v[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
        Xs = (A[:2,:2]@Xs[...,None])[...,0] + A[:2,-1] 
    else :
        Ai = torch.linalg.inv(A)
        # transform sample points
        Xs = (Ai[:-1,:-1]@XJ[...,None])[...,0] + Ai[:-1,-1]    
        # now diffeo, not semilagrange here
        nt = v.shape[0]
        for t in range(nt-1,-1,-1):
            Xs = Xs + interp(xv,-v[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
      
    return Xs 

def transform_image_source_to_target(xv,v,A,xI,I,XJ):
    phii = build_transform(xv,v,A,XJ,forward = False)    
    phiI = interp(xI,I,phii.permute(2,0,1),padding_mode="border")
    return phiI
      
def transform_image_target_to_source(xv,v,A,xJ,J,XI):
    phi = build_transform(xv,v,A,XJ=XI,forward = True)    
    phiiJ = interp(xJ,J,phi.permute(2,0,1),padding_mode="border")
    return phiiJ

def transform_points_source_to_target(xv,v,A,pointsI):
    # tensor
    A = torch.as_tensor(A,device = v.device)
    if isinstance(pointsI,torch.Tensor):
        pointsI = pointsI.to(device=v.device,dtype=A.dtype).clone()
    else:
        pointsI = torch.tensor(pointsI,device = v.device,dtype=A.dtype)
        
    nt = v.shape[0]
    for t in range(nt):            
        pointsI += interp(xv,v[t].permute(2,0,1),pointsI.T[...,None])[...,0].T/nt
    pointsI = (A[:2,:2]@pointsI.T + A[:2,-1][...,None]).T
    return pointsI

def transform_points_target_to_source(xv,v,A,pointsI):
    # tensor
    A = torch.as_tensor(A,device = v.device)
    if isinstance(pointsI,torch.Tensor):
        pointsI = pointsI.to(device=v.device,dtype=A.dtype).clone()
    else:
        pointsI = torch.tensor(pointsI,device = v.device,dtype=A.dtype)
        
    Ai = torch.linalg.inv(A)
    pointsI = (Ai[:2,:2]@pointsI.T + Ai[:2,-1][...,None]).T
    nt = v.shape[0]
    for t in range(nt):            
        pointsI += interp(xv,-v[t].permute(2,0,1),pointsI.T[...,None])[...,0].T/nt
    return pointsI




#%% Rasterization
def rasterize(x, y, g, dx=1.0, blur = 1.0, expand=1.0, normalization = True,use_windowing = True):   
    # Construct Meshgrid
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    minx,maxx = (minx+maxx)/2.0 - (maxx-minx)/2.0*expand, (minx+maxx)/2.0 + (maxx-minx)/2.0*expand
    miny,maxy = (miny+maxy)/2.0 - (maxy-miny)/2.0*expand, (miny+maxy)/2.0 + (maxy-miny)/2.0*expand
    X_ = np.arange(minx,maxx,dx)
    Y_ = np.arange(miny,maxy,dx)
    X = np.stack(np.meshgrid(X_,Y_)) 
    W = np.zeros((X.shape[1],X.shape[2],g.shape[1]))
    
    # Reshape gene expression and normilization
    g = np.resize(g,x.size)
    if not np.all(g == 1):
        g = normalize(g)
    
    count = 0
    # Gaussian kernel k(x_i, x_j) = exp(-(|x_i - x_j|^2/2) / 2\sigma^2)
    for x_,y_,g_ in zip(x,y,g): 
        if not use_windowing: # legacy version
            k = np.exp( - ( (X[0][...,None] - x_)**2 + (X[1][...,None] - y_)**2 )/(2.0*(dx*blur*2)**2)  )
            k /= np.sum(k,axis=(0,1),keepdims=True)
            k *= g_
            W += k
        else: # use a small window
            r = int(np.ceil(blur*4))
            col = np.round((x_ - X_[0])/dx).astype(int)
            row = np.round((y_ - Y_[0])/dx).astype(int)
            
            row0 = np.floor(row-r).astype(int)
            row1 = np.ceil(row+r).astype(int)                    
            col0 = np.floor(col-r).astype(int)
            col1 = np.ceil(col+r).astype(int)
            # we need boundary conditions
            row0 = np.minimum(np.maximum(row0,0),W.shape[0]-1)
            row1 = np.minimum(np.maximum(row1,0),W.shape[0]-1)
            col0 = np.minimum(np.maximum(col0,0),W.shape[1]-1)
            col1 = np.minimum(np.maximum(col1,0),W.shape[1]-1)
            
            k =  np.exp( - ( (X[0][row0:row1+1,col0:col1+1,None] - x_)**2 + (X[1][row0:row1+1,col0:col1+1,None] - y_)**2 )/(2.0*(dx*blur*2)**2)  )
            k /= np.sum(k,axis=(0,1),keepdims=True)  
            k *= g_

            W[row0:row1+1,col0:col1+1,:] += k #range of voxels -oka
            
        if not count%10000 or count ==(x.shape[0]-1):
            print(f'{count} of {x.shape[0]}')
        count += 1
    
    W = np.abs(W)
    W = W.transpose((-1,0,1))
    
    if normalization:
        W = normalize(W)
    
    return X_,Y_,W


def rasterize_image(x, y, img, blur = 1.0, dx = 1.0,sigma = 1.0,expand=1.0,ksize = 3, normalization = True):   
    # grasycale
    V = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel operator
    sobelx = cv2.Sobel(V, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(V, cv2.CV_64F, 0, 1, ksize=ksize)
    # Processed image
    Img = cv2.magnitude(sobelx, sobely)
    
    # Construct Meshgrid
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    minx,maxx = (minx+maxx)/2.0 - (maxx-minx)/2.0*expand, (minx+maxx)/2.0 + (maxx-minx)/2.0*expand
    miny,maxy = (miny+maxy)/2.0 - (maxy-miny)/2.0*expand, (miny+maxy)/2.0 + (maxy-miny)/2.0*expand
    X_ = np.arange(minx,maxx,dx)
    Y_ = np.arange(miny,maxy,dx)
    X = np.stack(np.meshgrid(X_,Y_)) 
    W = np.zeros((X.shape[1],X.shape[2]))
    
    count = 0

    for x_,y_ in zip(x,y): 
        r = int(np.ceil(blur*4))
        col = np.round((x_ - X_[0])/dx).astype(int)
        row = np.round((y_ - Y_[0])/dx).astype(int)
        
        row0 = np.floor(row-r).astype(int)
        row1 = np.ceil(row+r).astype(int)                    
        col0 = np.floor(col-r).astype(int)
        col1 = np.ceil(col+r).astype(int)
        # we need boundary conditions
        row0 = np.minimum(np.maximum(row0,0),W.shape[0]-1)
        row1 = np.minimum(np.maximum(row1,0),W.shape[0]-1)
        col0 = np.minimum(np.maximum(col0,0),W.shape[1]-1)
        col1 = np.minimum(np.maximum(col1,0),W.shape[1]-1)

        col_ = np.round(x_).astype(int)
        row_ = np.round(y_).astype(int)
        
        col0_ = col_ - (col-col0)
        col1_ = col_ + (col1-col)
        row0_ = row_ - (row-row0)
        row1_ = row_ + (row1-row)
        
        
        window = Img[row0_:row1_+1,col0_:col1_+1]
        smooth_grad = gaussian_filter(window, sigma = sigma)
#        smooth_grad /= np.sum(smooth_grad,keepdims=True)  
        W[row0:row1+1,col0:col1+1] += smooth_grad 
            
        if not count%10000 or count ==(x.shape[0]-1):
            print(f'{count} of {x.shape[0]}')
        count += 1
    
    W = np.abs(W)

    
    if normalization:
        W = normalize(W)
    
    return X_,Y_,W

def rasterize_channel(x, y, s, dx=1.0, blur = 1.0, expand=1.0,normalization = True):
    
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    minx,maxx = (minx+maxx)/2.0 - (maxx-minx)/2.0*expand, (minx+maxx)/2.0 + (maxx-minx)/2.0*expand
    miny,maxy = (miny+maxy)/2.0 - (maxy-miny)/2.0*expand, (miny+maxy)/2.0 + (maxy-miny)/2.0*expand
    X_ = np.arange(minx,maxx,dx)
    Y_ = np.arange(miny,maxy,dx)
    
    X = np.stack(np.meshgrid(X_,Y_)) # note this is xy order, not row col order

    W = np.zeros((X.shape[1],X.shape[2],s.shape[1]))
    
    # Normalize 
    s_normalized = np.stack([normalize(s[:,i]) for i in range(s.shape[1])], axis=1)
    
    count = 0
    for x_,y_ in zip(x,y):
        r = int(np.ceil(blur*4))
        col = np.round((x_ - X_[0])/dx).astype(int)
        row = np.round((y_ - Y_[0])/dx).astype(int)
        
        row0 = np.floor(row-r).astype(int)
        row1 = np.ceil(row+r).astype(int)                    
        col0 = np.floor(col-r).astype(int)
        col1 = np.ceil(col+r).astype(int)
        # we need boundary conditions
        row0 = np.minimum(np.maximum(row0,0),W.shape[0]-1)
        row1 = np.minimum(np.maximum(row1,0),W.shape[0]-1)
        col0 = np.minimum(np.maximum(col0,0),W.shape[1]-1)
        col1 = np.minimum(np.maximum(col1,0),W.shape[1]-1)
        
        k = np.exp( - ( (X[0][row0:row1+1,col0:col1+1,None] - x_)**2 + (X[1][row0:row1+1,col0:col1+1,None] - y_)**2 )/(2.0*(dx*blur*2)**2)  )
        k /= np.sum(k,axis=(0,1),keepdims=True)*dx**2
        
        factor = s_normalized[count]
        W[row0:row1+1,col0:col1+1,:] += k * factor
        
        if not count%10000 or count ==(x.shape[0]-1):
            print(f'{count} of {x.shape[0]}')
        count += 1
        
    W = np.abs(W)
    W = W.transpose((-1,0,1))
    # normalize
    if normalization:
        W = np.stack([normalize(W[i,:,:]) for i in range(W.shape[0])], axis=0)
    
    # rename
    output = X_,Y_,W
    return output




#%% Accuracy
def spatial_cross_correlation(slice1, slice2,use_rep = 'spatial',beta = 1.0):
    coords_1 = slice1.obsm[use_rep]
    coords_2 = slice2.obsm[use_rep]

    assert np.all(slice1.var_names == slice2.var_names), "Gene sets must match."

    genes = slice1.var_names

    # 计算距离矩阵，返回权重矩阵W
    dist = distance_matrix(coords_1, coords_2)
    W = np.exp(-beta * dist**2)

    S_ij = np.sum(W)
    n = W.shape[0]

    results = []

    for gene in genes:
        x = slice1[:, gene].X
        y = slice2[:, gene].X

        x = x.toarray().flatten() if hasattr(x, "toarray") else x.flatten()
        y = y.toarray().flatten() if hasattr(y, "toarray") else y.flatten()

        x_mean = x.mean()
        y_mean = y.mean()

        numerator = np.sum(W * np.outer(x - x_mean, y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

        if denominator == 0:
            score = 0.0
        else:
            score = n / S_ij * numerator / denominator

        results.append((gene, score))

    results.sort(key=lambda x: x[1], reverse=True)
    results = pd.DataFrame(results,columns= ['gene','score'])
    results = results.set_index('gene')
    
    return results

def joint_clustering_evaluation(adata_concat,use_rep = 'spatial'):
    # 标准化
    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    sc.pp.highly_variable_genes(adata_concat, n_top_genes=6000, subset=True, flavor="seurat")
    sc.pp.scale(adata_concat, max_value=10)
    # PCA
    sc.tl.pca(adata_concat, svd_solver='arpack')

    # 用GMM聚类
    X_pca = adata_concat.obsm['X_pca'][:,:30]
    # 空间信息
    coords = adata_concat.obsm[use_rep]
    # 标准化空间坐标和 PCA 表达
    scaler_expr = StandardScaler().fit(X_pca)
    X_pca_scaled = scaler_expr.transform(X_pca)
    scaler_coord = StandardScaler().fit(coords)
    coords_scaled = scaler_coord.transform(coords)
    # 合并：表达 + 空间
    X_joint = np.concatenate([X_pca_scaled, coords_scaled], axis=1)  

    Nclust = len(adata_concat.obs['Ground Truth'].unique() )
    
    gmm = GaussianMixture(n_components=Nclust, init_params='kmeans', covariance_type='full', random_state=42)
    gmm_labels = gmm.fit_predict(X_joint)
    adata_concat.obs['gmm'] = gmm_labels.astype(str)
    # 可视化UMAP
#    sc.pp.neighbors(adata_concat, n_neighbors=15, use_rep="X_pca")
#    sc.tl.umap(adata_concat)
#    sc.pl.umap(adata_concat, color=['gmm', 'slice_name'], title=["GMM Clustering", "Source vs Target"])
    # 计算 ARI/NMI
    ari = adjusted_rand_score(adata_concat.obs["gmm"], adata_concat.obs["Ground Truth"])
    nmi = normalized_mutual_info_score(adata_concat.obs["gmm"], adata_concat.obs["Ground Truth"])
    return ari, nmi


def ratio_and_accuracy(slice1, slice2, k=1, use_rep='spatial', ratio=0.8, kmax=51):
    r = 0
    # 在不同的最近邻数 k 下反复运行 MNN 匹配，直到从 slice1 中有足够比例（例如 80%）的细胞/spot 在 slice2 中找到了 MNN 配对，并在此基础上评估它们的标签（如细胞类型）是否一致。
    while r < ratio and k <= kmax:
        mnn_dict = compute_mnn_dict(slice1, slice2, use_rep=use_rep, k= k, verbose=0)

        ref_names = slice1.obs_names.to_list()
        target_names = slice2.obs_names.to_list()
        
        ref_amount = len(ref_names)
        ref_spots = 0
              
        all = 0
        correct = 0
        # 对每个 MNN 对进行匹配评估：
        # 对于 slice1 中每个参与匹配的 spot，找出它的 Ground Truth 标签。
        # 对应的匹配点来自 slice2，比较标签是否一致，如果一致就算预测正确。
        for key, matched_names in mnn_dict.items():
            if key in ref_names:
                ref_spots += 1
                pos1 = ref_names.index(key)
                gt1 = slice1.obs["Ground Truth"].iloc[pos1]
                for i in matched_names:
                    if i in target_names:
                        pos2 = target_names.index(i)
                        gt2 = slice2.obs["Ground Truth"].iloc[pos2]
                        all += 1
                        if gt1 == gt2:
                            correct += 1
        if all == 0:
            accu = 0
        else:
            accu = correct / all

        r = round(ref_spots / ref_amount, 2)
        if r < ratio:
            k += 5

    return [r, accu]



def compute_mnn_dict(slice1,slice2, k = 50, use_rep = 'spatial', approx = True, verbose = 1):
    ds1 = slice1.obsm[use_rep]
    ds2 = slice2.obsm[use_rep]
    names1 = slice1.obs_names.to_list()
    names2 = slice2.obs_names.to_list()
    # if k>1，one point in ds1 may have multiple MNN points in ds2.
    match = mnn(ds1, ds2, names1, names2, knn=k, approx = approx)
    # 构建无向图表示 MNN 对，MNN 配对作为边。点为细胞名称，边表示 MNN 关系。
    G = nx.Graph()
    G.add_edges_from(match)
    # 获取图中的邻接信息
    # 所有参与 MNN 的细胞点。
    node_names = np.array(G.nodes)
    # 邻接矩阵的稀疏表示。
    adj = nx.adjacency_matrix(G)
    # 把每个节点的邻居索引（即 MNN）拆分成一个列表。
    tmp = np.split(adj.indices, adj.indptr[1:-1])
    # 构建 MNN 字典
    mnns = {}
    for idx, key in enumerate(node_names):
        neighbor_idxs = tmp[idx]
        mnns[key] = list(node_names[neighbor_idxs])
        
    return(mnns)
    
def mnn(ds1, ds2, names1, names2, knn = 20, approx = True):
    if approx: 
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(n_neighbors = knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match

def nn_approx(ds1, ds2, names1, names2, knn=50):
    # 使用hnswlib实现快速近似k近邻搜索
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match




#%% plots
def plot_Merfish_qc(adata):
    sc.pp.calculate_qc_metrics(adata, expr_type='counts', var_type='genes',
                               qc_vars=(), percent_top=(50, 100), 
                               log1p= False, inplace = True)   
    #filter cells based on transcript counts and volume of the cell
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].set_title("Total transcripts per cell")
    sn.histplot(
        adata.obs["total_counts"],
        kde=False,
        ax=axs[0],
    )
    axs[1].set_title("Unique genes per cell")
    sn.histplot(
        adata.obs["n_genes_by_counts"],
        kde=False,
        ax=axs[1],
    )
    axs[2].set_title("Volume of segmented cells")
    sn.histplot(
        adata.obs["volume"],
        kde=False,
        ax=axs[2],
    )

def plot_slice(adata,layer_to_color_map,title,ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    # image
    img = adata.uns['lowres_image']
    ax.imshow(img)
    # sactter
    colors = list(adata.obs['Ground Truth'].astype('str').map(layer_to_color_map))
    ax.scatter(adata.obsm['spatial'][:,0],adata.obsm['spatial'][:,1],s=2,color=colors)
    ax.set_title(f'{title}',size=12)
    ax.axis('off')

def plot_fitness(solution_fitness):
    fitness_values = np.array(solution_fitness)
    plt.plot(1/fitness_values-1)
    plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations (Log Scale)')
    plt.show()
    
    
# =============================================================================
# def plot_Visium_qc(adata):
#     adata.var_names_make_unique()
#     adata.var["mt"] = adata.var_names.str.startswith("MT-")
#     sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
#     #filter cells based on transcript counts and volume of the cell
#     fig, axs = plt.subplots(2, 2, figsize=(9, 10))
#     axs[0,0].set_title("Total transcripts per cell")
#     sn.histplot(
#         adata.obs["total_counts"],
#         kde=False,
#         ax=axs[0,0],
#     )
#     axs[0,1].set_title("Unique genes per cell")
#     sn.histplot(
#         adata.obs["n_genes_by_counts"],
#         kde=False,
#         ax=axs[0,1],
#     )
#     axs[1,0].set_title("Total cells per gene")
#     sn.histplot(
#         adata.var["total_counts"],
#         kde=False,
#         ax=axs[1,0],
#     )
#     axs[1,1].set_title("Unique cell per gene")
#     sn.histplot(
#         adata.var["n_cells_by_counts"],
#         kde=False,
#         ax=axs[1,1],
#     )
#     
# =============================================================================


#%% Results
def alignedada(resu, source):
    v = resu['v']
    xv = resu['xv']
    A = resu['A']
    ### Plots results
    spatial = source.obsm['spatial']
    spatial = spatial[:,[1,0]]
    spatial = torch.as_tensor(spatial,device = v.device)

    tpointsI= transform_points_source_to_target(xv,v,A, spatial)
    #switch tensor from cuda to cpu for plotting with numpy
    if tpointsI.is_cuda:
        tpointsI = tpointsI.cpu()
    aligned_spatial = tpointsI[:,[1,0]].numpy()
    
    aligned_source = source.copy()
    aligned_source.obsm['aligned'] = aligned_spatial
    
    return aligned_source

def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, list):
        return [to_cpu(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_cpu(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    else:
        return obj

#%% plot results
###
### Biological analysis using cossin similarity
###
def plot_cossim(ada_source,ada_target,common_genes,gene_names_to_keep,ax,title,pixel_size = 30,fs0 = 10):
    # 给每个 AnnData 加上 pixel 信息
    def assign_pixel(adata, coord_key,pixel_size= pixel_size):
        coords = adata.obsm[coord_key]
        pixel_key = [tuple(int(x // pixel_size) for x in xy) for xy in coords]
        adata.obs['pixel'] = pixel_key

    assign_pixel(ada_source,coord_key= 'spatial')
    assign_pixel(ada_target,coord_key= 'spatial')

    # Step 4: 按 pixel 聚合表达（对 common_genes）
    def aggregate_by_pixel(adata, genes):
        df = pd.DataFrame(
            adata[:, genes].X.toarray() if hasattr(adata.X, 'toarray') else adata[:, genes].X,
            columns=genes
        )
        df['pixel'] = adata.obs['pixel'].values
        grouped = df.groupby('pixel').sum()
        grouped.index = pd.Index(grouped.index)  # 强制为 index
        return grouped


    expr_source_pixel = aggregate_by_pixel(ada_source, common_genes)
    expr_target_pixel = aggregate_by_pixel(ada_target, common_genes)

    # Step 5: 找出两者都有的 pixel
    shared_pixels = expr_source_pixel.index.intersection(expr_target_pixel.index)
    # 仅保留共享的 pixel
    expr_source_pixel = expr_source_pixel.loc[shared_pixels]
    expr_target_pixel = expr_target_pixel.loc[shared_pixels]


    # Step 6: 计算 cosine similarity
    results = []
    for gene in common_genes:
        vec1 = expr_source_pixel[gene].values.reshape(1, -1)
        vec2 = expr_target_pixel[gene].values.reshape(1, -1)
        
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            similarity = np.nan  # 跳过全零表达
        else:
            similarity = cosine_similarity(vec1, vec2)[0, 0]
        results.append((gene, similarity))

    # Step 7: 转成 DataFrame 输出
    sim_df = pd.DataFrame(results, columns=['gene', 'cosine_similarity'])

    # print(np.median(sim_df.iloc[:,1]))

    # ------------------------
    # 总表达量 和余弦相似度
    # ------------------------
    total_expr = expr_source_pixel[common_genes].sum(axis=0) + expr_target_pixel[common_genes].sum(axis=0)
    gene_total_expr = total_expr.to_frame(name='total_expr')
    gene_total_expr['gene'] = gene_total_expr.index
    merged = pd.merge(sim_df, gene_total_expr, on='gene')

    # bins
    bins = np.linspace(0, 1, 11)
    merged['sim_bin'] = pd.cut(merged['cosine_similarity'], bins=bins)
    # 按 bin 聚合表达量
    grouped = merged.groupby('sim_bin')['total_expr'].sum().reset_index()
    # 使用右边界作为刻度
    bin_rights = [interval.right for interval in grouped['sim_bin']]
    # 设置标签
    yticks = bin_rights
    yticklabels = [str(round(tick, 1)) for tick in bin_rights]

    ###
    ### Plots
    ###
    # 1. 获取 gene → similarity 映射
    gene_sims = []
    for gene in gene_names_to_keep:
        val = sim_df.query("gene == @gene")['cosine_similarity'].iloc[0]
        gene_sims.append((gene, val))

    # 2. 合并：对 similarity 四舍五入后分组
    merged = {}
    for gene, val in gene_sims:
        key = round(val, 2)  # 保留三位小数作为合并标准
        if key not in merged:
            merged[key] = {'genes': [], 'raw_vals': []}
        merged[key]['genes'].append(gene)
        merged[key]['raw_vals'].append(val)

    # 3. 画主图
    ax.barh(y=bin_rights, width=grouped['total_expr'], color='gray', height=0.1)

    # 添加均值线
    median_val = np.median(sim_df.iloc[:, 1])
    ax.axhline(y=median_val, color='red', linestyle='--', linewidth=1)
    ax.text(x=max(grouped['total_expr']) * 0.5, y=median_val,
             s=f'Median: {median_val:.2f}', color='red', va='top', ha='right',size = fs0)
    # 4. 遍历 merged 画标线
    for key, info in merged.items():
        genes = info['genes']
        raw_vals = info['raw_vals']
        mean_sim = np.mean(raw_vals)  # 使用实际平均值作为线位置
        label = ",".join(genes)

        ax.axhline(y=mean_sim, color='black', linestyle='--', linewidth=1)
        ax.text(x=max(grouped['total_expr']) * 1.1, y=mean_sim,
                 s=f'{label}: {mean_sim:.2f}',
                 va='bottom', ha='right',size = fs0,fontstyle='italic')
    # 美化图
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.set_ylabel("Cosine Similarity")
    ax.set_xlabel("Total Expression")
    ax.set_title(title)
    

