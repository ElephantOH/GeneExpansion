
import torch
import scanpy as sc
import pandas as pd
from scipy import sparse
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange



# 修改后的 ManifoldDataset 类支持两种模式
class ManifoldDataset(Dataset):
    def __init__(self,
                 h5ad_path,
                 map_path=None,  # 允许map_path为None
                 data_min=0,
                 data_max=6,
                 device="cuda:0",
                 mode="full",  # 添加模式参数
                 data_cache=None):  # 可选：用于轻量模式的数据缓存

        self.mode = mode
        self.adata = sc.read(h5ad_path)
        self.adata.layers["counts"] = self.adata.X.copy()

        # 仅在完整模式下进行归一化
        if self.mode == "full":
            X_norm = sc.pp.normalize_total(self.adata, target_sum=1e4, inplace=False)['X']
            X_log = sc.pp.log1p(X_norm, copy=True)
            self.adata.layers["lognorm"] = X_log

        self.data_min = data_min
        self.data_max = data_max
        self.device = device

        # 轻量模式支持从缓存加载数据
        if mode == "lightweight" and data_cache is not None:
            self.cell_types = data_cache.get("cell_types", None)
        else:
            self.cell_types = self.adata.obs['cell_type'].values

        # 仅在完整模式下需要映射
        if mode == "full" and map_path is not None:
            self.df_map = pd.read_csv(map_path)
            self.coords = self.df_map[['x', 'y']].to_numpy()

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        # 轻量模式只返回cell_type
        if self.mode == "lightweight":
            return {"cell_type": self.cell_types[idx]}

        # 完整模式计算完整信息
        # 加载基因矩阵
        gene_matrix = self.mapping_gene(self.reshape_gene(self.adata, idx))
        gene_matrix = torch.tensor(gene_matrix, dtype=torch.float32)
        gene_matrix = rearrange(gene_matrix, 'h w -> 1 h w')  # 添加通道维度

        gene_matrix = torch.clamp(gene_matrix, min=self.data_min, max=self.data_max)
        resize_value = (self.data_max - self.data_min) / 2.
        gene_matrix = (gene_matrix - resize_value) / resize_value

        sparse_code = self.get_sparse_code(gene_matrix)

        return {
            "cell_type": self.cell_types[idx],
            "sparse_code": sparse_code
        }

    # 轻量模式专用方法 - 快速获取所有cell_type
    def get_all_cell_types(self):
        return self.cell_types

    # 以下方法仅用于完整模式
    def reshape_gene(self, adata, row_index):
        x_row = adata.layers["lognorm"][row_index, :]
        # 转 dense 并拉平为 1D
        if sparse.issparse(x_row):
            x_row = x_row.toarray().flatten()
        else:
            x_row = np.array(x_row).flatten()
        n = len(x_row)
        new_size = int(np.ceil(np.sqrt(n)))
        total_size = new_size ** 2
        padding_length = total_size - n
        # 填充 0
        x_padded = np.pad(x_row, (0, padding_length), mode='constant')
        # reshape
        reshaped_matrix = x_padded.reshape(new_size, new_size)
        return reshaped_matrix


    def mapping_gene(self, gene_matrix):

        if self.coords.shape[0] != gene_matrix.size:
            raise ValueError(
                f"映射行数 {self.coords.shape[0]} 与 gene_matrix 元素数 {gene_matrix.size} 不符！"
            )

        gene_matrix_new = np.zeros_like(gene_matrix)
        # 将 gene_matrix flatten 成 1D，逐值搬到新坐标
        flat = gene_matrix.ravel()  # 行优先展平，与 csv 顺序一致
        for val, (x, y) in zip(flat, self.coords):
            gene_matrix_new[int(y), int(x)] = val
        return gene_matrix_new


    def get_sparse_code(self, gene_matrix):
        B, H, W = gene_matrix.shape
        device = gene_matrix.device
        assert H == 180 and W == 180, "输入图像必须是180x180"

        # 预计算中心点坐标 (89.5)
        center = (W - 1) / 2.0

        # 向量化计算切比雪夫距离 (避免重复创建网格)
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        chebyshev_dist = torch.max(torch.abs(x - center), torch.abs(y - center))

        # 直接展平所有数据用于向量化聚合
        flat_dist = chebyshev_dist.view(-1).long()  # [32400]
        flat_genes = gene_matrix.view(B, -1)  # [B, 32400]

        # 向量化聚合 (避免逐batch循环)
        layer_sums = torch.zeros(B, 90, device=device)
        layer_counts = torch.zeros(B, 90, device=device)

        # 一操作完成所有batch的聚合
        layer_sums.scatter_add_(
            1,
            flat_dist.unsqueeze(0).expand(B, -1),
            flat_genes
        )
        layer_counts.scatter_add_(
            1,
            flat_dist.unsqueeze(0).expand(B, -1),
            torch.ones_like(flat_genes)
        )

        # 计算均值并处理空层
        sparse_layer_code = torch.where(
            layer_counts > 0,
            layer_sums / layer_counts,
            torch.zeros(1, device=device)
        )

        return sparse_layer_code

