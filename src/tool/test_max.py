import scanpy as sc
import numpy as np
from scipy import sparse


def read_data(h5ad_path):
    adata = sc.read(h5ad_path)
    adata.layers["counts"] = adata.X.copy()
    X_norm = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X']
    X_log = sc.pp.log1p(X_norm, copy=True)
    adata.layers["lognorm"] = X_log
    return adata


# 示例调用
adata = read_data("../../datasets/Test2M/adata_hvg_top32400_only.h5ad")


# 检查最小最大值并统计 >6.3 的值
def count_above_threshold(adata, layer="lognorm", threshold=6.3):
    data_mat = adata.layers[layer]

    # 检查是否稀疏矩阵
    if sparse.issparse(data_mat):
        # 直接访问非零元素
        non_zero_data = data_mat.data
        # 统计非零元素中大于阈值的数量（零元素不可能大于正阈值）
        count = (non_zero_data > threshold).sum()
        # 可选：检查最值
        print(f"[稀疏矩阵] > {threshold} 的元素数: {count}")
        print(f"非零最小值: {non_zero_data.min():.4f}, 非零最大值: {non_zero_data.max():.4f}")
    else:
        # 密集矩阵直接计算
        count = (data_mat > threshold).sum()
        print(f"[密集矩阵] > {threshold} 的元素数: {count}")
        print(f"全局最小值: {data_mat.min():.4f}, 全局最大值: {data_mat.max():.4f}")
    return count


# 调用统计函数
count_above_6_3 = count_above_threshold(adata, threshold=6.3)