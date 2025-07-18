import numpy as np
import scipy.sparse as sp
import anndata as ad
import gc
import h5py
import os

# === 1. 路径 ===
npy_path = "../../outputs/train10m/diffusion_model_norm/test7/cell_test7_repaint_ddim_pro.npy" # 数据尺寸: (82892, 32400) | 数据类型: float32
h5ad_path = "../../datasets/Test10M/test_7_top32400.h5ad"
out_path = "../../outputs/train10m/diffusion_model_norm/test7/cell_test7_repaint_ddim.h5ad"

# === 2. 使用内存映射加载数据 ===
print("使用内存映射加载数据...")
dense = np.load(npy_path, mmap_mode='r')  # 内存映射而不是全量加载
print(f"数据尺寸: {dense.shape} | 数据类型: {dense.dtype}") # 数据尺寸: (82892, 32400) | 数据类型: float32


# === 3. 检查稀疏度 (前1000行样本) ===
print("检查数据稀疏度...")
sample = dense[:1000] if dense.shape[0] > 1000 else dense
non_zero_fraction = np.count_nonzero(sample) / sample.size
print(f"非零元素占比: {non_zero_fraction:.2%}")

# === 4. 内存优化策略 ===
if non_zero_fraction < 0.8:  # 当稀疏度足够高时使用CSR
    # 分块构建稀疏矩阵 (内存节约方案)
    print(f"使用分块CSR转换 (稀疏度 {non_zero_fraction:.2%})...")
    chunksize = 1000  # 按行分块处理
    sparse_data = []
    sparse_indices = []
    sparse_indptr = [0]

    for start in range(0, dense.shape[0], chunksize):
        end = min(start + chunksize, dense.shape[0])
        chunk = dense[start:end]  # 内存映射访问当前块
        nonzero_chunk = chunk != 0  # 非零掩码

        # 收集块内非零数据
        for i in range(chunk.shape[0]):
            row_nonzero = nonzero_chunk[i]
            sparse_data.append(chunk[i][row_nonzero].astype(np.float32))  # 保持原始精度
            sparse_indices.append(np.flatnonzero(row_nonzero))
            sparse_indptr.append(sparse_indptr[-1] + np.sum(row_nonzero))

        # 及时释放内存
        del chunk, nonzero_chunk
        gc.collect()
        print(f"处理进度: {end}/{dense.shape[0]} ({(end / dense.shape[0]):.1%})")

    # 构建最终稀疏矩阵
    sparse_mat = sp.csr_matrix(
        (np.concatenate(sparse_data),
         np.concatenate(sparse_indices),
         np.array(sparse_indptr)),
        shape=dense.shape
    )
    print(f"CSR矩阵构建完成 | 非零元素: {sparse_mat.nnz}")
else:
    print("数据过于稠密，使用原始数组 (非稀疏模式)")
    sparse_mat = dense  # 直接使用密集数组（需要足够内存）

# === 5. 加载原始 h5ad ===
print("加载原始AnnData...")
adata = ad.read_h5ad(h5ad_path)

# === 4. 尺寸检查与分析 ===
if dense.shape[0] != adata.shape[0]:
    print(f"尺寸不匹配: \n"
          f"  - .npy 形状: {dense.shape}\n"
          f"  - h5ad 形状: {adata.shape}\n"
          "开始分析多余样本...")

    import scanpy as sc

    adata.layers["counts"] = adata.X.copy()
    X_norm = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X']
    X_log = sc.pp.log1p(X_norm, copy=True)
    adata.layers["lognorm"] = X_log

    # 分析多余样本的位置
    n_samples = min(dense.shape[0], adata.shape[0])

    # 比较前5个样本
    print("\n比较前5个样本的MSE:")
    for i in range(5):
        npy_row = dense[i]
        adata_row = adata.layers["lognorm"][i, :].toarray().flatten()
        mse = np.mean((npy_row - adata_row) ** 2)
        print(f"样本 {i}: MSE = {mse:.6f}")

    # 比较最后5个样本
    print("\n比较最后5个样本的MSE:")
    for i in range(-5, 0):
        # 处理h5ad负索引
        adata_idx = adata.shape[0] + i
        # 处理npy负索引
        npy_idx = dense.shape[0] + i

        npy_row = dense[npy_idx]
        adata_row = adata.layers["lognorm"][i, :].toarray().flatten()
        mse = np.mean((npy_row - adata_row) ** 2)
        print(f"样本 (npy index: {npy_idx}, h5ad index: {adata_idx}): MSE = {mse:.6f}")

    # 决定是否自动修复
    print("\n分析结论：")
    print("1. 如开头MSE较高，可能多余样本在开头")
    print("2. 如结尾MSE较高，可能多余样本在结尾")
    print("3. 建议检查输出并决定修复策略")
    raise ValueError("尺寸不匹配，请根据上述分析结果修复数据")


# === 7. 添加新图层 ===
print("加载添加...")
adata.layers["repaint"] = sparse_mat

# === 8. 清理内存后保存 ===
del dense
gc.collect()
print(f"保存到 {out_path} (此操作可能占用较多内存)...")
adata.write_h5ad(out_path)
print("保存完成!")