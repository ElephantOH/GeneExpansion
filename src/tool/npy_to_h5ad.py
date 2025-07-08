import numpy as np
import scipy.sparse as sp
import anndata as ad

# === 1. 路径 ===
npy_path  = "../../outputs/train10m/diffusion_model_norm/test1/cell_repaint_noitce_test1.npy"     # .npy 数组文件
h5ad_path = "../../datasets/Test10M/test_1_top32400.h5ad"      # 原始 h5ad
out_path  = "../../outputs/train10m/diffusion_model_norm/test1/cell_ddim_test1.npy"   # 输出文件

# === 2. 读取旧的 h5ad ===
adata = ad.read_h5ad(h5ad_path)

# === 3. 读取 .npy 并转稀疏 ===
dense = np.load(npy_path, mmap_mode=None)     # 如需节省内存可改成 mmap_mode='r'
sparse_mat = sp.csr_matrix(dense)             # 常用 CSR，也可以用 csc_matrix

# === 4. 尺寸检查（推荐） ===
if sparse_mat.shape != adata.shape:
    raise ValueError(
        f"矩阵尺寸不符：npy={sparse_mat.shape}, adata={adata.shape}；"
        "请确认行列顺序或是否需要转置"
    )

# === 5. 写入新图层 ===
adata.layers["repaint"] = sparse_mat

# === 6. 保存 ===
adata.write_h5ad(out_path)
print(f"已保存到 {out_path}")