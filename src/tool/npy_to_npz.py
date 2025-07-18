import numpy as np
from scipy import sparse
from tqdm import tqdm  # 进度条（可选）


input_path = "../../outputs/train10m/diffusion_model_norm/test6/cell_test6_repaint_ddim.npy"

output_path = input_path[:-1] + 'z'

# ===== 1. 使用内存映射加载数组 (避免完全加载到内存) =====
dense_array = np.load(input_path, mmap_mode='r')  # 关键修改：mmap_mode='r'

# ===== 2. 分块处理数组 (控制内存峰值) =====
n_rows, n_cols = dense_array.shape
chunk_size = 1000  # 每次处理的行数，根据可用内存调整（越大越快但越吃内存）

# 预分配稀疏矩阵组件 (COO格式)
rows, cols, data = [], [], []

# 分块迭代处理数组
for start_idx in tqdm(range(0, n_rows, chunk_size)):
    end_idx = min(start_idx + chunk_size, n_rows)

    # 提取当前分块（内存映射仅加载所需部分）
    chunk = dense_array[start_idx:end_idx]

    # 找到当前分块中非零元素的位置和值
    x_coords, y_coords = np.nonzero(chunk)
    values = chunk[x_coords, y_coords]

    # 调整行索引（全局坐标）
    x_coords += start_idx

    # 收集结果
    rows.append(x_coords)
    cols.append(y_coords)
    data.append(values)

# ===== 3. 构建最终稀疏矩阵 =====
rows = np.concatenate(rows)
cols = np.concatenate(cols)
data = np.concatenate(data)

# 创建 COO 格式稀疏矩阵（自动转换为 CSR）
sparse_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

# ===== 4. 保存优化 =====
# 方案 A: 直接保存 (确保有足够内存)
sparse.save_npz(output_path, sparse_matrix)