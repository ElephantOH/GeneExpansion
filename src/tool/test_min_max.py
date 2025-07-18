import numpy as np

# 1. 使用内存映射加载大文件（只读模式）
data = np.load('../../outputs/train10m/diffusion_model_norm/test7/cell_test7_inpaint_ddnm.npy', mmap_mode='r')

# 2. 只读取第一行数据（避免全文件加载）
first_row = data[0]  # 仅读取第一行（18,000个元素）

# 3. 计算最大值和最小值
min_val = np.min(first_row)
max_val = np.max(first_row)

print("第一个元素的最小值:", min_val)
print("第一个元素的最大值:", max_val)