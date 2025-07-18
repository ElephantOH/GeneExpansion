import numpy as np
import os

# 配置路径
npy_path = "../../outputs/train10m/manifold_diffusion_model_norm/test7/cell_test7_manifold_ddim.npy"
output_dir = os.path.dirname(npy_path)
output_path = os.path.join(output_dir, "cell_test7_manifold_ddim_pro.npy")

# 计算内存占用 (确认可用)
dense = np.load(npy_path, mmap_mode='r')  # 内存映射而不是全量加载
print(f"数据尺寸: {dense.shape} | 数据类型: {dense.dtype}")
shape = dense.shape
dtype_size = np.float32().itemsize  # float32的字节大小
total_bytes = np.prod(shape) * dtype_size
print(f"原始数据内存占用: {total_bytes / (1024 ** 3):.2f} GB")
del dense

# 内存安全操作（虽然50G足够，但仍优化步骤）
def clamp_and_scale(data):
    """分段处理数据：Clamp → 缩放[0,1] → 缩放[0,6]"""
    # Step 1: Clamp到[-1, 1] (就地操作节省内存)
    np.clip(data, -1, 1, out=data)

    # Step 2: 从[-1,1]线性变换到[0,1] (公式：(x+1)/2)
    data += 1  # 就地加法
    data /= 2.0  # 就地除法

    # Step 3: 从[0,1]缩放至[0,6] (公式：x*6)
    data *= 6  # 就地乘法
    return data


# 处理主逻辑
def process_large_npy():
    # 内存映射方式加载大文件（避免立即占用全部内存）
    mmap_data = np.load(npy_path, mmap_mode='r')

    # 创建输出缓冲区（预分配连续内存）
    processed = np.empty_like(mmap_data, dtype=np.float32)

    # 分段处理（即使内存充足也推荐分批，避免峰值内存翻倍）
    chunk_size = 20000  # 根据系统调整块大小
    for i in range(0, shape[0], chunk_size):
        chunk = mmap_data[i:i + chunk_size]
        processed[i:i + chunk_size] = clamp_and_scale(chunk.copy())  # 复制块到内存处理

        # 显示进度
        print(f"处理进度: {min(i + chunk_size, shape[0])}/{shape[0]} 行")

    # 保存结果
    np.save(output_path, processed)
    print(f"处理完成！结果保存至: {output_path}")


if __name__ == "__main__":
    process_large_npy()