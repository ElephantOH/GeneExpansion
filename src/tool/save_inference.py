#!/usr/bin/env python3
"""
flatten_expression_batch.py

批量展平 180×180 空间转录组表达矩阵：
    1. 读取基因坐标表 (gene_coordinates_32400.csv)
    2. flatten_expression_batch()：将形状 (B,180,180) → (B,32400)
    3. save_to_single_file()：把所有 batch 追加保存到一个 flattened_all.npy
"""

from pathlib import Path
import numpy as np
import pandas as pd


# --------------------------- 核心函数 --------------------------- #
def flatten_expression_batch(expr_batch: np.ndarray, coords: pd.DataFrame) -> np.ndarray:
    """
    将形状 (B,180,180) 的表达矩阵展平成 (B,32400)。
    """
    if expr_batch.ndim != 3 or expr_batch.shape[1:] != (180, 180):
        raise ValueError("expr_batch 必须是形状 (B,180,180) 的 3D 数组")
    if len(coords) != 180 * 180:
        raise ValueError("coords 行数必须是 32400")

    x_idx = coords["x"].to_numpy()
    y_idx = coords["y"].to_numpy()
    flat = expr_batch[:, y_idx, x_idx]  # (B,32400)
    return flat


def save_to_single_file(flat_batch: np.ndarray, file_path: str) -> None:
    """
    追加保存 (B,32400) 到 *一个* .npy 文件。若文件不存在则直接创建。
    """
    file_path = Path(file_path)

    if file_path.exists():
        # ---------- 简单做法：读→拼→写 ----------
        existing = np.load(file_path)
        combined = np.concatenate([existing, flat_batch], axis=0)
        np.save(file_path, combined)

        # ---------- 如需更节省内存，请改用 open_memmap ----------
        # with np.load(file_path, mmap_mode='r') as existing:
        #     old_len = existing.shape[0]
        #     new_len = old_len + flat_batch.shape[0]
        #     dtype = existing.dtype
        #
        # # 创建或扩容 memmap（注意：文件必须重新写）
        # fp_tmp = file_path.with_suffix(".tmp")
        # mmp = np.lib.format.open_memmap(fp_tmp, mode='w+',
        #                                 dtype=dtype, shape=(new_len, flat_batch.shape[1]))
        # mmp[:old_len] = existing
        # mmp[old_len:] = flat_batch
        # mmp.flush()
        # fp_tmp.replace(file_path)  # 原子替换
    else:
        np.save(file_path, flat_batch)


# --------------------------- CLI / demo --------------------------- #
def save_inference(expr_batch):
    # 1. 读入基因坐标表
    coords = pd.read_csv(
        "/Users/tangqing/PycharmProjects/get_data/src/process_data/hvg_top/gene_coordinates_32400.csv"
    )  # 确保列名 gene,x,y

    # 2. 生成随机测试数据：例如一次产生 5 个样本
    # batches = 5
    # expr_batch = np.random.rand(batches, 180, 180).astype(np.float32)

    # 3. 展平
    flat_batch = flatten_expression_batch(expr_batch, coords)

    # 4. 追加保存到单个文件
    save_to_single_file(flat_batch, file_path="flattened_all.npy")

    # print(f"✓ 已将 {batches} 个 batch 追加到 'flattened_all.npy'")

