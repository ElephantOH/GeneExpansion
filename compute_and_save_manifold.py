import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import joblib
from collections import defaultdict

from src.data.manifold_dataset import ManifoldDataset


@hydra.main(version_base=None, config_path="configs", config_name="compute_and_save_manifold.yaml")
def main(cfg: DictConfig):
    # 第一阶段：轻量模式快速收集cell_type信息
    print("开始轻量模式扫描cell_type分布...")
    lightweight_dataset = ManifoldDataset(
        h5ad_path=cfg.h5ad_path,
        mode="lightweight",
        device=cfg.device
    )

    # 使用更快的批量获取方式
    all_cell_types = lightweight_dataset.get_all_cell_types()
    cell_type_counts = defaultdict(int)
    for ct in tqdm(all_cell_types, desc="扫描cell_type分布"):
        cell_type_counts[ct] += 1

    print(f"发现 {len(cell_type_counts)} 种cell_type")

    # 第二阶段：仅创建一次完整模式数据集
    print("创建完整模式数据集...")
    full_dataset = ManifoldDataset(
        h5ad_path=cfg.h5ad_path,
        map_path=cfg.map_path,
        data_min=cfg.data_min,
        data_max=cfg.data_max,
        device=cfg.device,
        mode="full"
    )

    # 第三阶段：按cell_type处理
    os.makedirs(cfg.output_path, exist_ok=True)
    centers_dict = {}  # 用于存储所有聚类中心

    def dynamic_k(n_samples):
        # 动态计算聚类数量：最小100，最大500，基于样本数量
        if n_samples <= 100:
            return n_samples
        return max(100, min(500, int(n_samples * 0.05)))

    # 检查现有字典文件
    dict_file = os.path.join(cfg.output_path, "manifold_centers_dict.joblib")
    if os.path.exists(dict_file):
        print(f"加载现有字典文件: {dict_file}")
        centers_dict = joblib.load(dict_file)
    else:
        print("未找到现有字典文件，将创建新字典")
        centers_dict = {}

    # 确定缺失的细胞类型
    existing_keys = set(centers_dict.keys())
    missing_cell_types = [
        ct for ct in cell_type_counts.keys()
        if ct not in existing_keys or cfg.force_recompute
    ]

    if not missing_cell_types:
        print("\n所有细胞类型已存在字典中，无需处理。")
        return

    print("\n缺失的细胞类型:")
    for ct in missing_cell_types:
        count = cell_type_counts[ct]
        print(f"- {ct} (样本数: {count})")

    print(f"\n需要处理 {len(missing_cell_types)}/{len(cell_type_counts)} 个细胞类型")

    # 仅处理缺失的细胞类型
    for idx, cell_type in enumerate(tqdm(missing_cell_types, desc="处理缺失细胞类型", total=len(missing_cell_types))):
        count = cell_type_counts[cell_type]

        # 确定当前cell_type的样本索引
        indices = [i for i, ct in enumerate(all_cell_types) if ct == cell_type]

        # 创建当前cell_type的子集
        subset = torch.utils.data.Subset(full_dataset, indices)
        dataloader = DataLoader(
            dataset=subset,
            batch_size=1024,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )

        # 计算K值
        k = dynamic_k(count)
        print(f"\n处理 {cell_type}: 样本数={count}, 聚类中心数={k}")

        # 使用增量式K-Means
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(4096, count),
            compute_labels=False,
            random_state=42
        )

        # 分批处理
        for batch in tqdm(dataloader, desc=f"处理 {cell_type}", leave=False):
            sparse_codes = batch["sparse_code"].squeeze(1).numpy()
            kmeans.partial_fit(sparse_codes)

        # 获取中心点并添加到字典
        centers = kmeans.cluster_centers_
        centers_dict[cell_type] = centers
        print(f"生成的聚类中心形状: {centers.shape}")

        # 每隔几个cell_type保存一次作为检查点（可选）
        if cfg.checkpoint_interval > 0 and (idx + 1) % cfg.checkpoint_interval == 0:
            joblib.dump(centers_dict, dict_file)
            print(f"检查点保存: 已处理 {idx+1}/{len(missing_cell_types)} 个细胞类型")

    # 最终保存完整的字典
    joblib.dump(centers_dict, dict_file)
    print("\n所有细胞类型处理完成！")
    print(f"结果已保存到: {dict_file}")
    print(f"总共包含 {len(centers_dict)} 种细胞类型的聚类中心")



if __name__ == "__main__":
    main()