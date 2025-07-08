import os

import h5py
import torch
import scanpy as sc
import pandas as pd
from scipy import sparse
import numpy as np
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

def fix_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            head = f.read(3)
            has_bom = head == b'\xef\xbb\xbf'

        if has_bom or not head:
            with open(file_path, 'r', encoding='utf-8-sig' if has_bom else 'utf-8') as f:
                content = f.read()

            with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)

            action = "BOM removed" if has_bom else "LF normalized"
            print(f"Fixed: {file_path} ({action})")
        else:
            print(f"Skip: {file_path} (no BOM needed)")

    except UnicodeDecodeError:
        import os
        new_path = os.path.splitext(file_path)[0] + '.rom'
        os.rename(file_path, new_path)
        print(f"Converted to ROM: {new_path}")


class MultiFileGeneMaskDataModule(LightningDataModule):

    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        image_size: int = 184,
        pin_memory: bool = True,
        tokenizer: str = "ncbi_bert",
        mask_type: str = "true_mask",
        mask_ratio: float = 0.5,
        data_min: float = 0.0,
        data_max: float = 6.0,
        h5ad_paths: dict = {"train": "", "test": "", "val": ""},
        map_paths: dict = {"train": "", "test": "", "val": ""},
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.h5ad_paths = h5ad_paths
        self.map_paths = map_paths
        self.tokenizer = tokenizer
        self.pin_memory = pin_memory
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.data_min = data_min
        self.data_max = data_max

        if self.h5ad_paths["train"] != "None":
            self.train_dataset = MultiFileGeneMaskDataset(
                h5ad_paths=self.h5ad_paths["train"],
                map_path=self.map_paths["train"],
                tokenizer=self.tokenizer,
                mask_type=self.mask_type,
                mask_ratio=self.mask_ratio,
                data_min=self.data_min,
                data_max=self.data_max,
                shuffle=True,
                stage="train",
            )

        if self.h5ad_paths["test"] != "None":
            self.test_dataset = MultiFileGeneMaskDataset(
                h5ad_paths=self.h5ad_paths["test"],
                map_path=self.map_paths["test"],
                tokenizer=self.tokenizer,
                mask_type=self.mask_type,
                mask_ratio=self.mask_ratio,
                data_min=self.data_min,
                data_max=self.data_max,
                shuffle=False,
                stage="test",
            )

        if self.h5ad_paths["val"] != "None":
            self.val_dataset = MultiFileGeneMaskDataset(
                h5ad_paths=self.h5ad_paths["val"],
                map_path=self.map_paths["val"],
                tokenizer=self.tokenizer,
                mask_type=self.mask_type,
                mask_ratio=self.mask_ratio,
                data_min=self.data_min,
                data_max=self.data_max,
                shuffle=False,
                stage="val",
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class MultiFileGeneMaskDataset(Dataset):
    def __init__(self,
                 h5ad_paths,  # 修改为路径列表
                 map_path,
                 tokenizer,
                 mask_type,
                 mask_ratio,
                 data_min,
                 data_max,
                 shuffle=False,
                 seed=420,
                 stage="train"):
        """
        支持多文件按需加载的基因掩码数据集
        """
        self.h5ad_paths = h5ad_paths
        self.map_path = map_path
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.data_min = data_min
        self.data_max = data_max
        self.stage = stage

        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 用h5py快速获取文件样本数(不加载完整数据)
        self.cumulative_counts = [0]  # 累积样本数
        self.file_sample_counts = []  # 各文件样本数
        self.file_permutations = {}

        for path in h5ad_paths:
            adata = sc.read(path, backed="r")
            n_obs = adata.n_obs
            adata.file.close()
            self.file_sample_counts.append(n_obs)
            self.cumulative_counts.append(self.cumulative_counts[-1] + n_obs)

            if self.shuffle:
                self.file_permutations[path] = self.rng.permutation(n_obs)
            else:
                self.file_permutations[path] = np.arange(n_obs)

        # 当前文件状态
        self.current_file_index = -1  # 当前加载的文件索引
        self.current_adata = None  # 当前加载的adata

        # 文本编码器初始化
        fix_file(os.path.join(tokenizer, "vocab.txt"))
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

        # 基因位置映射
        self.df_map = pd.read_csv(map_path)
        self.coords = self.df_map[['x', 'y']].to_numpy()

        print(f"[Info] Total samples: {len(self)} | Files: {len(h5ad_paths)}")
        print(f"[Info] File shuffle: {'ON' if self.shuffle else 'OFF'} | Seed: {seed}")

    def __len__(self):
        return self.cumulative_counts[-1]  # 返回总样本数


    def __getitem__(self, idx):
        # 确定文件索引和文件内索引
        file_idx, in_file_idx = self.locate_sample(idx)

        # 按需加载新文件
        if file_idx != self.current_file_index:
            self.load_file(file_idx)

        # 获取当前样本在文件内的真实索引（应用shuffle映射）
        path = self.h5ad_paths[file_idx]
        true_idx = self.file_permutations[path][in_file_idx]

        # 获取当前样本数据
        return self.process_sample(true_idx)


    def locate_sample(self, global_idx):
        """将全局索引转换为(文件索引, 文件内索引)"""
        global_idx = int(global_idx)
        for i in range(len(self.cumulative_counts) - 1):
            if global_idx < self.cumulative_counts[i + 1]:
                return i, global_idx - self.cumulative_counts[i]
        raise IndexError(f"Index {global_idx} out of range")


    def load_file(self, file_idx):
        """加载指定索引的文件并释放前一文件"""
        # 释放现有资源
        if self.current_adata is not None:
            del self.current_adata
            self.current_adata = None

        # 加载新文件
        path = self.h5ad_paths[file_idx]
        self.current_adata = self.read_data(path)
        self.current_file_index = file_idx

        # 自动重置当前文件的shuffle
        if self.shuffle:
            # 使用当前epoch和文件路径创建唯一种子
            file_seed = hash((self.seed, path)) % (2 ** 32)
            file_rng = np.random.default_rng(file_seed)
            n = self.file_sample_counts[file_idx]
            self.file_permutations[path] = file_rng.permutation(n)
            print(f"[Info] Loaded (and shuffled) file {file_idx + 1}/{len(self.h5ad_paths)}: {path}")
        else:
            print(f"[Info] Loaded (NO shuffled) file {file_idx + 1}/{len(self.h5ad_paths)}: {self.h5ad_paths[file_idx]}")

    def reset_shuffle(self, seed=None):
        """重新生成所有文件的随机索引（用于epoch切换）"""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        for path in self.h5ad_paths:
            n = next((c for p, c in zip(self.h5ad_paths, self.file_sample_counts) if p == path), 0)
            self.file_permutations[path] = self.rng.permutation(n) if self.shuffle else np.arange(n)
        print(f"[Info] Shuffle sequences reset with seed: {self.seed}")


    def read_data(self, path):
        """读取并预处理单个h5ad文件"""
        adata = sc.read(path)
        adata.layers["counts"] = adata.X.copy()
        X_norm = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X']
        X_log = sc.pp.log1p(X_norm, copy=True)
        adata.layers["lognorm"] = X_log
        return adata


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


    def get_mask(self, gene_matrix, layer_threshold=74):
        if self.mask_type == "random":
            gene_mask = torch.bernoulli(torch.full_like(gene_matrix, self.mask_ratio))
        elif self.mask_type == "true_mask":
            # 获取张量形状和设备
            B, H, W = gene_matrix.shape  # 通常为 [1, 180, 180]
            device = gene_matrix.device

            # 创建距离中心点的坐标网格
            center_x = (W - 1) / 2.0
            center_y = (H - 1) / 2.0
            y_coords = torch.arange(H, device=device, dtype=torch.float32)
            x_coords = torch.arange(W, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # 计算每个点到中心点的绝对坐标差
            dx = torch.abs(grid_x - center_x)
            dy = torch.abs(grid_y - center_y)

            # 计算切比雪夫距离 (取 dx 和 dy 的最大值)
            chebyshev_dist = torch.max(dx, dy)

            # 计算层数: 向下取整，得到整数层索引
            layers = torch.floor(chebyshev_dist).long()

            # 创建空掩码 (与输入同形状)
            gene_mask = torch.zeros_like(gene_matrix)

            # 处理内层 (0-74层): 高遮盖区域
            inner_region = (layers <= layer_threshold)
            # 在内层区域按比例生成 0
            inner_mask = 1 - torch.bernoulli(
                torch.full((B, H, W), self.mask_ratio, device=device)
            )

            gene_mask[inner_region.unsqueeze(0)] = inner_mask[inner_region.unsqueeze(0)]  # 仅应用在内层区域

            # 处理外层 (75-89层): 高保留区域
            outer_region = (layers > layer_threshold)
            # 在外层区域生成高比例 1
            outer_mask = torch.bernoulli(
                torch.full((B, H, W), self.mask_ratio, device=device)
            )
            gene_mask[outer_region.unsqueeze(0)] = outer_mask[outer_region.unsqueeze(0)]  # 仅应用在外层区域

        else:
            gene_mask = torch.ones_like(gene_matrix)
        return gene_mask


    def get_sparse_code(self, gene_matrix):
        if gene_matrix.dim() == 4:
            # 压缩单通道维度 (C=1)
            gene_matrix = gene_matrix.squeeze(1)
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
        sparse_code = torch.where(
            layer_counts > 0,
            layer_sums / layer_counts,
            torch.zeros(1, device=device)
        )

        return sparse_code

    def process_sample(self, idx):
        """处理单个样本"""
        # 加载基因矩阵
        gene_matrix = self.mapping_gene(self.reshape_gene(self.current_adata, idx))
        gene_matrix = torch.tensor(gene_matrix, dtype=torch.float32)
        gene_matrix = rearrange(gene_matrix, 'h w -> 1 h w')  # 添加通道维度

        # 数据归一化
        gene_matrix = torch.clamp(gene_matrix, min=self.data_min, max=self.data_max)
        resize_value = (self.data_max - self.data_min) / 2.
        gene_matrix = (gene_matrix - resize_value) / resize_value

        # 提取文本描述
        cell_type = self.current_adata.obs.iloc[idx]["cell_type"]
        text_desc = f"Cell type: {cell_type}."

        # 文本编码
        text_input = self.tokenizer(
            text_desc,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )

        if self.stage == "train":
            return {
                "gene_matrix": gene_matrix,
                "text_input_ids": text_input.input_ids.squeeze(0),
                "text_attention_mask": text_input.attention_mask.squeeze(0),
                "cell_type": cell_type
            }

        # 生成随机掩码
        gene_mask = self.get_mask(gene_matrix)
        sparse_code = self.get_sparse_code(gene_matrix)

        return {
            "gene_matrix": gene_matrix,
            "gene_mask": gene_mask,
            "sparse_code": sparse_code,
            "text_input_ids": text_input.input_ids.squeeze(0),
            "text_attention_mask": text_input.attention_mask.squeeze(0),
            "cell_type": cell_type
        }