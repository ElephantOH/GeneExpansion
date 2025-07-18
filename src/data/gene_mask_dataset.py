import os
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

class GeneMaskDataModule(LightningDataModule):

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
            self.train_dataset = GeneMaskDataset(
                h5ad_path=self.h5ad_paths["train"],
                map_path=self.map_paths["train"],
                tokenizer=self.tokenizer,
                mask_type=self.mask_type,
                mask_ratio=self.mask_ratio,
                data_min=self.data_min,
                data_max=self.data_max,
            )

        if self.h5ad_paths["test"] != "None":
            self.test_dataset = GeneMaskDataset(
                h5ad_path=self.h5ad_paths["test"],
                map_path=self.map_paths["test"],
                tokenizer=self.tokenizer,
                mask_type=self.mask_type,
                mask_ratio=self.mask_ratio,
                data_min=self.data_min,
                data_max=self.data_max,
            )

        if self.h5ad_paths["val"] != "None":
            self.val_dataset = GeneMaskDataset(
                h5ad_path=self.h5ad_paths["val"],
                map_path=self.map_paths["val"],
                tokenizer=self.tokenizer,
                mask_type=self.mask_type,
                mask_ratio=self.mask_ratio,
                data_min=self.data_min,
                data_max=self.data_max,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
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


class GeneMaskDataset(Dataset):

    # 读数据集文件
    def read_data(self, h5ad_path):
        adata = sc.read(h5ad_path)
        adata.layers["counts"] = adata.X.copy()
        X_norm = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X']
        X_log = sc.pp.log1p(X_norm, copy=True)
        adata.layers["lognorm"] = X_log
        return adata

    def __init__(self,
                 h5ad_path,
                 map_path,
                 tokenizer,
                 mask_type,
                 mask_ratio,
                 data_min,
                 data_max):
        """
        基因掩码数据集
        """
        self.h5ad_path = h5ad_path
        self.map_path = map_path
        self.adata = self.read_data(self.h5ad_path)
        print(f"[Info] Matrix Size: {self.get_gene_size()}")

        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.data_min = data_min
        self.data_max = data_max

        fix_file(os.path.join(tokenizer, "vocab.txt"))
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

        # 映射初始化
        self.df_map = pd.read_csv(self.map_path)  # 默认列名: gene, x, y
        self.coords = self.df_map[['x', 'y']].to_numpy()  # 32 400 × 2, 顺序与 gene 行一致

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        # 加载基因矩阵
        idx = idx + 1200
        gene_matrix = self.mapping_gene(self.reshape_gene(self.adata, idx))
        gene_matrix = torch.tensor(gene_matrix, dtype=torch.float32)
        gene_matrix = rearrange(gene_matrix, 'h w -> 1 h w')  # 添加通道维度

        gene_matrix = torch.clamp(gene_matrix, min=self.data_min, max=self.data_max)
        resize_value = (self.data_max - self.data_min) / 2.
        gene_matrix = (gene_matrix - resize_value) / resize_value

        # 生成随机掩码 (0=缺失, 1=已知)
        gene_mask = self.get_mask(gene_matrix)

        # 提取文本描述
        cell_type = self.adata.obs.iloc[idx]["cell_type"]
        text_desc = f"Cell type: {cell_type}."
        # text_desc += f"Expressed genes: {', '.join(meta_data['marker_genes'][:3])}"

        # 文本编码
        text_input = self.tokenizer(
            text_desc,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )

        return_dict = {
            'indices': idx,
            "gene_matrix": gene_matrix,
            "gene_mask": gene_mask,
            "focus_mask": gene_mask,
            "text_input_ids": text_input.input_ids.squeeze(0),
            "text_attention_mask": text_input.attention_mask.squeeze(0),
            "cell_type": cell_type
        }

        return return_dict

    def get_gene_size(self):
        x_row = self.adata.layers["lognorm"][0, :]
        if sparse.issparse(x_row):
            x_row = x_row.toarray().flatten()
        else:
            x_row = np.array(x_row).flatten()
        n = len(x_row)
        new_size = int(np.ceil(np.sqrt(n)))
        return new_size


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