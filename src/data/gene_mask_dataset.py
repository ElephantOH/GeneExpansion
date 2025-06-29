import os
import hydra
import torch
import scanpy as sc
import pandas as pd
from scipy import sparse
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningDataModule
from lightning_fabric.fabric import Fabric
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
import torch.nn.functional as F
from einops import rearrange

class GeneMaskDataModule(LightningDataModule):

    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        image_size: int = 384,
        pin_memory: bool = True,
        tokenier: str = "ncbi_bert",
        mask_ratio: float = 0.5,
        h5_paths: dict = {"train": "", "test": "", "val": ""},
        txt_paths: dict = {"train": "", "test": "", "val": ""},
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.h5_paths = h5_paths
        self.txt_paths = txt_paths
        self.tokenier = tokenier
        self.pin_memory = pin_memory
        self.mask_ratio = mask_ratio

        self.train_dataset = GeneMaskDataset(
            h5_path=self.txt_paths["train"],
            txt_path=self.txt_paths["train"],
            tokenier=self.tokenier,
            mask_ratio=self.mask_ratio,
        )

        if self.h5_paths["test"] is not "None" and self.txt_paths["test"] is not "None":
            self.test_dataset = GeneMaskDataset(
                h5_path=self.txt_paths["test"],
                txt_path=self.txt_paths["test"],
                tokenier=self.tokenier,
                mask_ratio=self.mask_ratio,
            )

        if self.h5_paths["val"] is not "None" and self.txt_paths["val"] is not "None":
            self.val_dataset = GeneMaskDataset(
                h5_path=self.txt_paths["val"],
                txt_path=self.txt_paths["val"],
                tokenier=self.tokenier,
                mask_ratio=self.mask_ratio,
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

    # 读数据集文件（h5文件和txt信息）
    def read_data(self, h5_path, txt_path):
        adata = sc.read_10x_h5(h5_path)
        obs_df = pd.read_csv(txt_path, sep="\t", index_col=0)
        if not obs_df.index.equals(adata.obs_names):
            obs_df = obs_df.reindex(adata.obs_names)
        adata.obs = obs_df.copy()
        adata.layers["counts"] = adata.X.copy()
        X_norm = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X']
        X_log = sc.pp.log1p(X_norm, copy=True)
        adata.layers["lognorm"] = X_log
        return adata

    def __init__(self,
                 h5_path,
                 txt_path,
                 tokenier,
                 mask_ratio=0.5):
        """
        基因掩码数据集
        """
        self.h5_path = h5_path
        self.txt_path = txt_path

        self.adata = self.read_data(self.h5_path, self.txt_path)
        print(f"[Info] Gene. Matrix Size: {self.get_gene_size()}")

        self.mask_ratio = mask_ratio
        self.tokenizer = BertTokenizer.from_pretrained(tokenier)

    def __len__(self):
        return self.adata.n_obs

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

    def __getitem__(self, idx):
        # 加载基因矩阵
        gene_matrix = torch.tensor(self.reshape_gene(self.adata, idx), dtype=torch.float32)
        gene_matrix = rearrange(gene_matrix, 'h w -> 1 h w')  # 添加通道维度

        # 生成随机掩码 (0=缺失, 1=已知)
        gene_mask = torch.ones_like(gene_matrix)
        gene_mask[:, int(gene_mask.shape[1] * (1 - self.mask_ratio)):] = 0  # 模拟底部区域的缺失

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

        return {
            "gene_matrix": gene_matrix,
            "gene_mask": gene_mask,
            "focus_mask": None,
            "text_input_ids": text_input.input_ids.squeeze(0),
            "text_attention_mask": text_input.attention_mask.squeeze(0),
            "text_desc": text_desc
        }