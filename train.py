import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from lightning_fabric.fabric import Fabric
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
import torch.nn.functional as F
from einops import rearrange

from src.tool.utils import calculate_model_params


@hydra.main(version_base=None, config_path="configs", config_name="gene_diffusion")
def main(cfg: DictConfig):
    # 初始化Fabric (用于分布式训练)
    fabric = Fabric(accelerator="auto", devices=cfg.hardware.devices)
    fabric.launch()

    # 设置随机种子
    seed_everything(cfg.training.seed)

    # 创建数据模块
    dataset = instantiate(cfg.dataset, _recursive_=False)

    train_dataloader = fabric.setup_dataloaders(dataset.train_dataloader())
    if cfg.is_val:
        val_dataloader = fabric.setup_dataloaders(dataset.val_dataloader())

    # 创建模型
    model = instantiate(cfg.model, config=cfg)
    calculate_model_params(model)

    # 日志设置
    logger = pl.loggers.WandbLogger(
        project=cfg.logging.project,
        save_dir=cfg.logging.dir,
        log_model=True
    )

    # 创建训练器
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.training.epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(cfg.logging.dir, "checkpoints"),
                monitor="val/mse",
                mode="min",
                save_top_k=3
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        enable_model_summary=True,
        deterministic=True
    )

    # 使用Fabric包装模型
    model, optimizer = fabric.setup(model, model.configure_optimizers()[0])

    # 训练模型
    trainer.fit(model, train_dataloader, val_dataloader)