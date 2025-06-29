import os
from typing import Union, Optional

import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from lightning_fabric.fabric import Fabric
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from einops import rearrange


class SpatialGuidedUNet(UNet2DConditionModel):
    """添加空间注意力引导的改进UNet"""

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_weights: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        # 原始UNet前向传播...

        # 新增：空间注意力引导
        for resnet, attn in zip(self.down_blocks, self.attentions):
            # 应用空间注意力增强
            sample = self.apply_spatial_guidance(sample, attn, attention_mask, attention_weights)

        # 中间块处理...

        for resnet, attn in zip(self.up_blocks, self.attentions):
            # 应用空间注意力增强
            sample = self.apply_spatial_guidance(sample, attn, attention_mask, attention_weights)

        return sample

    def apply_spatial_guidance(self, x, attn_layer, mask, weights):
        """应用空间注意力引导"""
        # 1. 计算原始注意力
        orig_attn = attn_layer(x)

        # 2. 生成注意力引导
        guided_attn = torch.sigmoid(mask) * weights

        # 3. 融合引导注意力 (混合系数可学习)
        alpha = self.attention_alpha  # 可学习参数
        new_attn = (1 - alpha) * orig_attn + alpha * guided_attn

        return new_attn

class SpatialDiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        # 文本编码器 (冻结预训练权重)
        self.text_encoder = BertModel.from_pretrained("ncbi_bert")
        for param in self.text_encoder.parameters():
            param.requires_grad = self.hparams.train_text_encoder

        # 扩散模型UNet - 基于RePaint/Palette架构
        self.unet = UNet2DConditionModel(
            sample_size=config.image_size,
            in_channels=1,  # 单通道基因"图像"
            out_channels=1,
            cross_attention_dim=768,  # 文本向量维度
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 1024),
            norm_num_groups=8,
            time_embedding_type="positional",
        )

        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

        # 自定义损失权重
        self.gene_corr_loss_weight = config.loss.gene_corr_weight

    def forward(self, noisy_genes, timesteps, mask, text_emb, text_mask):
        # 拼接条件输入 (基因矩阵 + 掩码 + 文本特征)
        model_input = torch.cat([noisy_genes, mask], dim=1)

        # U-Net前向传播 (文本条件)
        return self.unet(
            model_input,
            timesteps,
            encoder_hidden_states=text_emb,
            encoder_attention_mask=text_mask
        ).sample

    def training_step(self, batch, batch_idx):
        # 准备数据
        clean_genes = batch["gene_matrix"]
        mask = batch["mask"]

        # 提取文本特征
        text_outputs = self.text_encoder(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"]
        )
        text_emb = text_outputs.last_hidden_state

        # 添加噪声
        noise = torch.randn_like(clean_genes)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (clean_genes.shape[0],), device=self.device
        )

        noisy_genes = self.noise_scheduler.add_noise(clean_genes, noise, timesteps)

        # 模型预测
        pred_noise = self.forward(noisy_genes, timesteps, mask, text_emb, batch["text_attention_mask"])

        # 基础MSE损失
        mse_loss = F.mse_loss(pred_noise, noise)

        # 基因相关性损失 (仅对预测值计算)
        pred_genes = self.predict_x0(noisy_genes, pred_noise, timesteps)
        gene_corr_loss = self.calculate_gene_correlation_loss(pred_genes, clean_genes, mask)

        # 总损失
        total_loss = mse_loss + self.gene_corr_loss_weight * gene_corr_loss

        # 记录指标
        self.log("train/mse_loss", mse_loss, prog_bar=True)
        self.log("train/gene_corr_loss", gene_corr_loss)
        self.log("train/total_loss", total_loss)

        return total_loss

    def predict_x0(self, noisy_genes, pred_noise, timesteps):
        # 使用噪声调度器预测原始数据
        alpha_prod = self.noise_scheduler.alphas_cumprod[timesteps]
        alpha_prod = alpha_prod.view(-1, 1, 1, 1).to(noisy_genes.device)

        pred_x0 = (noisy_genes - torch.sqrt(1 - alpha_prod) * pred_noise) / torch.sqrt(alpha_prod)
        return pred_x0

    def calculate_gene_correlation_loss(self, pred, target, mask):
        # 仅对masked区域计算
        masked_pred = torch.where(mask < 0.5, pred, torch.zeros_like(pred))
        masked_target = torch.where(mask < 0.5, target, torch.zeros_like(target))

        # 计算批次内基因间相关性
        pred_flat = rearrange(masked_pred, 'b c h w -> b (h w)')
        target_flat = rearrange(masked_target, 'b c h w -> b (h w)')

        # 皮尔逊相关系数
        pred_corr = torch.corrcoef(pred_flat.T)
        target_corr = torch.corrcoef(target_flat.T)

        # 对称KL散度作为相关性差异
        eps = 1e-8
        kl1 = F.kl_div(
            (pred_corr + eps).log(),
            target_corr,
            reduction='batchmean'
        )
        kl2 = F.kl_div(
            (target_corr + eps).log(),
            pred_corr,
            reduction='batchmean'
        )
        return (kl1 + kl2) / 2

    def validation_step(self, batch, batch_idx):
        # 验证阶段使用完整推理流程
        filled_genes = self.repaint_inference(
            batch["gene_matrix"],
            batch["mask"],
            batch["text_input_ids"],
            batch["text_attention_mask"],
            resample_steps=[900, 800, 700, 600, 500]
        )

        # 仅计算masked区域的精度
        mse = F.mse_loss(
            filled_genes[batch["mask"] < 0.5],
            batch["gene_matrix"][batch["mask"] < 0.5]
        )
        self.log("val/mse", mse, prog_bar=True)

        # 采样保存用于可视化
        if batch_idx == 0:
            self.logger.experiment.add_image(
                "Original",
                batch["gene_matrix"][0].squeeze(),
                dataformats="HW"
            )
            self.logger.experiment.add_image(
                "Masked",
                (batch["gene_matrix"] * batch["mask"])[0].squeeze(),
                dataformats="HW"
            )
            self.logger.experiment.add_image(
                "Filled",
                filled_genes[0].squeeze(),
                dataformats="HW"
            )

    def repaint_inference(self, original, mask, text_ids, text_attn, resample_steps):
        """RePaint风格推理流程"""
        device = self.device
        self.to(device)
        self.eval()

        # 文本编码
        text_outputs = self.text_encoder(
            input_ids=text_ids.to(device),
            attention_mask=text_attn.to(device)
        )
        text_emb = text_outputs.last_hidden_state

        # 初始噪声
        noisy_genes = torch.randn_like(original).to(device)

        # 复制原始数据
        current = original.clone().to(device)
        mask = mask.to(device)

        # 扩散步长 (反向)
        timesteps = list(reversed(range(len(self.noise_scheduler))))

        # RePaint推理循环
        for i, t in enumerate(timesteps):
            # 为当前时间步创建张量
            timestep = torch.tensor([t] * original.size(0), device=device)

            # 模型预测噪声
            with torch.no_grad():
                noise_pred = self.forward(
                    noisy_genes,
                    timestep,
                    mask,
                    text_emb,
                    text_attn.to(device)
                )

            # 更新噪声图像 (DDIM更新规则)
            noisy_genes = self.noise_scheduler.step(
                noise_pred, t, noisy_genes, eta=0.0
            ).prev_sample

            # RePaint重采样步骤
            if t in resample_steps:
                # 在mask区域重新注入噪声
                noise = torch.randn_like(noisy_genes)
                noisy_genes = torch.where(
                    mask > 0.5,
                    noisy_genes,  # 保持已知区域
                    self.noise_scheduler.add_noise(
                        original.to(device),
                        noise,
                        torch.tensor([t], device=device)
                    )  # 对mask区域重新噪声化
                )

        # 返回最终预测
        return noisy_genes.detach().cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.unet.parameters()},
                {"params": self.text_encoder.parameters() if self.hparams.train_text_encoder else []}
            ],
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )

        return [optimizer], [scheduler]