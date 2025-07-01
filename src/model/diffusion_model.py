import os

import torch
import torchvision
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from transformers import BertModel
import torch.nn.functional as F
from einops import rearrange

class DiffusionModel(LightningModule):
    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__()

        self.config = model_config
        self.save_hyperparameters(self.config)

        # 文本编码器 (冻结预训练权重)
        self.text_encoder = BertModel.from_pretrained(self.config.text_encoder)
        for param in self.text_encoder.parameters():
            param.requires_grad = self.config.train_text_encoder

        # 扩散模型UNet - 基于RePaint/Palette架构
        self.unet = UNet2DConditionModel(
            sample_size=self.config.sample_size,
            in_channels=self.config.in_channel,
            out_channels=self.config.out_channel,
            cross_attention_dim=self.config.cross_attention_dim,  # 文本向量维度
            layers_per_block=self.config.layers_per_block,
            block_out_channels=self.config.block_out_channels,
            down_block_types=self.config.down_block_types,
            up_block_types=self.config.up_block_types,
            norm_num_groups=self.config.norm_num_groups,
            time_embedding_type=self.config.time_embedding_type,
        )

        if self.config.use_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # 噪声调度器
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.ddpm_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.config.ddpm_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False,  # 禁用值裁剪
            set_alpha_to_one=False
        )

        self.dpm_solver = DPMSolverMultistepScheduler(
            num_train_timesteps=self.config.ddpm_timesteps,
            beta_schedule="linear",
            algorithm_type="dpmsolver++",  # 最佳算法
            solver_order=2,  # 平衡速度/质量
            prediction_type="epsilon",
            thresholding=False
        )


    def unmask_forward(self, noisy_genes, timesteps, text_emb, text_mask):
        # U-Net前向传播 (文本条件)
        return self.unet(
            noisy_genes,
            timesteps,
            encoder_hidden_states=text_emb,
            encoder_attention_mask=text_mask
        ).sample

    def training_step(self, batch, batch_idx, fabric):
        genes = fabric.to_device(batch["gene_matrix"])
        input_ids = fabric.to_device(batch["text_input_ids"])
        attention_mask = fabric.to_device(batch["text_attention_mask"])

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_emb = text_outputs.last_hidden_state

        noise = torch.randn_like(genes)

        # 时间步
        timesteps = torch.randint(
            0,
            self.ddpm_scheduler.config.num_train_timesteps,
            (genes.shape[0],),
            device=fabric.device  # 使用fabric.device
        )

        # 添加噪声（scheduler会自动处理设备）
        noisy_genes = self.ddpm_scheduler.add_noise(genes, noise, timesteps)

        # 模型预测
        pred_noise = self.unmask_forward(noisy_genes, timesteps, text_emb, attention_mask)

        # 损失计算
        return F.mse_loss(pred_noise, noise)


    def predict_x0(self, noisy_genes, pred_noise, timesteps):
        # 使用噪声调度器预测原始数据
        alpha_prod = self.ddpm_scheduler.alphas_cumprod[timesteps]
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

    def validation_step(self, batch, batch_idx, fabric, stage="test", solver="repaint_ddim"):
        # 使用RePaint推理
        if solver == "repaint_ddim":
            filled_genes = self.repaint_ddim_inference(
                fabric,
                batch["gene_matrix"],
                batch["gene_mask"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )
        elif solver == "unmask_ddim":
            filled_genes = self.unmask_ddim_inference(
                fabric,
                batch["gene_matrix"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )

        # 仅计算masked区域的精度
        total_mse = 0.0
        for i in range(batch["gene_matrix"].shape[0]):
            mse = F.mse_loss(
                filled_genes[i][batch["gene_mask"][i] < 0.5],
                batch["gene_matrix"][i][batch["gene_mask"][i] < 0.5]
            ).item()
            total_mse += mse
        total_mse /= batch["gene_matrix"].shape[0]

        # 可视化 - 保存对比图像到/tmp
        self.save_visualization(
            batch["gene_matrix"],
            batch["gene_mask"],
            filled_genes,
            batch_idx,
            stage
        )

        return {
            "mse": total_mse
        }

    def save_visualization(self, original, mask, filled, batch_idx, stage="test"):
        """保存对比图像：原始图像、mask后图像、修复结果"""
        # 确保目录存在
        os.makedirs("visual", exist_ok=True)

        # 获取batch中的第一个样本
        idx = 0
        orig_img = original[idx].squeeze().cpu()
        masked_img = (original * mask)[idx].squeeze().cpu()
        filled_img = filled[idx].squeeze().cpu()

        # 归一化到[0,1]范围
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        orig_img = normalize(orig_img)
        masked_img = normalize(masked_img)
        filled_img = normalize(filled_img)

        # 水平拼接三个图像
        combined = torch.cat([orig_img, masked_img, filled_img], dim=1)

        # 添加通道维度 (1, H, W)
        combined = combined.unsqueeze(0)

        # 保存图像
        filename = f"visual/{stage}_{self.current_epoch}_{batch_idx}.png"
        torchvision.utils.save_image(combined, filename)
        print(f"Saved Visualization to {filename}")

    def unmask_ddim_inference(self, fabric, original, text_ids, text_attn):
        """
        DDIM采样过程 - 适配Fabric多GPU环境

        参数：
        fabric: Fabric实例
        original: 参考输入 (仅用于形状，需由fabric.to_device处理)
        text_ids: 文本token ID (需由fabric.to_device处理)
        text_attn: 文本注意力掩码 (需由fabric.to_device处理)
        """
        # 确保使用fabric的设备
        device = fabric.device

        # 将输入数据转移到设备
        original = fabric.to_device(original)
        text_ids = fabric.to_device(text_ids)
        text_attn = fabric.to_device(text_attn)

        # 确保模型在评估模式
        self.eval()

        # 配置DDIM调度器
        scheduler = self.ddim_scheduler
        scheduler.set_timesteps(self.config.ddim_timesteps)
        timesteps = scheduler.timesteps.to(device)  # 确保时间步在正确设备

        # 文本编码
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=text_ids,
                attention_mask=text_attn
            )
            text_emb = text_outputs.last_hidden_state

        # 初始化噪声图像 - 基于输入批量大小
        batch_size = original.size(0)
        x_t = torch.randn(
            (batch_size, self.config.in_channel, self.config.sample_size, self.config.sample_size),
            device=device
        )

        # 主循环 - 使用no_backward_sync避免梯度计算
        with torch.inference_mode():
            for i, t in enumerate(timesteps):
                # 模型预测
                with torch.no_grad():
                    # 注意：t.expand()自动继承设备
                    noise_pred = self.unet(
                        x_t,
                        t.expand(x_t.size(0)),
                        encoder_hidden_states=text_emb,
                        encoder_attention_mask=text_attn
                    ).sample

                # DDIM更新步骤
                output = scheduler.step(
                    noise_pred,
                    t,
                    x_t,
                    eta=0.0  # 完全确定性采样
                )
                x_prev = output.prev_sample

                # 值裁剪确保在合理范围内
                x_prev = torch.clamp(x_prev, min=-1.0, max=1.0)

                # 更新当前状态
                x_t = x_prev

        return x_t


    def repaint_ddim_inference(self, fabric, original, mask, text_ids, text_attn, resample_steps=None):
        """
        DDIM采样过程 - 适配Fabric多GPU环境

        参数：
        fabric: Fabric实例
        original: 参考输入 (仅用于形状，需由fabric.to_device处理)
        text_ids: 文本token ID (需由fabric.to_device处理)
        text_attn: 文本注意力掩码 (需由fabric.to_device处理)
        """
        # 确保使用fabric的设备
        device = fabric.device

        # 将输入数据转移到设备
        original = fabric.to_device(original)
        mask = fabric.to_device(mask)
        text_ids = fabric.to_device(text_ids)
        text_attn = fabric.to_device(text_attn)

        # 确保模型在评估模式
        self.eval()

        # 配置DDIM调度器
        scheduler = self.ddim_scheduler
        scheduler.set_timesteps(self.config.ddim_timesteps)
        timesteps = scheduler.timesteps.to(device)  # 确保时间步在正确设备
        jump_length = self.config.ddpm_timesteps // self.config.ddim_timesteps  # 计算跳步长度

        # 文本编码
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=text_ids,
                attention_mask=text_attn
            )
            text_emb = text_outputs.last_hidden_state

        # 初始化噪声图像 - 基于输入批量大小
        batch_size = original.size(0)
        x_t = torch.randn_like(original, device=device)

        # 4. 设置重采样步数
        if resample_steps is None:
            resample_steps = scheduler.timesteps[::4].tolist()
        resample_steps = set(resample_steps)  # 转换为集合提高查找效率

        # 主循环 - 使用no_backward_sync避免梯度计算
        with torch.inference_mode():
            for i, t in enumerate(timesteps):
                # 模型预测
                with torch.no_grad():
                    # 注意：t.expand()自动继承设备
                    noise_pred = self.unet(
                        x_t,
                        t.expand(x_t.size(0)),
                        encoder_hidden_states=text_emb,
                        encoder_attention_mask=text_attn
                    ).sample

                prev_t = max((t.item() - jump_length), 0)  # 确保不越界

                # DDIM更新步骤
                output = scheduler.step(
                    noise_pred,
                    t,
                    x_t,
                    eta=0.0  # 完全确定性采样
                )
                x_prev = output.prev_sample

                # 值裁剪确保在合理范围内
                x_prev = torch.clamp(x_prev, min=-1.0, max=1.0)

                noisy_original_prev = self._get_noisy_image(original, prev_t)
                x_prev = mask * noisy_original_prev + (1 - mask) * x_prev

                if prev_t in resample_steps and prev_t > 0:
                    self.resample(original, mask, x_prev, prev_t, text_emb, text_attn, device)

                # 更新当前状态
                x_t = x_prev

        return x_t


    def resample(self, original, mask, x_prev, prev_t, text_emb, text_attn, device):
        for _ in range(self.config.resample_times):
            # 5.3.1 DDPM加噪回退
            x_next = self._ddpm_add_noise(x_prev, prev_t, prev_t + 1)

            # 5.3.2 DDPM正常去噪
            with torch.no_grad():
                noise_pred_next = self.unet(
                    x_next,
                    torch.tensor([prev_t + 1], device=device),
                    encoder_hidden_states=text_emb,
                    encoder_attention_mask=text_attn.to(device)
                ).sample

            # 执行DDPM去噪
            x_prev_denoised = self.ddpm_scheduler.step(
                noise_pred_next,
                prev_t + 1,
                x_next
            ).prev_sample

            x_prev_denoised = torch.clamp(x_prev_denoised, min=-1.0, max=1.0)

            # 5.3.3 Repaint区域替换
            noisy_original_prev = self._get_noisy_image(original, prev_t)
            x_prev_denoised = mask * noisy_original_prev + (1 - mask) * x_prev_denoised

            return x_prev_denoised

    def _get_noisy_image(self, original, t):
        """获取指定时间步的加噪真实图像"""
        noise = torch.randn_like(original, device=original.device)
        alpha_prod_t = self.ddpm_scheduler.alphas_cumprod[t]
        sqrt_alpha_prod = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_prod_t)

        return sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise

    def _ddpm_add_noise(self, x_prev, t_current, t_next):
        """执行DDPM加噪回退"""
        # 计算噪声系数
        noise = torch.randn_like(x_prev, device=x_prev.device)
        alpha_prod_t = self.ddpm_scheduler.alphas_cumprod[t_current]
        alpha_prod_t_next = self.ddpm_scheduler.alphas_cumprod[t_next]

        # 计算加噪系数
        coef1 = torch.sqrt(alpha_prod_t_next / alpha_prod_t)
        coef2 = torch.sqrt(1 - alpha_prod_t_next / alpha_prod_t)

        # 应用加噪公式
        return coef1 * x_prev + coef2 * noise
