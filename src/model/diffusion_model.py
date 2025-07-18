import os

import numpy as np
import pandas as pd
import torch
import torchvision
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from transformers import BertModel
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torch.utils.checkpoint import checkpoint
from src.tool.save_inference import flatten_expression_batch, save_to_single_file


class DiffusionModel(LightningModule):
    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__()

        self.config = model_config
        self.epoch = 0
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

        self.scheduler = None

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

        self.dpm_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.config.ddpm_timesteps,
            beta_schedule="linear",
            algorithm_type="dpmsolver++",  # 最佳算法
            solver_order=2,  # 平衡速度/质量
            prediction_type="epsilon",
            thresholding=False
        )

        self.map_csv = pd.read_csv(self.config.map_path)
        self.result_path = self.config.result_path

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

    def validation_step(self, batch, batch_idx, fabric, stage="test", solver="repaint_ddim"):
        # 使用RePaint推理
        if solver == "inpaint_repaint":
            filled_genes = self.repaint_ddim_inference(
                fabric,
                batch["gene_matrix"],
                batch["gene_mask"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )
        elif solver == "inpaint_ddnm":
            filled_genes = self.inpaint_ddnm_inference(
                fabric,
                batch["gene_matrix"],
                batch["gene_mask"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )
        elif solver == "inpaint_dps":
            filled_genes = self.inpaint_dps_inference(
                fabric,
                batch["gene_matrix"],
                batch["gene_mask"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )
        elif solver == "inpaint_spgd":
            filled_genes = self.inpaint_spgd_inference(
                fabric,
                batch["gene_matrix"],
                batch["gene_mask"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )
        elif solver == "inpaint_ddim":
            filled_genes = self.inpaint_ddim_inference(
                fabric,
                batch["gene_matrix"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )

        # 仅计算masked区域的精度
        total_mse = 0.0
        total_pcc = 0.0
        total_cossim = 0.0
        for i in range(batch["gene_matrix"].shape[0]):
            # 获取masked区域的数据
            mask = batch["gene_mask"][i] < 0.5
            pred = filled_genes[i][mask].view(-1)
            target = batch["gene_matrix"][i][mask].view(-1)

            # 计算MSE
            mse = F.mse_loss(pred, target).item()
            total_mse += mse

            # 计算皮尔逊相关系数(PCC)
            if pred.nelement() > 1:  # 避免当masked区域过小时的计算错误
                std_pred, std_target = torch.std(pred), torch.std(target)
                if std_pred > 1e-8 and std_target > 1e-8:
                    cov = torch.mean((pred - torch.mean(pred)) * (target - torch.mean(target)))
                    pcc = cov / (std_pred * std_target)
                    total_pcc += pcc.item()

            # 计算余弦相似度(COSSIM)
            # 确保向量非零
            if torch.norm(pred) > 1e-8 and torch.norm(target) > 1e-8:
                pred_normalized = pred / torch.norm(pred)
                target_normalized = target / torch.norm(target)
                cossim = torch.dot(pred_normalized, target_normalized).item()
                total_cossim += cossim

        # 取批次平均值
        count = batch["gene_matrix"].shape[0]
        total_mse /= count
        total_pcc /= count
        total_cossim /= count

        # 计算整个图像的SSIM
        total_ssim = 0.0
        for i in range(batch["gene_matrix"].shape[0]):
            img1 = filled_genes[i:i + 1]  # 保持[b, 1, 180, 180]维度
            img2 = batch["gene_matrix"][i:i + 1]

            # 动态计算图像数据范围
            min_val = min(torch.min(img1).item(), torch.min(img2).item())
            max_val = max(torch.max(img1).item(), torch.max(img2).item())
            data_range = max_val - min_val

            # 处理常数图像的情况
            if data_range < 1e-6:
                data_range = 1.0  # 避免除零错误

            ssim_val = ssim(img1, img2, data_range=data_range)
            total_ssim += ssim_val.item()
        total_ssim /= batch["gene_matrix"].shape[0]

        # 可视化 - 保存对比图像到/tmp
        if batch_idx == 0:
            self.save_visualization(
                batch["gene_matrix"],
                batch["gene_mask"],
                filled_genes,
                batch_idx,
                stage
            )

        indices = batch['indices']  # 获取批次中每个样本的全局索引
        filled_genes_gathered = fabric.all_gather(filled_genes)  # [所有GPU, B, C, H, W]
        indices_gathered = fabric.all_gather(indices)  # [所有GPU, B]

        if fabric.global_rank == 0:
            # 展平多GPU收集的结果
            filled_genes_flat = filled_genes_gathered.flatten(start_dim=0, end_dim=1)
            indices_flat = indices_gathered.flatten()

            # 按原始索引排序
            sorted_indices = torch.argsort(indices_flat)
            sorted_genes = filled_genes_flat[sorted_indices]

            # 保存排序后的结果
            # self.save_results(sorted_genes)

        return {
            "mse": total_mse,
            "pcc": total_pcc,
            "cossim": total_cossim,
            "ssim": total_ssim,
        }

    def save_visualization(self, original, mask, filled, batch_idx, stage="test"):

        os.makedirs(f"visual/{stage}", exist_ok=True)

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
        filename = f"visual/{stage}/{self.epoch}_{batch_idx}.png"
        self.epoch = self.epoch + 1
        torchvision.utils.save_image(combined.cpu().float(), filename)
        print(f"Saved Visualization to {filename}")


    def save_results(self, gene_matrix, id=0):
        # input [b, 1, 180, 180]
        os.makedirs(self.result_path, exist_ok=True)
        gene_matrix = torch.clamp(gene_matrix, min=-1, max=1)
        gene_matrix = (gene_matrix + 1.) * 3.
        gene_matrix = gene_matrix.squeeze(1).cpu().numpy().astype(np.float32)
        flat_batch = flatten_expression_batch(gene_matrix, self.map_csv)
        file_name = f"cell_{self.config.test_data_name}_{self.config.solver_type}.npy"
        save_to_single_file(flat_batch, file_path=os.path.join(self.result_path, file_name))
        print(f"✓已将 {gene_matrix.shape[0]} 个 batch 追加到 '{file_name}'")


    def inpaint_ddnm_inference(self, fabric, original, mask, text_ids, text_attn):
        device = fabric.device
        original = fabric.to_device(original)
        mask = fabric.to_device(mask)
        text_ids = fabric.to_device(text_ids)
        text_attn = fabric.to_device(text_attn)
        self.eval()

        # 配置DDIM调度器
        scheduler = self.ddim_scheduler
        scheduler.set_timesteps(self.config.ddim_timesteps)
        timesteps = scheduler.timesteps.to(device)

        # DDNM关键算子
        A_dagger = mask.clone()
        A_dagger_A = A_dagger * mask  # A†A运算

        # 文本编码
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=text_ids,
                attention_mask=text_attn
            )
            text_emb = text_outputs.last_hidden_state

        # 初始化噪声图像
        x_t = torch.randn_like(original, device=device)

        # DDNM主循环
        with torch.inference_mode():
            for i, t in enumerate(timesteps):
                # 1. 预测当前噪声
                noise_pred = self.unet(
                    x_t,
                    t.expand(x_t.size(0)),
                    encoder_hidden_states=text_emb,
                    encoder_attention_mask=text_attn
                ).sample

                # 2. 计算x₀|t（公式12）
                alpha_bar_t = self.ddim_scheduler.alphas_cumprod[t]
                # 防止除零错误
                alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-6)

                x0_t = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

                # 3. DDNM核心：零空间修正（公式13）
                x0_t_hat = A_dagger * original + (1 - A_dagger_A) * x0_t

                # 4. 重新计算噪声（基于修正的x0_t_hat）
                recalc_noise = (x_t - torch.sqrt(alpha_bar_t) * x0_t_hat) / torch.sqrt(1 - alpha_bar_t)

                # 5. 计算下一时间步（公式14）
                # 获取前一时间步索引
                prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0, device=device)

                # 获取累积alpha值
                alpha_bar_prev = (
                    self.ddim_scheduler.alphas_cumprod[prev_t]
                    if prev_t > 0
                    else torch.tensor(1.0, device=device)
                )
                alpha_bar_prev = torch.clamp(alpha_bar_prev, min=1e-6)

                # 计算x_{t-1}
                coef1 = torch.sqrt(alpha_bar_prev)
                coef2 = torch.sqrt(1 - alpha_bar_prev)
                x_t = coef1 * x0_t_hat + coef2 * recalc_noise

        return x_t

    def inpaint_dps_inference(self, fabric, original, mask, text_ids, text_attn):
        # 设备转移统一处理
        device = fabric.device
        original = fabric.to_device(original)
        mask = fabric.to_device(mask)
        text_ids = fabric.to_device(text_ids)
        text_attn = fabric.to_device(text_attn)
        self.eval()

        # 调度器设置
        scheduler = self.ddim_scheduler
        scheduler.set_timesteps(self.config.ddim_timesteps)
        alphas_cumprod = scheduler.alphas_cumprod.to(device)  # 关键！
        timesteps = scheduler.timesteps.to(device)
        jump_length = self.config.ddpm_timesteps // self.config.ddim_timesteps

        # 文本编码
        with torch.no_grad():
            text_outputs = self.text_encoder(text_ids, attention_mask=text_attn)
            text_emb = text_outputs.last_hidden_state

        # 初始化
        x_t = torch.randn_like(original, device=device)
        known_region = mask * original

        # DPS主循环
        with torch.no_grad():  # 替换inference_mode
            for i, t in enumerate(timesteps):
                # 噪声预测
                with torch.no_grad():
                    noise_pred = self.unet(
                        x_t,
                        t.expand(x_t.size(0)),
                        encoder_hidden_states=text_emb,
                        encoder_attention_mask=text_attn
                    ).sample

                # 后验均值估算 (公式10)
                alpha_bar_t = alphas_cumprod[t]
                hat_x_0 = (x_t - (1 - alpha_bar_t).sqrt() * noise_pred) / (alpha_bar_t.sqrt() + 1e-8)

                # 隔离梯度计算
                with torch.enable_grad():
                    # 重建计算图(仅x_t相关部分)
                    x_t_grad = x_t.detach().requires_grad_(True)
                    hat_x0_grad = (x_t_grad - (1 - alpha_bar_t).sqrt() * noise_pred) / (alpha_bar_t.sqrt() + 1e-8)
                    residual_grad = known_region - mask * hat_x0_grad
                    loss_grad = torch.norm(residual_grad, p=2) ** 2
                    grad = torch.autograd.grad(loss_grad, x_t_grad)[0]
                    grad.detach_()  # 分离梯度

                # DDIM更新（手动实现）
                t_prev = max(t - jump_length, 0)
                alpha_bar_prev = alphas_cumprod[t_prev]
                x_prev_uncond = (alpha_bar_prev.sqrt() * hat_x_0 +
                                 (1 - alpha_bar_prev).sqrt() * noise_pred)

                # DPS梯度更新
                residual_norm = torch.norm(known_region - mask * hat_x_0, p=2)
                zeta = 0.5 / (residual_norm + 1e-8)  # ζ' = 0.5
                x_prev = x_prev_uncond - zeta * grad

                x_t = x_prev  # 更新状态

        return x_t

    def inpaint_spgd_inference(self, fabric, original, mask, text_ids, text_attn):
        device = fabric.device
        original = fabric.to_device(original)
        mask = fabric.to_device(mask)
        text_ids = fabric.to_device(text_ids)
        text_attn = fabric.to_device(text_attn)
        self.eval()

        # ============== SPGD 参数 ==============
        N = 5  # 预热迭代次数
        zeta = 2.5  # 总学习率（inpainting任务）
        beta = 0.95  # ADM动量系数
        eps = 1e-8  # 数值稳定因子

        scheduler = self.ddim_scheduler
        scheduler.set_timesteps(self.config.ddim_timesteps)
        timesteps = scheduler.timesteps.to(device)
        jump_length = self.config.ddpm_timesteps // self.config.ddim_timesteps

        # 文本编码
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=text_ids, attention_mask=text_attn
            )
            text_emb = text_outputs.last_hidden_state

        # 初始化噪声图像
        x_t = torch.randn_like(original, device=device)

        # ============== 主循环（修复版） ==============
        with torch.no_grad():  # 改用no_grad允许局部梯度
            for i, t_index in enumerate(timesteps):
                current_timestep = timesteps[i]  # 获取当前时间步

                if i % 1 == 0:
                    # ====== 1. 预热循环 ======
                    x_t_j = x_t.clone()
                    adm_grad = torch.zeros_like(x_t_j)  # 设备安全初始化
                    for j in range(N):
                        # 启用局部梯度计算
                        with torch.enable_grad():
                            x_t_j = x_t_j.requires_grad_(True)  # 确保梯度追踪

                            # 预测噪声
                            noise_pred = checkpoint(  # 关键修改：用检查点包裹
                                self.unet,
                                x_t_j,
                                current_timestep.expand(x_t_j.size(0)),
                                text_emb,
                                text_attn
                            ).sample

                            # 计算x̂_0 (带数值保护)
                            alpha_t = scheduler.alphas_cumprod[t_index]
                            sqrt_alpha = torch.sqrt(alpha_t + eps)
                            x0_hat = (x_t_j - torch.sqrt(1 - alpha_t + eps) * noise_pred) / sqrt_alpha

                            # 计算mask区域损失
                            loss = torch.mean(mask * (x0_hat - original) ** 2)
                            loss.backward(retain_graph=False)  # 释放前向计算图
                            g_l = x_t_j.grad.clone()  # 获取梯度
                            x_t_j.grad = None

                            # 禁用梯度追踪
                        x_t_j = x_t_j.detach()

                        # ====== ADM平滑 ======
                        if j == 0:
                            adm_grad = g_l
                        else:
                            # 展平保持批次维度
                            adm_flat = adm_grad.flatten(start_dim=1)
                            g_l_flat = g_l.flatten(start_dim=1)

                            # 批次级相似度计算
                            sim = torch.cosine_similarity(adm_flat, g_l_flat, dim=1)
                            alpha_j = (sim + 1) / 2

                            # 动量更新 (维度自动广播)
                            adm_grad = alpha_j[:, None, None, None] * beta * adm_grad + \
                                       (1 - alpha_j[:, None, None, None] * beta) * g_l

                        # 更新状态
                        x_t_j = x_t_j - (zeta / N) * adm_grad

                        x_t = x_t_j

                # ====== 2. 去噪步骤 ======
                noise_pred = self.unet(
                    x_t,
                    timestep=current_timestep.expand(x_t.size(0)),
                    encoder_hidden_states=text_emb,
                    encoder_attention_mask=text_attn
                ).sample
                prev_t = max((current_timestep.item() - jump_length), 0)  # 确保不越界
                # DDIM更新
                output = scheduler.step(
                    model_output=noise_pred,
                    timestep=current_timestep,
                    sample=x_t,
                    eta=0.0
                )

                x_prev = output.prev_sample

                noisy_original_prev = self._get_noisy_image(original, prev_t)
                x_prev = mask * noisy_original_prev + (1 - mask) * x_prev

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

                # x_prev = torch.clamp(x_prev, min=-1.0, max=1.0)

                noisy_original_prev = self._get_noisy_image(original, prev_t)
                x_prev = mask * noisy_original_prev + (1 - mask) * x_prev

                if prev_t in resample_steps and prev_t > 0:
                    x_prev = self.resample(original, mask, x_prev, prev_t, text_emb, text_attn, device)

                # 更新当前状态
                x_t = x_prev

        return x_t

    def resample(self, original, mask, x_prev, prev_t, text_emb, text_attn, device):
        x_prev_denoised = x_prev
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

            # x_prev_denoised = torch.clamp(x_prev_denoised, min=-1.0, max=1.0)

            # 5.3.3 Repaint区域替换
            noisy_original_prev = self._get_noisy_image(original, prev_t)
            x_prev_denoised = mask * noisy_original_prev + (1 - mask) * x_prev_denoised

        return x_prev_denoised

    def inpaint_ddim_inference(self, fabric, original, mask, text_ids, text_attn):
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

        # 初始化噪声图像
        x_t = torch.randn_like(original, device=device)

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

                noisy_original_prev = self._get_noisy_image(original, prev_t)
                x_prev = mask * noisy_original_prev + (1 - mask) * x_prev

                # 更新当前状态
                x_t = x_prev

        return x_t

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
