import os
from datetime import time
from typing import Tuple

import joblib
import numpy as np
import torch
import torchvision
from datasets import tqdm
from omegaconf import DictConfig
import torch.nn.functional as F
from scipy.optimize import linprog
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from src.model.diffusion_model import DiffusionModel

#%%

class A_functions:
    """
    A class replacing the SVD of a matrix A, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def A(self, vec):
        """
        Multiplies the input vector by A
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])

    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))

    def A_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        temp = self.Ut(vec)
        singulars = self.singulars()

        factors = 1. / singulars
        factors[singulars == 0] = 0.

        #         temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))

    def A_pinv_eta(self, vec, eta):
        """
        Multiplies the input vector by the pseudo inverse of A with factor eta
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = singulars / (singulars * singulars + eta)
        #         print(temp.size(), factors.size(), singulars.size())
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))

    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        raise NotImplementedError()

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        raise NotImplementedError()

# Inpainting
class Inpainting(A_functions):
    def __init__(self, channels, img_dim, missing_indices, device):
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = torch.ones(channels * img_dim ** 2 - missing_indices.shape[0]).to(device)
        self.missing_indices = missing_indices
        self.kept_indices = torch.Tensor([i for i in range(channels * img_dim ** 2) if i not in missing_indices]).to(
            device).long()

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return out.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim ** 2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

    def Lambda(self, vec, a, sigma_y, sigma_t, eta):

        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]

        singulars = self._singulars
        lambda_t = torch.ones(temp.size(1), device=vec.device)
        temp_singulars = torch.zeros(temp.size(1), device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)

        lambda_t = lambda_t.reshape(1, -1)
        out = out * lambda_t

        result = torch.zeros_like(temp)
        result[:, self.kept_indices] = out[:, :self.kept_indices.shape[0]]
        result[:, self.missing_indices] = out[:, self.kept_indices.shape[0]:]
        return result.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out_vec = torch.zeros_like(temp_vec)
        out_vec[:, :self.kept_indices.shape[0]] = temp_vec[:, self.kept_indices]
        out_vec[:, self.kept_indices.shape[0]:] = temp_vec[:, self.missing_indices]

        temp_eps = epsilon.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out_eps = torch.zeros_like(temp_eps)
        out_eps[:, :self.kept_indices.shape[0]] = temp_eps[:, self.kept_indices]
        out_eps[:, self.kept_indices.shape[0]:] = temp_eps[:, self.missing_indices]

        singulars = self._singulars
        d1_t = torch.ones(temp_vec.size(1), device=vec.device) * sigma_t * eta
        d2_t = torch.ones(temp_vec.size(1), device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5

        temp_singulars = torch.zeros(temp_vec.size(1), device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1)
        d2_t = d2_t.reshape(1, -1)
        out_vec = out_vec * d1_t
        out_eps = out_eps * d2_t

        result_vec = torch.zeros_like(temp_vec)
        result_vec[:, self.kept_indices] = out_vec[:, :self.kept_indices.shape[0]]
        result_vec[:, self.missing_indices] = out_vec[:, self.kept_indices.shape[0]:]
        result_vec = result_vec.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

        result_eps = torch.zeros_like(temp_eps)
        result_eps[:, self.kept_indices] = out_eps[:, :self.kept_indices.shape[0]]
        result_eps[:, self.missing_indices] = out_eps[:, self.kept_indices.shape[0]:]
        result_eps = result_eps.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

        return result_vec + result_eps


#%%

class DeqIRDiffusionModel(DiffusionModel):
    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__(model_config)
        self.deq_timestep = 15
        self.eta = 0.15

    def compute_x_tk(self, z_s, t_index, A, x_T, abars):
        """ 实现Proposition 1的解析解 (修正3: 参数和计算逻辑) """
        T = len(abars)
        k = T - t_index - 1  # 当前处理的降噪步

        # 系数计算
        I_minus_pinv = torch.eye(A.size(0)) - A.pinv()
        coeff = torch.sqrt(abars[k] / abars[0])  # √(ᾱ_k / ᾱ_T)

        # 公式9的三部分
        term1 = coeff * (I_minus_pinv @ x_T)
        term2 = A.pinv(z_s[t_index])

        # 加权和修正：使用预计算的z_s序列
        weighted_sum = 0
        for j in range(k + 1, T):
            weight = torch.sqrt(abars[k] / abars[j])
            weighted_sum += weight * (I_minus_pinv @ z_s[j])

        return term1 + term2 + weighted_sum

    def anderson_solver(self, F, x_init, y, A, max_iter=15, m=5):
        """ 严格遵循Algorithm 1实现 """
        x = x_init.clone()
        residuals = []
        history = [x]

        for k in range(max_iter):
            x_new = F(x, y, A)
            g = x_new - x
            residuals.append(g)

            # Anderson加速步骤
            m_k = min(m, k + 1)
            if m_k == 0:
                x_next = x_new
            else:
                # 构建残差矩阵G
                G = torch.stack(residuals[-m_k:], dim=0)  # [m_k, ...]

                # 最小二乘解：min||Gα||^2 s.t. Σα_i=1
                try:
                    # 使用伪逆求解
                    G_flat = G.view(m_k, -1)
                    A_mat = G_flat @ G_flat.t()
                    b = torch.ones(m_k, 1, device=G.device)
                    alpha = torch.linalg.lstsq(A_mat, b)[0]
                    alpha = alpha / alpha.sum()  # 确保Σα=1
                except:
                    alpha = torch.ones(m_k, device=G.device) / m_k

                # 更新状态
                x_next = torch.zeros_like(x_new)
                for i in range(m_k):
                    x_next += alpha[i] * F(history[-(m_k - i)], y, A)

            history.append(x_next)
            x = x_next

        return x

    def compute_x_states(self, z_s, A, x_T, alpha_bars):
        """ 严格实现论文Proposition 1的并行计算 """
        T = len(alpha_bars)
        I_minus_Apinv = torch.eye(A.size(0)) - A.pinv()

        new_states = []
        for k in range(T):
            # 当前要计算的状态x_{T-k}
            # 第一项：系数 * (I - A^†A) * x_T
            coeff1 = torch.sqrt(alpha_bars[T - k - 1] / alpha_bars[T - 1])
            term1 = coeff1 * (I_minus_Apinv @ x_T)

            # 第二项：A^†A * z_{T-k+1}
            term2 = A.pinv(z_s[k])

            # 第三项：求和项
            sum_term = 0
            for s in range(T - k - 1, T):
                weight = torch.sqrt(alpha_bars[T - k - 1] / alpha_bars[s])
                sum_term += weight * (I_minus_Apinv @ z_s[s])

            new_states.append(term1 + term2 + sum_term)

        return torch.stack(new_states)

    def calculate_coefficients(self, alpha_bars):
        """ 严格遵循论文公式(9)计算系数 """
        T = len(alpha_bars)
        c0_list = []
        c1_list = []
        c2_list = []
        eta = self.eta

        # 注意：alpha_bars是降序（从T到0）
        for i in range(T):
            # 当前时间步t对应的ᾱ_t
            abar_t = alpha_bars[i]

            # 计算α_t = ᾱ_t / ᾱ_{t-1}（当i=0时，ᾱ_{t-1}为1）
            if i == 0:
                alpha_t = abar_t
            else:
                alpha_t = abar_t / alpha_bars[i - 1]

            # 计算公式9系数
            sqrt_one_minus_abar = torch.sqrt(1 - abar_t)
            c0_val = sqrt_one_minus_abar * (torch.sqrt(1 - eta ** 2) - (1 - eta ** 2))
            c1_val = sqrt_one_minus_abar * eta
            c2_val = torch.sqrt(1 - eta ** 2) * sqrt_one_minus_abar

            c0_list.append(c0_val)
            c1_list.append(c1_val)
            c2_list.append(c2_val)

        return c0_list, c1_list, c2_list

    def deqir_ddim_inference(self, fabric, original, mask, text_ids, text_attn, total_steps=15, opt_steps=5, lr=0.01):
        # 初始化
        device = fabric.device
        original = fabric.to_device(original)
        mask = fabric.to_device(mask)
        self.eval()

        # 文本编码（保留原条件机制）
        with torch.no_grad():
            text_outputs = self.text_encoder(input_ids=text_ids, attention_mask=text_attn)
            text_emb = text_outputs.last_hidden_state

        # 退化算子（修复任务）
        A = lambda x: mask * x
        A_pinv = lambda y: y  # 伪逆

        # 初始化噪声（优化变量）
        x_T = torch.randn_like(original, device=device)
        x_T.requires_grad_(True)

        # 设置DDIM调度器
        scheduler = self.ddim_scheduler
        scheduler.set_timesteps(total_steps)
        timesteps = scheduler.timesteps.to(device)
        alpha_bars = scheduler.alphas_cumprod[timesteps]  # 关键：从调度器获取ᾱ_t

        # 计算系数
        c0, c1, c2 = self.calculate_coefficients(alpha_bars)

        # 定义固定点函数F（公式10）
        def F(x_states, y, A):
            # 计算所有z_s（公式9）
            z_s_list = []
            for i, t in enumerate(timesteps):
                # 扩散模型去噪
                noise_pred = self.unet(x_states[i], t.expand(x_states.size(1)), text_emb).sample

                # 计算z_s = c0*ε + c1*A^†y + c2*ε_random
                z_s = c0[i] * noise_pred + c1[i] * A_pinv(y) + c2[i] * torch.randn_like(y)
                z_s_list.append(z_s)

            # 并行计算新状态（Proposition 1）
            return self.compute_x_states(z_s_list, A, x_T, alpha_bars)

        # 优化循环（Algorithm 2）
        for opt_step in range(opt_steps):
            with torch.no_grad():
                # 初始化状态
                x_states = x_T.repeat(len(timesteps), 1, 1, 1)

                # Anderson求解器
                x_states = self.anderson_solver(
                    F, x_states, original, A, max_iter=15
                )
                x0_star = x_states[-1]  # 最终恢复结果

            # 梯度反传优化
            with torch.enable_grad():
                # 计算损失（公式12）
                loss = F.mse_loss(A(x0_star), original)
                loss.backward()

                # 更新初始噪声（公式13）
                with torch.no_grad():
                    x_T.data -= lr * x_T.grad.data
                    x_T.grad.zero_()

        return x0_star.detach()

    def validation_step(self, batch, batch_idx, fabric, stage="test", solver="repaint_ddim"):
        # 使用RePaint推理
        if solver == "manifold_dpm":
            filled_genes = self.manifold_dpm_inference(
                fabric,
                batch["gene_matrix"],
                batch["gene_mask"],
                batch["cell_type"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )
        elif solver == "manifold_ddim":
            filled_genes = self.manifold_ddim_inference(
                fabric,
                batch["gene_matrix"],
                batch["gene_mask"],
                batch["cell_type"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )
        elif solver == "deqir_ddim":
            filled_genes = self.deqir_ddim_inference(
                fabric,
                batch["gene_matrix"],
                batch["gene_mask"],
                batch["cell_type"],
                batch["text_input_ids"],
                batch["text_attention_mask"]
            )
        elif solver == "repaint_ddim":
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
            self.save_results(sorted_genes)

        return {
            "mse": total_mse,
            "pcc": total_pcc,
            "cossim": total_cossim,
            "ssim": total_ssim,
        }