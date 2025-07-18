# v2025_7_17a
import os
from typing import Tuple

import joblib
import numpy as np
import torch
import torchvision
from omegaconf import DictConfig
import torch.nn.functional as F
from scipy.optimize import linprog
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from src.model.diffusion_model import DiffusionModel

# %%

from typing import Tuple
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=DeprecationWarning)


def constrained_mean(delta, low_bounds, up_bounds, metric):
    """为可修改点计算满足总和约束的最优变化量"""
    # 没有可修改点时直接返回
    if len(low_bounds) == 0:
        return np.array([])

    if metric == 'l2':
        # 二分搜索查找最优Lambda
        lo = np.min(low_bounds) - 1
        hi = np.max(up_bounds) + 1

        for _ in range(50):  # 最大迭代次数
            lam = (lo + hi) / 2
            d_i = np.clip(lam, low_bounds, up_bounds)
            total = np.sum(d_i)

            if total < delta:
                lo = lam
            elif total > delta:
                hi = lam
            else:
                break

            if hi - lo < 1e-6:
                break

        d_i = np.clip((lo + hi) / 2, low_bounds, up_bounds)
        return d_i.astype(np.float32)  # 转换为float32

    else:
        raise ValueError(f"不支持的优化类型: {metric}")


def fast_batch_modify_sequence(
        X: torch.Tensor,
        Y: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        metric: str = 'l1',
        fabric=None
) -> torch.Tensor:
    """
        批量修改序列以满足目标平均值B，最小化指定距离度量 (支持L1, L2, MinMax)
        目前修改只考虑l1和l2
        参数:
        X : (M, N) 原始序列值 ∈ [-1, 1]
        Y : (M, N) 掩码 (1=不可修改, 0=可修改)
        A : (M, 1) 初始平均值
        B : (M, 1) 目标平均值
        metric : 距离度量 ('l1', 'l2', 'minmax')

        返回:
        X_new : (M, N) 修改后的序列
        """
    assert X.shape == Y.shape, "X and Y must have the same shape"
    assert A.shape == B.shape and A.dim() == 2 and A.shape[1] == 1, "A and B must be (M, 1) tensors"

    if fabric is not None:
        fabric.barrier()
        device = fabric.device
        X, Y, A, B = X.to(device), Y.to(device), A.to(device), B.to(device)

    if torch.all(Y == 1):
        return X

    device = X.device
    dtype = X.dtype
    M, N = X.shape

    if metric == 'l1':
        delta = (B - A) * N
        modifiable_mask = (Y == 0)

        # 使用内存高效操作
        low_bounds = torch.empty_like(X)
        up_bounds = torch.empty_like(X)
        torch.clamp(-1 - X, min=-2, max=0, out=low_bounds)
        torch.clamp(1 - X, min=0, max=2, out=up_bounds)
        low_bounds.masked_fill_(~modifiable_mask, 0)
        up_bounds.masked_fill_(~modifiable_mask, 0)

        sort_metric = low_bounds + up_bounds
        sort_metric = sort_metric.masked_fill(
            ~modifiable_mask, torch.finfo(dtype).max
        )

        # 使用float32排序防止精度丢失
        sorted_indices = torch.argsort(sort_metric.to(torch.float32), dim=1)

        # 高精度累加
        low_vec = torch.gather(low_bounds, 1, sorted_indices).cumsum(1, dtype=torch.float32)
        up_vec = torch.gather(up_bounds, 1, sorted_indices).cumsum(1, dtype=torch.float32)

        # 关键计算转float32
        target_vec = (low_vec + up_vec) / 2.0
        delta_float = delta.to(torch.float32)
        k_index = torch.searchsorted(target_vec, delta_float, right=False).clamp(0, N - 1)

        low_vals = torch.gather(low_vec, 1, k_index)
        up_vals = torch.gather(up_vec, 1, k_index)

        # 处理除零风险
        N_minus_k = (N - k_index).clamp_min(1).to(torch.float32)
        lambda_opt = (delta_float - (low_vals + up_vals) / 2) / N_minus_k

        # 输出转半精度
        lambda_opt = lambda_opt.to(dtype).expand_as(X)
        d = torch.clamp(lambda_opt, min=low_bounds, max=up_bounds)
        return torch.where(modifiable_mask, X + d, X).to(dtype)

    elif metric == 'l1_old':
        # 计算每个问题所需的总变化量
        delta = (B - A) * N  # (M,1)

        modifiable_mask = (Y == 0)

        # 修正1: 显式隔离不可修改位置
        low_bounds = torch.clamp(-1 - X, min=-2, max=0).masked_fill(~modifiable_mask, 0)
        up_bounds = torch.clamp(1 - X, min=0, max=2).masked_fill(~modifiable_mask, 0)

        # 修正2: 构建安全的排序指标（不可修改位置置为inf）
        sort_metric = low_bounds + up_bounds
        sort_metric = sort_metric.masked_fill(~modifiable_mask, float('inf'))

        sorted_indices = torch.argsort(sort_metric, dim=1)  # (M, N)

        low_vec = torch.gather(low_bounds, 1, sorted_indices).cumsum(1, dtype=dtype)  # (M, N)
        up_vec = torch.gather(up_bounds, 1, sorted_indices).cumsum(1, dtype=dtype)  # (M, N)

        # 修正3: 使用delta原始维度(M,1)，避免squeeze
        k_index = torch.searchsorted(
            (low_vec + up_vec) / 2.0,
            delta,  # 直接使用(M,1)!
            right=False
        ).clamp(0, N - 1)  # (M,1)

        # 保持k_index为(M,1)用于gather
        low_vals = torch.gather(low_vec, 1, k_index)  # (M,1)
        up_vals = torch.gather(up_vec, 1, k_index)  # (M,1)

        # 修正4: 安全类型转换
        N_minus_k = (N - k_index).to(dtype)  # (M,1)

        lambda_opt = (delta - (low_vals + up_vals) / 2) / N_minus_k  # (M,1)

        # 计算变化量并应用
        d = torch.clamp(
            lambda_opt.expand_as(X),
            min=low_bounds,
            max=up_bounds
        ).to(dtype)
        return torch.where(modifiable_mask, X + d, X).to(dtype)

    elif metric == "l2":

        # 计算每个问题所需的总变化量
        delta = (B - A) * N  # (M,1)

        modifiable_mask = (Y == 0)

        # 修正1: 显式隔离不可修改位置
        low_bounds = torch.clamp(-1 - X, min=-2, max=0).masked_fill(~modifiable_mask, 0)
        up_bounds = torch.clamp(1 - X, min=0, max=2).masked_fill(~modifiable_mask, 0)

        # 修正2: 构建安全的排序指标（不可修改位置置为inf）
        sort_metric = low_bounds + up_bounds
        sort_metric = sort_metric.masked_fill(~modifiable_mask, float('inf'))

        sum_low = torch.where(modifiable_mask, low_bounds, 0).sum(dim=1, keepdim=True)
        sum_high = torch.where(modifiable_mask, up_bounds, 0).sum(dim=1, keepdim=True)
        delta_clamped = torch.clamp(delta, min=sum_low, max=sum_high)

        # 3. 简化二分搜索（合并冗余计算）
        lamb_low = torch.full((M, 1), -3.0, device=device, dtype=dtype)
        lamb_high = torch.full((M, 1), 3.0, device=device, dtype=dtype)
        diff_threshold = 1e-6  # 添加提前终止条件

        # 4. 优化迭代：减少计算量并提前终止
        for _ in range(20):  # 减少迭代次数（20次足够满足精度）
            with torch.no_grad():
                lamb_mid = (lamb_low + lamb_high) / 2

                # 关键优化：避免全量expand_as和clamp
                d_temp = torch.maximum(low_bounds, torch.minimum(up_bounds, lamb_mid))
                sum_d = torch.where(modifiable_mask, d_temp, 0).sum(dim=1, keepdim=True)

                # 更新搜索边界（使用原地操作）
                update_mask = sum_d < delta_clamped
                lamb_low = torch.where(update_mask, lamb_mid, lamb_low)
                lamb_high = torch.where(~update_mask, lamb_mid, lamb_high)

                # 提前终止检测：检查收敛性
                if torch.all((lamb_high - lamb_low) < diff_threshold):
                    break

        # 5. 使用已计算的中间值避免重复clamp
        lamb_final = (lamb_low + lamb_high) / 2
        d_final = torch.maximum(low_bounds, torch.minimum(up_bounds, lamb_final))

        return torch.where(modifiable_mask, X + d_final, X)


def batch_modify_sequence(
        X: torch.Tensor,
        Y: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        metric: str = 'l2'
) -> torch.Tensor:
    """
    批量修改序列以满足目标平均值B，最小化指定距离度量 (支持L1, L2, MinMax)

    参数:
    X : (M, N) 原始序列值 ∈ [-1, 1]
    Y : (M, N) 掩码 (1=不可修改, 0=可修改)
    A : (M, 1) 初始平均值
    B : (M, 1) 目标平均值
    metric : 距离度量 ('l1', 'l2', 'minmax')

    返回:
    X_new : (M, N) 修改后的序列
    dist_values : (M,) 每个序列的最小距离值
    """
    assert X.shape == Y.shape, "X and Y must have the same shape"
    assert A.shape == B.shape and A.dim() == 2 and A.shape[1] == 1, "A and B must be (M, 1) tensors"

    M, N = X.shape
    device = X.device
    dtype = X.dtype  # 获取输入的数据类型
    X_new = X.clone()
    dist_values = torch.zeros(M, device=device, dtype=dtype)

    # 计算每个问题所需的总变化量
    delta = (B - A) * N

    # 如果没有可修改点（全部掩码为1），直接返回原序列
    if torch.all(Y == 1):
        return X, torch.zeros(M, device=device, dtype=dtype)

    # 计算可修改点的边界约束
    modifiable_mask = (Y == 0)  # (M, N)
    low_bounds = torch.clamp(-1 - X, min=-2, max=0)
    up_bounds = torch.clamp(1 - X, min=0, max=2)

    if metric == 'l1':
        # === 向量化 KKT 求解器 (O(MKlogK)) ===
        # 计算每个可修改点的原始排序
        sorted_indices = torch.argsort(low_bounds + up_bounds, dim=1)  # (M, N)

        # 构造累积和向量
        low_vec = torch.gather(low_bounds, 1, sorted_indices).cumsum(1)
        up_vec = torch.gather(up_bounds, 1, sorted_indices).cumsum(1)

        # 寻找最优阈值点 k
        k_index = torch.searchsorted(
            (low_vec + up_vec) / 2.0,
            delta.expand(-1, N),
            right=False
        ).clamp(0, N - 1)

        # 计算最优 lambda (广播索引)
        idx_exp = k_index.unsqueeze(1).expand(-1, N)
        low_vals = torch.gather(low_vec, 1, idx_exp[:, :1])
        up_vals = torch.gather(up_vec, 1, idx_exp[:, :1])
        lambda_opt = (delta - (low_vals + up_vals) / 2) / (N - k_index)

        # 计算变化量 (使用向量化剪辑)
        d = torch.clamp(
            lambda_opt.expand_as(X),
            min=low_bounds,
            max=up_bounds
        )

        # 应用修改并计算距离
        X_new = torch.where(modifiable_mask, X + d, X)

    elif metric == 'l1_old':
        # ===== L1距离最小化 =====
        for i in range(M):
            if not modifiable_mask[i].any():
                continue  # 没有可修改点，直接跳过

            # 提取当前问题的可修改点
            mod_idx = torch.where(modifiable_mask[i])[0]
            K = len(mod_idx)
            low_i = low_bounds[i, mod_idx].cpu().numpy()
            up_i = up_bounds[i, mod_idx].cpu().numpy()
            delta_i = delta[i].item()

            # 设置线性规划问题
            c = np.zeros(2 * K)
            c[K:] = 1  # 目标函数系数 (t_i部分)

            # 等式约束: sum(d_i) = delta
            A_eq = np.zeros((1, 2 * K))
            A_eq[0, :K] = 1
            b_eq = np.array([delta_i])

            # 不等式约束矩阵 (总共4K个约束)
            A_ub = np.zeros((4 * K, 2 * K))
            b_ub = np.zeros(4 * K)

            # 约束1: d_i <= t_i (K个约束)
            A_ub[:K, :K] = np.eye(K)
            A_ub[:K, K:] = -np.eye(K)

            # 约束2: -d_i <= t_i (K个约束)
            A_ub[K:2 * K, :K] = -np.eye(K)
            A_ub[K:2 * K, K:] = -np.eye(K)

            # 约束3: d_i >= low_i (K个约束) => -d_i <= -low_i
            A_ub[2 * K:3 * K, :K] = -np.eye(K)
            b_ub[2 * K:3 * K] = -low_i

            # 约束4: d_i <= up_i (K个约束)
            A_ub[3 * K:4 * K, :K] = np.eye(K)
            b_ub[3 * K:4 * K] = up_i

            # 变量边界
            bounds = [(None, None)] * K + [(0, None)] * K

            # 求解线性规划 (使用HiGHS方法)
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                              bounds=bounds, method='highs')
            except Exception as e:
                # print(f"警告: 线性规划求解失败，问题{i}: {e}")
                # 使用回退方案（L2方法）
                d_i = constrained_mean(delta_i, low_i, up_i, 'l2')
                d_i_tensor = torch.tensor(d_i, device=device, dtype=dtype)
                X_new[i, mod_idx] = X[i, mod_idx] + d_i_tensor
                dist_values[i] = np.sqrt(np.sum(np.square(d_i)))
                continue

            if res.success:
                d_i = res.x[:K]
                # 将numpy数组转换为与输入相同类型的tensor
                d_i_tensor = torch.tensor(d_i, device=device, dtype=dtype)
                # 更新序列
                X_new[i, mod_idx] = X[i, mod_idx] + d_i_tensor
                dist_values[i] = np.sum(np.abs(d_i))
            else:
                # 使用回退方案（L2方法）
                # print(f"警告: L1优化失败，问题{i}")
                d_i = constrained_mean(delta_i, low_i, up_i, 'l2')
                d_i_tensor = torch.tensor(d_i, device=device, dtype=dtype)
                X_new[i, mod_idx] = X[i, mod_idx] + d_i_tensor
                dist_values[i] = np.sqrt(np.sum(np.square(d_i)))

    elif metric == 'l2':
        # ===== L2距离最小化 (完全向量化) =====
        # 计算每个问题所需分配的总量
        sum_low = torch.where(modifiable_mask, low_bounds, 0).sum(dim=1, keepdim=True)
        sum_high = torch.where(modifiable_mask, up_bounds, 0).sum(dim=1, keepdim=True)

        # 确保delta在可行范围内 (题目保证有解)
        delta_clamped = torch.clamp(delta, min=sum_low, max=sum_high)

        # 双线性搜索: 计算初始搜索范围
        lamb_low = torch.full((M, 1), -3.0, device=device, dtype=dtype)
        lamb_high = torch.full((M, 1), 3.0, device=device, dtype=dtype)

        # 二分搜索找到最优lambda (最多迭代30次)
        for _ in range(30):
            lamb_mid = (lamb_low + lamb_high) / 2

            # 对每个点应用剪辑 (使用广播)
            d_clamped = torch.clamp(
                lamb_mid.expand_as(X),
                min=low_bounds,
                max=up_bounds
            )

            # 计算当前变化量和 (仅考虑可修改点)
            sum_d = torch.where(modifiable_mask, d_clamped, 0).sum(dim=1, keepdim=True)

            # 更新搜索边界
            mask_low = (sum_d < delta_clamped)
            lamb_low = torch.where(mask_low, lamb_mid, lamb_low)
            lamb_high = torch.where(~mask_low, lamb_mid, lamb_high)

        # 计算最终变化量
        final_lamb = (lamb_low + lamb_high) / 2
        d_final = torch.clamp(
            final_lamb.expand_as(X),
            min=low_bounds,
            max=up_bounds
        )

        # 应用变化
        X_new = torch.where(modifiable_mask, X + d_final, X)

        # 计算距离值
        d_abs = torch.where(modifiable_mask, torch.abs(d_final), 0)


    elif metric == 'minmax':
        # ===== MinMax距离最小化 =====
        # 初始化二分搜索边界
        M_low = torch.zeros(M, 1, device=device, dtype=dtype)
        abs_low = torch.abs(low_bounds)
        abs_up = torch.abs(up_bounds)
        abs_bound = torch.where(abs_low > abs_up, abs_low, abs_up)
        M_high, _ = torch.max(torch.where(modifiable_mask, abs_bound, torch.tensor(0.0, device=device, dtype=dtype)),
                              dim=1, keepdim=True)
        M_high = torch.clamp(M_high, min=1e-6)  # 防止全零

        # 二分搜索最小M值 (最多迭代30次)
        for _ in range(30):
            M_mid = (M_low + M_high) / 2

            # 计算有效变化范围
            low_vals = torch.maximum(low_bounds, -M_mid)
            high_vals = torch.minimum(up_bounds, M_mid)

            # 计算变化量边界和
            sum_low_val = torch.where(modifiable_mask, low_vals, 0).sum(dim=1, keepdim=True)
            sum_high_val = torch.where(modifiable_mask, high_vals, 0).sum(dim=1, keepdim=True)

            # 检查可行性
            feasible = (sum_low_val <= delta) & (delta <= sum_high_val)

            # 更新搜索边界
            M_high = torch.where(feasible, M_mid, M_high)
            M_low = torch.where(feasible, M_low, M_mid)

        # 分配变化量
        low_vals_final = torch.maximum(low_bounds, -M_high)
        high_vals_final = torch.minimum(up_bounds, M_high)

        # 初始化为下界
        d_minmax = low_vals_final.clone()
        current_sum = torch.where(modifiable_mask, d_minmax, 0).sum(dim=1, keepdim=True)
        rem = delta - current_sum

        # 计算可调整量
        slack = high_vals_final - low_vals_final
        slack_sum = torch.where(modifiable_mask, slack, 0).sum(dim=1, keepdim=True)
        slack_sum = torch.clamp(slack_sum, min=1e-10)  # 防止除以零

        # 比例分配剩余变化量
        ratio = rem / slack_sum
        d_add = slack * ratio

        # 应用变化
        d_minmax += d_add
        X_new = torch.where(modifiable_mask, X + d_minmax, X)

        # 计算MinMax距离
        d_abs = torch.abs(d_minmax)
        dist_values, _ = torch.max(torch.where(modifiable_mask, d_abs, torch.zeros_like(d_abs)), dim=1)

    else:
        raise ValueError(f"不支持的距离度量: {metric}. 请选择 'l1', 'l2' 或 'minmax'")

    return X_new


# %%

class ManifoldDiffusionModel(DiffusionModel):
    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__(model_config)
        self.correction_steps = {
            480: list(range(78, 90)),
            380: list(range(56, 78)),
            280: list(range(34, 56)),
            180: list(range(12, 34)),
            80: list(range(0, 12))
        }
        print("correction step: ", self.correction_steps)
        self.manifold_dictionary = joblib.load(self.config.manifold_path)

    def fast_projection(self, X, Y, T, max_iters=50, device='cuda'):
        # 确保所有输入都在正确设备和类型
        X = X.clone().detach().to(device).float().requires_grad_(True)  # 关键：确保X需要梯度
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32, device=device)
        else:
            Y = Y.to(device).float()
        T = T.clone().detach().to(device).float()

        # 预计算统计量
        with torch.no_grad():
            mean = Y.mean(dim=0, keepdim=True)
            cov = torch.cov(Y.T)
            reg = 1e-6 * torch.eye(cov.size(0), device=device)
            cov_inv = torch.linalg.pinv(cov + reg)

        # 优化设置
        delta = torch.zeros_like(X, requires_grad=True, device=device)
        optimizer = torch.optim.LBFGS([delta], lr=0.2, max_iter=4, history_size=5, line_search_fn='strong_wolfe')

        # 创建闭包函数
        def closure():
            optimizer.zero_grad()

            # 核心计算 - 确保这个图能计算梯度
            X_hat = X + delta

            # 计算马氏距离部分
            diff = X_hat - mean
            mal_dist = torch.sum(diff * (diff @ cov_inv))

            # 计算最近邻距离
            rand_idx = torch.randperm(Y.size(0))[:20]  # 随机取20点
            nn_dist = torch.cdist(X_hat, Y[rand_idx]).min()

            # 组合损失函数
            loss = mal_dist + 0.2 * nn_dist

            # 计算梯度
            loss.backward()
            return loss

        # 执行优化
        for _ in range(max(max_iters // 4, 1)):  # 确保至少运行1次
            optimizer.step(closure)

            # 应用约束
            with torch.no_grad():
                T = torch.abs(T)  # 确保T是正数
                delta.data = torch.clamp(delta.data, -T, T)

        # 返回结果，断开计算图
        return (X + delta).detach()

    def correction(self, original, x0_pred, mask, scheduler, t, cell_type, text_emb, text_attn, modify_layer, fabric,
                   device):

        x0_pred_cor = original * mask + x0_pred * (1 - mask)
        X_batch = self.get_sparse_code(x0_pred_cor)
        T_batch = self.get_sparse_code((1 - mask))
        # 创建存储校正结果的张量
        X_hat_batch = torch.zeros_like(X_batch)

        for b in range(x0_pred.shape[0]):
            X = X_batch[b].unsqueeze(0)
            if cell_type[b] in self.manifold_dictionary:
                Y = self.manifold_dictionary[cell_type[b]]
            else:
                print("[Warning] cell type not found in manifold dictionary]")
                Y = self.manifold_dictionary["Mast Cell"]
            if not isinstance(Y, torch.Tensor):
                Y = torch.tensor(Y, device=device, dtype=torch.float32)
            else:
                Y = Y.to(device)
            T = T_batch[b]
            T = torch.abs(T)

            X_hat_batch[b] = self.fast_projection(X, Y, T, max_iters=30, device=device)

        # 90 -> 180*180
        for layer in modify_layer:
            X = self.extract_layer_i(x0_pred_cor, layer)
            Y = self.extract_layer_i(mask, layer)

            A = X.mean(dim=1, keepdim=True)
            B = X_hat_batch[:, layer].unsqueeze(1)

            Y_new = fast_batch_modify_sequence(X, Y, A, B, "l1_old")

            x0_pred_cor = self.restore_layer_i(x0_pred_cor, Y_new, layer)

        # 重采样
        x0_pred_cor = x0_pred_cor.unsqueeze(1)

        x_t = self._get_noisy_image(x0_pred_cor * (1 - mask) + original * mask, t)
        # x_t = torch.clamp(x_t, -1.0, 1.0)

        with torch.no_grad():
            # 注意：t.expand()自动继承设备
            noise_pred = self.unet(
                x_t,
                t.expand(x_t.size(0)),
                encoder_hidden_states=text_emb,
                encoder_attention_mask=text_attn
            ).sample

            if self.scheduler == "manifold_ddim":
                x_prev = scheduler.step(
                    noise_pred,
                    t,
                    x_t,
                    eta=0.0  # 完全确定性采样
                ).prev_sample

            else:
                x_prev = scheduler.step(
                    noise_pred,
                    t,
                    x_t,
                    # eta=0.0  # 完全确定性采样
                ).prev_sample

        # x_prev = torch.clamp(x_prev, min=-1.0, max=1.0)
        return x_prev

    def get_chebyshev_coordinates(self, i: int, device=None):
        """获取第i层所有像素的坐标"""
        H, W = 180, 180
        center = (W - 1) / 2.0

        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        # 计算切比雪夫距离
        dist = torch.max(torch.abs(x - center), torch.abs(y - center))
        # 筛选距离为i的像素
        mask = (torch.floor(dist) == i)

        # 收集所有符合条件像素的坐标
        positions = torch.nonzero(mask, as_tuple=True)
        return positions[0], positions[1]  # (rows, cols)

    def extract_layer_i(self, X: torch.Tensor, i: int) -> torch.Tensor:
        """
        提取图像X的第i切比雪夫层的所有像素值
        参数:
            X: 输入图像 [B, H, W]
            i: 目标层数
        返回:
            Y: 嵌入向量 [B, C]，C是该层的像素数量
        """
        if X.dim() == 4:
            # 压缩单通道维度 (C=1)
            X = X.squeeze(1)
        assert len(X.shape) == 3, "输入张量必须是3维 [B, H, W]"

        B = X.shape[0]
        device = X.device

        # 获取该层所有像素的坐标
        rows, cols = self.get_chebyshev_coordinates(i, device=device)
        num_pixels = rows.shape[0]

        # 创建索引张量
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(-1, num_pixels).flatten()
        row_indices = rows.view(1, -1).expand(B, -1).flatten()
        col_indices = cols.view(1, -1).expand(B, -1).flatten()

        # 提取所有值
        values = X[batch_indices, row_indices, col_indices]

        # 重新组织形状为[B, num_pixels]
        return values.view(B, num_pixels)

    def restore_layer_i(self, X: torch.Tensor, Y: torch.Tensor, i: int) -> torch.Tensor:
        """
        将嵌入值Y恢复到图像X的第i切比雪夫层
        参数:
            X: 原始图像 [B, H, W]
            Y: 嵌入向量 [B, C]，C是该层的像素数量
            i: 目标层数
        返回:
            restored: 恢复后的图像 [B, H, W]
        """
        if X.dim() == 4:
            # 压缩单通道维度 (C=1)
            X = X.squeeze(1)
        assert len(X.shape) == 3, "输入张量必须是3维 [B, H, W]"
        B, H, W = X.shape
        device = X.device

        # 获取该层所有像素的坐标
        rows, cols = self.get_chebyshev_coordinates(i, device=device)
        num_pixels = rows.shape[0]

        # 验证嵌入向量形状匹配
        assert Y.shape == (B, num_pixels), f"嵌入向量形状应为{[B, num_pixels]}，实际为{Y.shape}"

        # 创建索引张量
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(-1, num_pixels).flatten()
        row_indices = rows.view(1, -1).expand(B, -1).flatten()
        col_indices = cols.view(1, -1).expand(B, -1).flatten()

        # 复制原始图像以避免修改
        restored = X.clone()

        # 恢复值到目标层
        restored[batch_indices, row_indices, col_indices] = Y.flatten()
        # .to(dtype=restored.dtype)

        return restored

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
        dtype = gene_matrix.dtype

        # 显式指定张量数据类型与输入一致
        layer_sums = torch.zeros(B, 90, device=device, dtype=dtype)
        layer_counts = torch.zeros(B, 90, device=device, dtype=dtype)

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

    def manifold_dpm_inference(self, fabric, original, mask, cell_type, text_ids, text_attn):

        device = fabric.device

        # 将输入数据转移到设备
        original = fabric.to_device(original)
        mask = fabric.to_device(mask)
        text_ids = fabric.to_device(text_ids)
        text_attn = fabric.to_device(text_attn)

        # 确保模型在评估模式
        self.eval()

        # 配置DPM调度器
        scheduler = self.dpm_scheduler
        scheduler.set_timesteps(self.config.dpm_timesteps)
        timesteps = scheduler.timesteps.to(device)  # 确保时间步在正确设备
        jump_length = self.config.ddpm_timesteps // self.config.dpm_timesteps  # 计算跳步长度

        # 文本编码
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=text_ids,
                attention_mask=text_attn
            )
            text_emb = text_outputs.last_hidden_state

        # 初始化噪声图像 - 基于输入批量大小
        x_t = torch.randn_like(original, device=device)

        # 主循环 - 使用no_backward_sync避免梯度计算

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

            # 更新步骤
            x_prev = scheduler.step(
                noise_pred,
                t,
                x_t,
                # eta=0.0  # 完全确定性采样
            ).prev_sample

            alpha_bar_t = scheduler.alphas_cumprod[t]
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            noisy_original_prev = self._get_noisy_image(original, prev_t)
            x_prev = mask * noisy_original_prev + (1 - mask) * x_prev

            if prev_t in self.correction_steps and prev_t > 0:
                modify_layer = self.correction_steps[t]
                x_prev = self.correction(original, x_0_pred, mask, scheduler, t, cell_type, text_emb, text_attn,
                                         modify_layer, device)

            # 更新当前状态
            x_t = x_prev

        return x_t

    def manifold_ddim_inference(self, fabric, original, mask, cell_type, text_ids, text_attn):

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

        # 主循环 - 使用no_backward_sync避免梯度计算

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

            prev_t = int(max((t.item() - jump_length), 0))  # 确保不越界

            # DDIM更新步骤
            x_prev = scheduler.step(
                noise_pred,
                t,
                x_t,
                eta=0.0  # 完全确定性采样
            ).prev_sample

            x_0_pred = scheduler.step(
                noise_pred,
                t,
                x_t,
                eta=0.0  # 完全确定性采样
            ).pred_original_sample

            noisy_original_prev = self._get_noisy_image(original, prev_t)
            x_prev = mask * noisy_original_prev + (1 - mask) * x_prev

            if prev_t in self.correction_steps:
                modify_layer = self.correction_steps[prev_t]
                x_prev = self.correction(original, x_0_pred, mask, scheduler, t, cell_type, text_emb, text_attn,
                                         modify_layer, fabric, device)

            # 更新当前状态
            x_t = x_prev

        return x_t

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
