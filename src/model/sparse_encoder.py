import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from sklearn.decomposition import DictionaryLearning
from transformers import AutoModel, AutoTokenizer


class SparseEncoder:
    def __init__(self, dictionary, text_reducer, device='cuda'):
        self.dictionary = dictionary
        self.text_reducer = text_reducer
        self.device = device
        self.D = dictionary.view(dictionary.size(0), -1).t()  # 字典矩阵 (d x k)
        self.D_norms = torch.norm(self.D, dim=0)  # 字典原子的范数

    def text_aware_omp(self, patches, text_features, distances, n_nonzeros, max_iter=100, tol=1e-6):
        """
        文本和距离感知的正交匹配追踪(OMP)算法
        参数:
            patches: 输入图像块 [batch_size, patch_size, patch_size]
            text_features: 文本特征 [batch_size, text_dim]
            distances: 切比雪夫距离 [batch_size]
            n_nonzeros: 每个块的非零系数数量 [batch_size]
        返回:
            稀疏编码后的图像块 [batch_size, patch_size, patch_size]
        """
        batch_size = patches.size(0)
        patch_size = patches.size(1)
        k = self.D.size(1)  # 字典原子数量

        # 展平图像块 [batch_size, d]
        Y = patches.view(batch_size, -1)

        # 初始化
        X = torch.zeros(batch_size, k, device=self.device)  # 稀疏系数
        residual = Y.clone()  # 初始残差
        support = torch.zeros(batch_size, k, dtype=torch.bool, device=self.device)  # 支持集

        # 文本特征影响字典原子选择
        text_weights = self.text_reducer(text_features)  # [batch_size, reduced_dim]
        text_weights = F.softmax(text_weights, dim=1)  # 归一化

        # 距离权重 - 距离越大，稀疏性越强
        dist_weights = 1.0 / (1.0 + distances)  # [batch_size]
        dist_weights = dist_weights.view(-1, 1)  # [batch_size, 1]

        # 迭代选择原子
        for i in range(max_iter):
            # 计算残差与字典原子的相关系数
            correlations = torch.abs(residual @ self.D)  # [batch_size, k]

            # 应用文本权重 - 增强与文本相关的原子
            correlations *= text_weights.unsqueeze(1).expand(-1, k, -1).mean(dim=2)

            # 应用距离权重 - 距离越大，稀疏性越强
            correlations *= dist_weights

            # 归一化相关系数
            correlations /= (self.D_norms + 1e-8)

            # 找到最相关的原子（排除已选择的）
            correlations[support] = -np.inf  # 排除已选原子
            max_vals, max_idxs = torch.max(correlations, dim=1)  # [batch_size]

            # 更新支持集
            support[range(batch_size), max_idxs] = True

            # 检查是否达到目标稀疏度
            active_mask = (i < n_nonzeros - 1).float()  # 需要继续迭代的样本

            # 对于每个样本，使用最小二乘法更新系数
            for b in range(batch_size):
                if active_mask[b] > 0.5:  # 还需要选择更多原子
                    # 获取当前支持集
                    S = support[b].nonzero(as_tuple=True)[0]

                    # 提取子字典
                    D_S = self.D[:, S]

                    # 最小二乘解
                    try:
                        # 使用伪逆求解
                        D_S_pinv = torch.pinverse(D_S)
                        x_S = D_S_pinv @ Y[b]

                        # 更新系数
                        X[b, S] = x_S

                        # 更新残差
                        residual[b] = Y[b] - D_S @ x_S
                    except:
                        # 奇异矩阵处理
                        residual[b] = 0
                else:
                    # 已达到目标稀疏度，停止更新
                    pass

            # 检查残差是否足够小
            if torch.norm(residual) < tol:
                break

        # 使用稀疏系数重建图像块
        reconstructed = X @ self.D.t()
        return reconstructed.view(batch_size, patch_size, patch_size)

    def distance_aware_lasso(self, patches, distances, alpha=0.1, max_iter=1000):
        """
        距离感知的Lasso回归
        参数:
            patches: 输入图像块 [batch_size, patch_size, patch_size]
            distances: 切比雪夫距离 [batch_size]
            alpha: L1正则化强度
        返回:
            稀疏编码后的图像块 [batch_size, patch_size, patch_size]
        """
        batch_size = patches.size(0)
        patch_size = patches.size(1)
        k = self.D.size(1)  # 字典原子数量

        # 展平图像块 [batch_size, d]
        Y = patches.view(batch_size, -1)

        # 初始化系数
        X = torch.zeros(batch_size, k, device=self.device)

        # 距离权重 - 距离越大，正则化越强
        dist_weights = 1.0 / (1.0 + distances)  # [batch_size]
        alphas = alpha * dist_weights.view(-1, 1)  # [batch_size, 1]

        # 迭代求解
        for _ in range(max_iter):
            for j in range(k):
                # 计算残差
                residual = Y - X @ self.D.t() + X[:, j:j + 1] * self.D[:, j]

                # 计算相关系数
                corr = residual @ self.D[:, j]

                # 软阈值更新
                X[:, j] = torch.sign(corr) * torch.relu(torch.abs(corr) - alphas[:, 0])

        # 重建图像块
        reconstructed = X @ self.D.t()
        return reconstructed.view(batch_size, patch_size, patch_size)

    def sparse_encode(self, patches, text_features, distances, n_nonzeros):
        """
        自适应选择稀疏编码方法
        - 对于低稀疏度使用OMP
        - 对于中等稀疏度使用Lasso
        - 对于高稀疏度使用近似SVD
        """
        batch_size = patches.size(0)

        # 根据稀疏度选择方法
        results = torch.zeros_like(patches)

        # 分组处理
        low_sparsity_mask = n_nonzeros <= 5
        med_sparsity_mask = (n_nonzeros > 5) & (n_nonzeros <= 10)
        high_sparsity_mask = n_nonzeros > 10

        # 低稀疏度: OMP
        if torch.any(low_sparsity_mask):
            low_patches = patches[low_sparsity_mask]
            low_text = text_features[low_sparsity_mask]
            low_dist = distances[low_sparsity_mask]
            low_nnz = n_nonzeros[low_sparsity_mask]

            encoded = self.text_aware_omp(low_patches, low_text, low_dist, low_nnz)
            results[low_sparsity_mask] = encoded

        # 中等稀疏度: Lasso
        if torch.any(med_sparsity_mask):
            med_patches = patches[med_sparsity_mask]
            med_dist = distances[med_sparsity_mask]

            # 计算自适应alpha
            alpha = 0.2 * (1.0 - med_dist / med_dist.max())
            encoded = self.distance_aware_lasso(med_patches, med_dist, alpha=alpha)
            results[med_sparsity_mask] = encoded

        # 高稀疏度: 近似SVD
        if torch.any(high_sparsity_mask):
            high_patches = patches[high_sparsity_mask]
            encoded = self.approximate_svd(high_patches, n_nonzeros[high_sparsity_mask])
            results[high_sparsity_mask] = encoded

        return results

    def approximate_svd(self, patches, n_nonzeros):
        """
        基于SVD的近似稀疏编码
        """
        batch_size = patches.size(0)
        patch_size = patches.size(1)
        d = patch_size * patch_size

        # 展平图像块 [batch_size, d]
        Y = patches.view(batch_size, -1)

        # 计算字典的SVD
        U, S, Vt = torch.svd(self.D)

        # 重建每个图像块
        reconstructed = torch.zeros_like(Y)
        for b in range(batch_size):
            # 计算系数
            coeffs = Vt.t() @ (torch.diag(1.0 / (S + 1e-8)) @ U.t() @ Y[b])

            # 保留最大的n_nonzeros个系数
            _, idxs = torch.topk(torch.abs(coeffs), k=int(n_nonzeros[b]))
            mask = torch.zeros_like(coeffs)
            mask[idxs] = 1
            coeffs = coeffs * mask

            # 重建
            reconstructed[b] = U @ torch.diag(S) @ Vt.t() @ coeffs

        return reconstructed.view(batch_size, patch_size, patch_size)