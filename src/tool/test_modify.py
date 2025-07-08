import torch
import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=DeprecationWarning)


def batch_modify_sequence(
        X: torch.Tensor,
        Y: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        metric: str = 'l2'
) -> Tuple[torch.Tensor, torch.Tensor]:
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
                print(f"警告: 线性规划求解失败，问题{i}: {e}")
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
                print(f"警告: L1优化失败，问题{i}")
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
        dist_values = torch.sqrt(torch.sum(d_abs ** 2, dim=1))

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

    return X_new, dist_values


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


# 测试函数
def test_batch_modification():
    torch.manual_seed(42)

    # 生成测试数据 (M=3个问题, N=5个点)
    M, N = 3, 5
    X = torch.tensor([
        [0.7645, 0.8300, -0.2343, 0.9186, -0.2191],
        [0.2018, -0.4869, 0.5873, 0.8815, -0.7336],
        [0.8692, 0.1872, 0.7388, 0.1354, 0.4822]
    ], dtype=torch.float32)

    Y = torch.tensor([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ], dtype=torch.float32)

    # 计算初始平均值
    A = X.mean(dim=1, keepdim=True)

    # 设置目标平均值
    B = torch.tensor([
        [0.3139],
        [0.0064],
        [0.3560]
    ], dtype=torch.float32)

    metrics = ['l1', 'l2', 'minmax']

    print(f"初始序列:\n{X}\n")
    print(f"掩码:\n{Y}\n")
    print(f"初始平均值:\n{A}\n目标平均值:\n{B}\n")

    for metric in metrics:
        print(f"\n===== 使用距离度量: {metric} =====")
        X_new, dist_values = batch_modify_sequence(X, Y, A, B, metric)
        new_avg = X_new.mean(dim=1, keepdim=True)

        print(f"修改后序列:\n{X_new}")
        print(f"实际平均值:\n{new_avg}")
        print(f"距离值:\n{dist_values}")
        print(f"平均值误差:\n{new_avg - B}")
        print("-" * 50)


if __name__ == "__main__":
    test_batch_modification()