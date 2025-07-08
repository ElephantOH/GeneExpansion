import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
from scipy import ndimage
from sklearn.decomposition import DictionaryLearning
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

from src.model.sparse_encoder import SparseEncoder


class SparseDict:
    def __init__(self, model_config: DictConfig, device, **kwargs):
        self.config = model_config
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size
        self.dict_size = self.config.dict_size
        self.text_embed_dim = self.config.text_embed_dim
        self.device = device

        # 初始化中心点
        self.center_point = torch.tensor([self.image_size / 2, self.image_size / 2], device=device)

        # 文本编码器
        self.tokenizer = BertTokenizer.from_pretrained(self.config.tokenier)
        self.text_encoder = BertModel.from_pretrained(self.config.text_encoder)
        for param in self.text_encoder.parameters():
            param.requires_grad = self.config.train_text_encoder

        # 空字典和模型
        self.dictionary = None
        self.sparse_constraint = None
        self.sparsity_patterns = []
        self.class_sparsity_profiles = {}

        # 文本降维器
        self.text_reducer = nn.Sequential(
            nn.Linear(self.text_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)

    def get_text_embedding(self, description):
        """获取文本嵌入"""
        with torch.no_grad():
            text_input = self.tokenizer(
                description,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt"
            )


            text_outputs = self.text_encoder(
                input_ids=text_input.input_ids,
                attention_mask=text_input.attention_mask
            )
            text_emb = text_outputs.last_hidden_state
            # text_emb = text_outputs.last_hidden_state[:, 0, :]

        return text_emb

    def analyze_sparsity_pattern(self, image, description=None):
        """分析稀疏性模式（考虑类别）"""
        # 转换为numpy用于传统图像处理
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # 计算中心点
        center_y, center_x = ndimage.center_of_mass(image)
        center = (center_x, center_y)

        # 创建距离映射图
        h, w = image.shape
        y_coords, x_coords = np.indices((h, w))
        chebyshev_dist = np.maximum(np.abs(x_coords - center_x), np.abs(y_coords - center_y))

        # 提取非零像素点的距离分布
        threshold = -1.0
        sparse_points = image > threshold
        dist_values = chebyshev_dist[sparse_points]

        # 统计稀疏性分布
        max_dist = max(center_x, w - center_x, center_y, h - center_y)
        bin_edges = np.linspace(0, max_dist, 20)
        density, _ = np.histogram(dist_values, bins=bin_edges)

        # 分析稀疏模式特征
        density_curve = density / (density.max() + 1e-8)
        peak_positions = self.find_local_peaks(density_curve)
        dead_zones = self.find_sparse_zones(density_curve)

        pattern = {
            "density_curve": density_curve,
            "peak_positions": peak_positions,
            "dead_zones": dead_zones,
            "center": center,
            "max_dist": max_dist
        }

        # 添加类别稀疏性概要
        if description:
            if description not in self.class_sparsity_profiles:
                self.class_sparsity_profiles[description] = []
            self.class_sparsity_profiles[description].append(pattern)

        self.sparsity_patterns.append(pattern)
        return pattern

    def get_class_sparsity_profile(self, description):
        """获取类别稀疏性概要（平均）"""
        if description in self.class_sparsity_profiles:
            profiles = self.class_sparsity_profiles[description]
            avg_density = np.mean([p["density_curve"] for p in profiles], axis=0)
            all_peaks = [p for profile in profiles for p in profile["peak_positions"]]
            all_zones = [z for profile in profiles for z in profile["dead_zones"]]

            # 计算最常见的峰值和稀疏区
            peak_counts = np.bincount(all_peaks, minlength=20)
            zone_counts = np.bincount(all_zones, minlength=20)

            peaks = np.where(peak_counts > len(profiles) / 2)[0].tolist()
            dead_zones = np.where(zone_counts > len(profiles) / 2)[0].tolist()

            return {
                "density_curve": avg_density,
                "peak_positions": peaks,
                "dead_zones": dead_zones
            }

        # 回退到通用概要
        return self.aggregate_pattern

    def find_local_peaks(self, density):
        """识别密度曲线中的局部峰值"""
        peaks = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i - 1] and density[i] > density[i + 1]:
                peaks.append(i)
        return peaks

    def find_sparse_zones(self, density, threshold=0.3):
        """识别稀疏区域"""
        return [i for i, d in enumerate(density) if d < threshold]

    def compute_chebyshev_distance(self, img, center=None):
        """计算图像各点到中心的切比雪夫距离"""
        if center is None:
            center = (img.shape[1] / 2, img.shape[0] / 2)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(img.shape[0], device=self.device),
            torch.arange(img.shape[1], device=self.device),
            indexing='ij'
        )
        return torch.max(
            torch.abs(x_coords - center[0]),
            torch.abs(y_coords - center[1])
        )

    def initialize_radial_dictionary(self, text_emb=None):
        """创建基于放射状基础模板的初始化字典（考虑文本）"""
        # 基础模板：中心块、放射线、过渡模式
        center_patch = torch.zeros(self.patch_size, self.patch_size, device=self.device)
        center_patch[self.patch_size // 2 - 1:self.patch_size // 2 + 2,
        self.patch_size // 2 - 1:self.patch_size // 2 + 2] = 1

        radial_patches = []
        angles = torch.linspace(0, 2 * np.pi, 16, device=self.device)
        for angle in angles:
            patch = torch.zeros(self.patch_size, self.patch_size, device=self.device)
            start_x, start_y = self.patch_size // 2, self.patch_size // 2
            end_x = int(start_x + 5 * torch.cos(angle))
            end_y = int(start_y + 5 * torch.sin(angle))

            # 简化的直线绘制
            xs = torch.linspace(start_x, end_x, 10)
            ys = torch.linspace(start_y, end_y, 10)
            for x, y in zip(xs, ys):
                x, y = int(x), int(y)
                if 0 <= x < self.patch_size and 0 <= y < self.patch_size:
                    patch[y, x] = 1
            radial_patches.append(patch)

        # 组合基础原子
        base_atoms = torch.stack([center_patch] + radial_patches)

        # 添加文本感知的随机变异
        n_random = self.dict_size - len(base_atoms)
        if text_emb is not None:
            # 使用文本嵌入生成更多相关的原子
            text_emb_reduced = self.text_reducer(text_emb)
            text_emb_repeated = text_emb_reduced.unsqueeze(1).unsqueeze(2).repeat(1, self.patch_size, self.patch_size)

            # 生成文本引导的变异
            random_atoms = torch.rand(n_random, self.patch_size, self.patch_size, device=self.device)
            text_guidance = text_emb_repeated.expand(n_random, -1, -1)
            text_guided_atoms = 0.6 * random_atoms + 0.4 * text_guidance
        else:
            # 普通随机原子
            text_guided_atoms = 0.3 * torch.randn(n_random, self.patch_size, self.patch_size, device=self.device)

        return torch.clamp(torch.cat([base_atoms, text_guided_atoms], dim=0), 0, 1)

    def extract_patches(self, image_batch):
        """从图像批次中提取补丁"""
        patches = []
        for img in image_batch:
            # 添加通道维度
            img = img.unsqueeze(0).unsqueeze(0) if img.dim() == 2 else img

            # 展开为补丁
            patches.append(img.unfold(2, self.patch_size, self.patch_size)
                           .unfold(3, self.patch_size, self.patch_size)
                           .permute(0, 2, 3, 1, 4, 5)
                           .reshape(-1, self.patch_size, self.patch_size))
        return torch.cat(patches)

    def train_ksvd_dictionary(self, image_batch, descriptions=None):
        """基于图像和文本描述训练字典"""
        # 分析所有图像的稀疏模式
        patterns = []
        for i, img in enumerate(image_batch):
            desc = descriptions[i] if descriptions else None
            patterns.append(self.analyze_sparsity_pattern(img, desc))

        # 聚合稀疏模式数据
        self.aggregate_pattern = {
            "density_curve": np.mean([p["density_curve"] for p in patterns], axis=0),
            "peak_positions": np.unique(np.concatenate([p["peak_positions"] for p in patterns])),
            "dead_zones": np.unique(np.concatenate([p["dead_zones"] for p in patterns]))
        }

        # 获取文本嵌入p[././
        if descriptions:
            text_embs = self.get_text_embedding(descriptions)
            text_emb_reduced = self.text_reducer(text_embs)
        else:
            text_embs = None
            text_emb_reduced = None

        # 初始化字典（文本感知）
        init_dict = self.initialize_radial_dictionary(text_emb_reduced)

        # 转换为适合sklearn的形状
        patches_flat = self.extract_patches(image_batch)
        patches_flat = patches_flat.reshape(patches_flat.shape[0], -1).cpu().numpy()
        init_dict = init_dict.reshape(init_dict.shape[0], -1).cpu().numpy()

        # 训练字典
        dict_learner = DictionaryLearning(
            n_components=self.dict_size,
            transform_algorithm='omp',
            fit_algorithm='cd',
            dict_init=init_dict,
            random_state=0,
            transform_n_nonzero_coefs=10,
            alpha=0.5,
            n_jobs=-1
        )
        dict_learner.fit(patches_flat)
        dictionary = torch.tensor(dict_learner.components_, device=self.device)
        dictionary = dictionary.reshape(self.dict_size, self.patch_size, self.patch_size)

        self.dictionary = nn.Parameter(dictionary, requires_grad=False)
        return self.dictionary

    def create_sparse_constraint(self):
        """创建自适应稀疏约束层（包含文本感知）"""
        if self.dictionary is None:
            raise RuntimeError("Dictionary must be trained first")

        self.sparse_constraint = SparseConstraint(
            ks_dict=self.dictionary,
            center_point=self.center_point,
            text_reducer=self.text_reducer,
            patch_size=self.patch_size
        ).to(self.device)
        return self.sparse_constraint

    def train_sparse_constraint(self, data_loader, epochs=10, lr=1e-3):
        """训练稀疏约束层（包括特征提取器和预测器）"""
        if self.sparse_constraint is None:
            self.create_sparse_constraint()

        optimizer = torch.optim.Adam(self.sparse_constraint.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.sparse_constraint.train()
            total_loss = 0.0

            for images, masks, descriptions in data_loader:
                images = images.to(self.device)
                masks = masks.to(self.device) if masks is not None else None

                # 获取文本嵌入
                text_embs = self.get_text_embedding(descriptions)

                optimizer.zero_grad()

                # 前向传播
                corrected = self.sparse_constraint(images, text_embs, masks)

                # 计算损失：重建误差 + 稀疏性约束
                recon_loss = criterion(corrected, images)
                sparse_loss = torch.mean(torch.abs(corrected))

                loss = recon_loss + 0.1 * sparse_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    def correct_sparsity(self, x, text_embs, mask=None):
        """应用稀疏度校正（考虑文本类别）"""
        if self.sparse_constraint is None:
            raise RuntimeError("Sparse constraint not initialized")

        with torch.no_grad():
            # 如果text_embs是文本描述列表，转换为嵌入
            if isinstance(text_embs, list):
                text_embs = self.get_text_embedding(text_embs)

            return self.sparse_constraint(x, text_embs, mask=mask)

    def save_weights(self, path):
        """保存模型权重和字典"""
        if self.sparse_constraint is None:
            raise RuntimeError("Model not trained")

        torch.save({
            'dictionary': self.dictionary,
            'text_reducer': self.text_reducer.state_dict(),
            'model_state': self.sparse_constraint.state_dict(),
            'sparsity_pattern': self.aggregate_pattern,
            'class_profiles': self.class_sparsity_profiles,
            'center_point': self.center_point
        }, path)

    def load_weights(self, path):
        """加载模型权重和字典"""
        checkpoint = torch.load(path, map_location=self.device)
        self.dictionary = checkpoint['dictionary'].to(self.device)
        self.center_point = checkpoint['center_point'].to(self.device)
        self.aggregate_pattern = checkpoint['sparsity_pattern']
        self.class_sparsity_profiles = checkpoint.get('class_profiles', {})

        # 加载文本降维器
        if 'text_reducer' in checkpoint:
            self.text_reducer.load_state_dict(checkpoint['text_reducer'])

        # 初始化约束层
        self.sparse_constraint = SparseConstraint(
            ks_dict=self.dictionary,
            center_point=self.center_point,
            text_reducer=self.text_reducer,
            patch_size=self.patch_size
        ).to(self.device)
        self.sparse_constraint.load_state_dict(checkpoint['model_state'])


class SparseConstraint(nn.Module):
    def __init__(self, ks_dict, center_point, text_reducer, patch_size=8, lambda_sparse=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.lambda_sparse = lambda_sparse
        self.register_buffer('dictionary', ks_dict)
        self.register_buffer('center_point', center_point)
        self.text_reducer = text_reducer

        # 视觉特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # 结合文本的稀疏度预测器
        self.sparsity_predictor = nn.Sequential(
            nn.Linear(32 + 1 + 32, 64),  # 视觉特征(32) + 距离(1) + 文本特征(32)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.encoder = SparseEncoder(ks_dict, text_reducer)

    def get_bin_index(self, distance, max_dist, n_bins=20):
        """将距离映射到桶索引"""
        return min(n_bins - 1, int((distance / max_dist) * n_bins))

    def predict_sparsity_level(self, patch, distance, max_dist, text_feature):
        """动态预测每个块的稀疏度（考虑文本）"""
        # 提取视觉特征
        patch = patch.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        visual_features = self.feature_extractor(patch).squeeze()

        # 标准化距离
        normalized_dist = distance / max_dist

        # 组合所有特征
        combined = torch.cat([
            visual_features,
            torch.tensor([normalized_dist], device=patch.device),
            text_feature
        ])

        # 预测稀疏度级别 (3-15范围)
        sparsity = self.sparsity_predictor(combined)
        return torch.clamp(5 * torch.sigmoid(sparsity) + 3, min=3, max=15)

    def forward(self, x, text_embs, mask=None):
        """应用自适应稀疏约束（考虑文本类别）"""
        B, C, H, W = x.shape

        # 处理文本嵌入
        text_features = self.text_reducer(text_embs)  # 降维文本特征
        if len(text_features) != B:
            text_features = text_features.expand(B, -1)  # 确保每个图像有文本特征

        # 获取所有块的位置
        patch_coords_x = torch.arange(0, W, self.patch_size, device=x.device)
        patch_coords_y = torch.arange(0, H, self.patch_size, device=x.device)

        # 计算最大可能距离
        max_dist = torch.max(torch.abs(
            torch.stack([self.center_point[0] - 0, self.center_point[1] - 0,
                         self.center_point[0] - W, self.center_point[1] - H])
        )).item()

        # 处理每个块
        for b in range(B):
            for y in patch_coords_y:
                for x_coord in patch_coords_x:
                    top = y.int().item()
                    left = x_coord.int().item()
                    bottom = min(top + self.patch_size, H)
                    right = min(left + self.patch_size, W)

                    # 跳过不完整块
                    if (bottom - top) < 2 or (right - left) < 2:
                        continue

                    # 提取当前块
                    patch = x[b:b + 1, :, top:bottom, left:right]

                    # 计算块中心到全局中心的距离
                    patch_center = torch.tensor(
                        [left + (right - left) / 2, top + (bottom - top) / 2],
                        device=x.device
                    )
                    distance = torch.max(torch.abs(patch_center - self.center_point)).item()

                    # 动态预测稀疏度（考虑文本）
                    dynamic_sparsity = self.predict_sparsity_level(
                        patch.squeeze(),
                        distance,
                        max_dist,
                        text_features[b]
                    ).item()

                    # 应用稀疏编码（简化版，实际应用中应替换为优化实现）
                    coded_patch = self.sparse_encode(patch, int(dynamic_sparsity))

                    # 更新输出（考虑掩码）
                    if mask is None:
                        x[b:b + 1, :, top:bottom, left:right] = coded_patch
                    else:
                        mask_patch = mask[b:b + 1, :, top:bottom, left:right]
                        update = self.lambda_sparse * (coded_patch - patch) * (1 - mask_patch)
                        x[b:b + 1, :, top:bottom, left:right] += update

        return x

    def sparse_encode(self, patch, n_nonzero):
        """简化的稀疏编码（占位符）"""
        # 在实际系统中应替换为高效的OMP或LARS实现
        return patch