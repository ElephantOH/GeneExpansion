import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from scipy import ndimage
from sklearn.decomposition import DictionaryLearning
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

import torch
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from joblib import Parallel, delayed

class SparseDictionary:
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
        self.tokenizer = BertTokenizer.from_pretrained(self.config.tokenizer)
        self.text_encoder = BertModel.from_pretrained(self.config.text_encoder).to(self.device)
        for param in self.text_encoder.parameters():
            param.requires_grad = self.config.train_text_encoder

        # 空字典和模型
        self.dictionary = None
        self.sparse_constraint = None
        self.sparsity_patterns = []
        self.class_sparsity_profiles = {}

        # 文本降维器 (修复输出维度问题)
        self.text_reducer = nn.Sequential(
            nn.Linear(self.text_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)

    def get_text_embedding(self, input_ids, attention_mask):
        """从tokenized输入获取文本嵌入（修复维度问题）"""
        with torch.no_grad():
            # 将输入移到设备
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # 获取BERT嵌入
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 使用CLS标记作为句子表示
            text_emb = text_outputs.last_hidden_state[:, 0, :]
        return text_emb  # [batch_size, 768]

    def analyze_sparsity_pattern(self, image, description=None):
        """分析稀疏性模式（处理单通道图像）"""
        # 转换为numpy用于传统图像处理
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()  # 移除通道维度 [180, 180]

        # 计算中心点
        center_y, center_x = ndimage.center_of_mass(image)
        center = (center_x, center_y)

        # 创建距离映射图
        h, w = image.shape
        y_coords, x_coords = np.indices((h, w))
        chebyshev_dist = np.maximum(np.abs(x_coords - center_x), np.abs(y_coords - center_y))

        # 提取非零像素点的距离分布
        threshold = np.percentile(image, 80)  # 自适应阈值
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
        """创建基于放射状基础模板的初始化字典"""
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

        # 添加随机原子
        n_random = self.dict_size - len(base_atoms)
        random_atoms = torch.rand(n_random, self.patch_size, self.patch_size, device=self.device)

        # 结合文本嵌入 (如果提供)
        if text_emb is not None:
            text_emb = text_emb.mean(dim=0)  # 平均批次的文本嵌入
            text_emb = text_emb[:min(text_emb.size(0), n_random * self.patch_size * self.patch_size)]
            text_shaped = text_emb.view(n_random, self.patch_size, self.patch_size)
            random_atoms = 0.7 * random_atoms + 0.3 * text_shaped

        return torch.clamp(torch.cat([base_atoms, random_atoms], dim=0), 0, 1)

    def extract_patches(self, image_batch):
        """从图像批次中提取补丁 (处理批量和通道维度)"""
        # image_batch: [batch_size, 1, H, W]
        patches = []
        for img in image_batch:
            # 移除通道维度
            img = img.squeeze(0) if img.dim() == 3 else img.squeeze(1)  # [H, W]

            # 展开为补丁
            img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            unfolded = img.unfold(2, self.patch_size, self.patch_size) \
                .unfold(3, self.patch_size, self.patch_size) \
                .permute(0, 2, 3, 1, 4, 5) \
                .reshape(-1, self.patch_size, self.patch_size)
            patches.append(unfolded)

        return torch.cat(patches, dim=0)  # [total_patches, patch_size, patch_size]

    def train_ksvd_dictionary(self, image_batch, descriptions, text_input_ids, text_attention_mask):
        """训练字典 (修复文本维度)"""
        # 1. 分析稀疏模式
        patterns = []
        for i, img in enumerate(image_batch):
            desc = descriptions[i] if descriptions else None
            patterns.append(self.analyze_sparsity_pattern(img.cpu(), desc))

        # 聚合稀疏模式
        self.aggregate_pattern = {
            "density_curve": np.mean([p["density_curve"] for p in patterns], axis=0),
            "peak_positions": np.unique(np.concatenate([p["peak_positions"] for p in patterns])),
            "dead_zones": np.unique(np.concatenate([p["dead_zones"] for p in patterns]))
        }

        # 2. 获取文本嵌入 (修复维度问题)
        text_embs = self.get_text_embedding(text_input_ids, text_attention_mask)  # [batch_size, 768]
        text_emb_reduced = self.text_reducer(text_embs)  # [batch_size, 64]

        # 3. 初始化字典
        init_dict = self.initialize_radial_dictionary(text_emb_reduced)

        # 4. 提取补丁
        patches_flat = self.extract_patches(image_batch)
        patches_flat = patches_flat.reshape(patches_flat.shape[0], -1).cpu().numpy()
        init_dict = init_dict.reshape(init_dict.shape[0], -1).cpu().numpy()

        # 5. 训练字典
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

    def save_dictionary(self, path):
        """保存训练好的字典"""
        torch.save(self.dictionary.cpu(), path)



class LargeScaleSparseDictionary(SparseDictionary):
    def __init__(self, model_config, device, n_jobs=-1):
        super().__init__(model_config, device)
        self.n_jobs = n_jobs  # 并行核数
        self.batch_size = 50000  # 批次处理补丁数

    def extract_patches_parallel(self, image_batch):
        """并行提取补丁优化"""

        def process_image(img):
            img = img.squeeze()
            patches = img.unfold(0, self.patch_size, self.patch_size) \
                .unfold(1, self.patch_size, self.patch_size) \
                .reshape(-1, self.patch_size, self.patch_size)
            return patches

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_image)(img.cpu().numpy()) for img in image_batch
        )
        return torch.tensor(np.vstack(results), device=self.device)

    def train_minibatch_ksvd(self, train_loader):
        """增量式字典训练优化"""
        # 初始化字典
        text_emb_reduced = self.compute_mean_text_embedding(train_loader)
        init_dict = self.initialize_radial_dictionary(text_emb_reduced)
        dict_components = init_dict.cpu().numpy().reshape(self.dict_size, -1)

        # 初始化在线学习器
        dict_learner = MiniBatchDictionaryLearning(
            n_components=self.dict_size,
            batch_size=4096,  # GPU友好的批次大小
            n_iter=50,  # 迭代次数减少但增加通过次数
            dict_init=dict_components,
            transform_algorithm='omp',
            transform_n_nonzero_coefs=10,
            n_jobs=self.n_jobs,
            verbose=True
        )

        # 增量训练循环
        total_patches = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch['gene_matrix'].to(self.device)
            patches = self.extract_patches(images)
            patches_flat = patches.reshape(-1, 81).cpu().numpy()

            # 分批处理避免内存溢出
            for i in range(0, len(patches_flat), self.batch_size):
                batch_patches = patches_flat[i:i + self.batch_size]
                dict_learner.partial_fit(batch_patches)

            total_patches += len(patches_flat)
            print(f"Processed {total_patches / 1e6:.2f}M patches")

        # 最终训练结果
        dictionary = torch.tensor(dict_learner.components_, device=self.device)
        self.dictionary = nn.Parameter(dictionary.reshape(
            self.dict_size, self.patch_size, self.patch_size),
            requires_grad=False
        )
        return self.dictionary

    def compute_mean_text_embedding(self, train_loader):
        """计算类别文本嵌入的均值"""
        text_embs = []
        for batch in tqdm(train_loader):
            input_ids = batch['text_input_ids'].to(self.device)
            attn_mask = batch['text_attention_mask'].to(self.device)
            embs = self.get_text_embedding(input_ids, attn_mask)
            text_embs.append(self.text_reducer(embs))

        return torch.mean(torch.cat(text_embs), dim=0, keepdim=True)