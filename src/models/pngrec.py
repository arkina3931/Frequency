# -*- coding: utf-8 -*-
# Integrated file: PNG graph decomposer + PNGRec recommender.
# Model logic is kept from the uploaded files; only the external PNG import is disabled so PNGRec uses the local PNG class.

# =========================
# Part 1: models/png.py
# =========================

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import dirichlet

try:
    from pytorch_wavelets import DWT1D
except ImportError as e:
    DWT1D = None
    _DWT1D_IMPORT_ERROR = e

class PNG(nn.Module):
    """
    User-Item Popular Niche Interest Graph (UIG)模型
    基于小波变换提取用户-物品交互图中的流行兴趣和小众兴趣，优化内存版本
    所有张量强制为float32类型，避免类型冲突
    """
    def __init__(self, R: torch.Tensor, wavelet='db3', level=3, device=None, batch_size=32):
        super(PNG, self).__init__()
        if DWT1D is None:
            raise ImportError(
                'PNGRec requires pytorch_wavelets. Install it with `pip install pytorch_wavelets`.'
            ) from _DWT1D_IMPORT_ERROR
        device = R.device if device is None else torch.device(device)
        # 强制输入矩阵为float32类型（核心修改）
        self.R = R.coalesce().to(device=device, dtype=torch.float32)
        self.device = device
        self.wavelet = wavelet
        self.level = level
        self.batch_size = int(batch_size)
        self.dtype = torch.float32  # 统一类型标识
        
        # 矩阵维度
        self.num_users = R.size(0)
        self.num_items = R.size(1)
        
        # 小波变换层（共享权重）
        self.dwt = DWT1D(wave=wavelet, J=level).to(device)
        
        # 预计算索引和值（已为float32）
        self.indices = self.R.indices()  # [2, nnz]
        self.values = self.R.values()    # [nnz]（float32）
        
        # 存储中间结果（延迟计算）
        self.item_popularity = None
        self.user_activity = None
        self.user_low = None  # 用户低频系数（大众兴趣）
        self.user_high = None # 用户高频系数（小众兴趣）
        self.item_low = None  # 物品低频系数（大众兴趣）
        self.item_high = None # 物品高频系数（小众兴趣）
        
        # 输出结果
        self.UIPG = None
        self.UING = None
        self.quadrants = None

    def _compute_item_popularity(self):
        """计算物品流行度（强制float32）"""
        if self.item_popularity is None:
            # 初始化流行度向量（显式指定float32）
            pop = torch.zeros(self.num_items, device=self.device, dtype=self.dtype)
            # 按物品索引累加交互值（带对数惩罚）
            counts = torch.bincount(self.indices[1], minlength=self.num_items)
            log_counts = torch.log1p(counts.to(self.dtype))  # 确保log_counts为float32
            pop.scatter_add_(0, self.indices[1], self.values * log_counts[self.indices[1]])
            # 归一化（L2范数）
            self.item_popularity = pop / (torch.norm(pop) + 1e-8)
        return self.item_popularity

    def _compute_user_activity(self):
        """计算用户活跃度（强制float32）"""
        if self.user_activity is None:
            # 初始化活跃度向量（显式指定float32）
            act = torch.zeros(self.num_users, device=self.device, dtype=self.dtype)
            # 依赖物品流行度
            pop = self._compute_item_popularity()
            # 按用户索引累加（除以流行度惩罚）
            pop_vals = pop[self.indices[1]] + 1e-8  # 避免除零
            act.scatter_add_(0, self.indices[0], self.values / torch.log1p(pop_vals))
            # 归一化（L2范数）
            self.user_activity = act / (torch.norm(act) + 1e-8)
        return self.user_activity

    def _get_user_signals(self, user_idx):
        """生成单个用户的交互信号（强制float32）"""
        mask = (self.indices[0] == user_idx)
        item_ids = self.indices[1][mask]
        vals = self.values[mask]
        
        # 构建信号向量（显式指定float32）
        signal = torch.zeros(self.num_items, device=self.device, dtype=self.dtype)
        signal[item_ids] = vals * (1 + self.user_activity[user_idx])
        return signal.unsqueeze(0).unsqueeze(0)  # [1,1,D]

    def _get_item_signals(self, item_idx):
        """生成单个物品的交互信号（强制float32）"""
        mask = (self.indices[1] == item_idx)
        user_ids = self.indices[0][mask]
        vals = self.values[mask]
        
        # 构建信号向量（显式指定float32）
        signal = torch.zeros(self.num_users, device=self.device, dtype=self.dtype)
        signal[user_ids] = vals * (1 + self.item_popularity[item_idx])
        return signal.unsqueeze(0).unsqueeze(0)  # [1,1,D]

    def _wavelet_transform_batch(self, is_user=True, batch_size=None):
        """批量小波变换（强制float32）"""
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        num_entities = self.num_users if is_user else self.num_items
        # 初始化低频/高频张量（显式指定float32）
        low_freq = torch.zeros(num_entities, device=self.device, dtype=self.dtype)
        high_freq = torch.zeros(num_entities, device=self.device, dtype=self.dtype)
        
        # 分批次处理
        for start in range(0, num_entities, batch_size):
            end = min(start + batch_size, num_entities)
            batch_low = []
            batch_high = []
            
            for idx in range(start, end):
                # 获取信号（用户/物品）
                if is_user:
                    signal = self._get_user_signals(idx)
                else:
                    signal = self._get_item_signals(idx)
                
                # 小波变换
                lf, hf_list = self.dwt(signal)
                # 低频系数：全局趋势（取均值）
                batch_low.append(torch.mean(lf).item())  # 转为Python标量，避免类型冲突
                # 高频系数：细节变化（取能量和）
                hf_energy = torch.sum(torch.stack([torch.norm(hf) for hf in hf_list])).item()
                batch_high.append(hf_energy)
            
            # 批量写入结果（确保float32）
            low_freq[start:end] = torch.tensor(batch_low, device=self.device, dtype=self.dtype)
            high_freq[start:end] = torch.tensor(batch_high, device=self.device, dtype=self.dtype)
        
        return low_freq, high_freq

    def _dirichlet_weight(self, low, high):
        """基于Dirichlet分布计算权重（输出float32）"""
        # 确保alpha为正值且为float32
        alpha_low = torch.clamp(low, min=1e-6)
        alpha_high = torch.clamp(high, min=1e-6)
        # 归一化alpha
        alpha_sum = alpha_low + alpha_high
        alpha_low /= alpha_sum
        alpha_high /= alpha_sum
        
        # 批量采样Dirichlet分布
        weights = []
        for a1, a2 in zip(alpha_low.cpu().numpy(), alpha_high.cpu().numpy()):
            weights.append(dirichlet.rvs([a1*10, a2*10], size=1)[0, 0])
        
        # 转换为float32张量
        return torch.tensor(weights, device=self.device, dtype=self.dtype)

    def forward(self):
        """前向传播（所有输出强制为float32）"""
        # 1. 计算基础指标
        self._compute_item_popularity()
        self._compute_user_activity()
        
        # 2. 小波变换（分用户和物品）
        print("PNGRec: processing user signals...")
        self.user_low, self.user_high = self._wavelet_transform_batch(is_user=True)
        print("PNGRec: processing item signals...")
        self.item_low, self.item_high = self._wavelet_transform_batch(is_user=False)
        
        # 3. Dirichlet权重计算
        user_pop_weight = self._dirichlet_weight(self.user_low, self.user_high)  # [U]
        item_pop_weight = self._dirichlet_weight(self.item_low, self.item_high)  # [I]
        user_niche_weight = 1 - user_pop_weight
        item_niche_weight = 1 - item_pop_weight
        
        # 4. 构建兴趣图（稀疏表示，强制float32）
        # 流行兴趣图：用户大众权重 × 物品大众权重 × 交互值
        uipg_vals = self.values * user_pop_weight[self.indices[0]] * item_pop_weight[self.indices[1]]
        self.UIPG = torch.sparse_coo_tensor(
            self.indices, 
            uipg_vals,  # 已为float32
            (self.num_users, self.num_items), 
            device=self.device,
            dtype=self.dtype  # 显式指定为float32
        ).coalesce()
        
        # 小众兴趣图
        uing_vals = self.values * user_niche_weight[self.indices[0]] * item_niche_weight[self.indices[1]]
        self.UING = torch.sparse_coo_tensor(
            self.indices, 
            uing_vals, 
            (self.num_users, self.num_items), 
            device=self.device,
            dtype=self.dtype
        ).coalesce()
        
        # 5. 四象限表征（稀疏存储，float32）
        self.quadrants = {
            'pop_user_pop_item': torch.sparse_coo_tensor(
                self.indices, 
                self.values * user_pop_weight[self.indices[0]] * item_pop_weight[self.indices[1]],
                (self.num_users, self.num_items), 
                device=self.device,
                dtype=self.dtype
            ).coalesce(),
            'pop_user_niche_item': torch.sparse_coo_tensor(
                self.indices, 
                self.values * user_pop_weight[self.indices[0]] * item_niche_weight[self.indices[1]],
                (self.num_users, self.num_items), 
                device=self.device,
                dtype=self.dtype
            ).coalesce(),
            'niche_user_pop_item': torch.sparse_coo_tensor(
                self.indices, 
                self.values * user_niche_weight[self.indices[0]] * item_pop_weight[self.indices[1]],
                (self.num_users, self.num_items), 
                device=self.device,
                dtype=self.dtype
            ).coalesce(),
            'niche_user_niche_item': torch.sparse_coo_tensor(
                self.indices, 
                self.values * user_niche_weight[self.indices[0]] * item_niche_weight[self.indices[1]],
                (self.num_users, self.num_items), 
                device=self.device,
                dtype=self.dtype
            ).coalesce()
        }
        
        return self.UIPG, self.UING, self.quadrants
    
# 测试用例（模拟大规模稀疏矩阵）
def test_uig_memory_efficiency():
    # 生成19445×7050的稀疏矩阵（模拟输入）
    num_users, num_items = 19445, 7050
    nnz = 118551  # 非零元素数量
    
    # 随机生成交互数据
    user_idx = torch.randint(0, num_users, (nnz,))
    item_idx = torch.randint(0, num_items, (nnz,))
    values = torch.rand(nnz) * 0.5  # 交互强度
    
    # 构建稀疏矩阵
    indices = torch.stack([user_idx, item_idx])
    R = torch.sparse_coo_tensor(indices, values, (num_users, num_items), device='cpu')
    
    # 模型测试
    model = UIG(R, wavelet='db3', level=2)
    UIPG, UING, quads = model.forward()
    print("quads:", quads['pop_user_pop_item'].to_dense())
    # 验证结果
    print(f"UIPG 非零元素: {UIPG._nnz()}")
    print(f"UING 非零元素: {UING._nnz()}")
    print(f"四象限非零元素: {[quads[k]._nnz() for k in quads]}")
    
    # 内存使用检查
    print(f"用户低频系数内存: {model.user_low.element_size() * model.user_low.nelement() / 1024 / 1024:.2f} MB")
    print(f"物品高频系数内存: {model.item_high.element_size() * model.item_high.nelement() / 1024 / 1024:.2f} MB")

# if __name__ == "__main__":
#     test_uig_memory_efficiency()

# =========================
# Part 2: models/pngrec.py
# =========================

"""
Author: orangeheyue@gmail
Paper Reference:
	IEEE AAAI 2026: PNGRec: Popular-Niche Wavelet Graph Learning for Multimodal Recommendation
Sourece Code:
	https://github.com/orangeai-research/PNGRec.git
	https://github.com/orangeheyue/PNGRec.git
"""


import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph

# from models.uig import UIG
# from models.UIG import UIG 
# from models.UIG2 import GraphWaveletDecomposer
# from models.UIGV1 import UIG
# from models.UIGV2 import UIG
# from models.UIGV3 import UIG
# from models.mmp import WaveletInterestNet

# from models.png import PNG  # integrated locally above; keep PNG class in this file

class PNGRec(GeneralRecommender):
	def __init__(self, config, dataset):
		super(PNGRec, self).__init__(config, dataset)
		require_both_modalities = config['png_require_both_modalities']
		self.png_require_both_modalities = True if require_both_modalities is None else bool(require_both_modalities)
		if self.v_feat is None or self.t_feat is None:
			message = (
				'PNGRec requires both image and text features. Please check vision_feature_file/text_feature_file '
				'in the dataset config and ensure both .npy files exist.'
			)
			if self.png_require_both_modalities:
				raise ValueError(message)
			raise NotImplementedError(message + ' Single-modality PNGRec fallback is not implemented.')
		self.sparse = True
		self.cl_loss = config['cl_loss']
		self.n_ui_layers = config['n_ui_layers']
		self.embedding_dim = config['embedding_size']
		self.n_layers = config['n_layers']
		self.reg_weight = config['reg_weight']
		self.image_knn_k = config['image_knn_k']
		self.text_knn_k = config['text_knn_k']
		self.dropout_rate = config['dropout_rate']
		self.temperature = config['temperature']
		if isinstance(self.temperature, list):
			self.temperature = self.temperature[0]
		self.temperature = float(self.temperature)
		self.png_wavelet = config['png_wavelet'] or 'db5'
		self.png_level = int(config['png_level'] or 3)
		self.png_graph_batch_size = int(config['png_graph_batch_size'] or 32)
		self.dropout = nn.Dropout(p=self.dropout_rate)

		self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

		self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
		self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
		nn.init.xavier_uniform_(self.user_embedding.weight)
		nn.init.xavier_uniform_(self.item_id_embedding.weight)

		dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
		image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.image_knn_k, self.sparse))
		text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.text_knn_k, self.sparse))

		self.norm_adj = self.get_adj_mat()
		self.R_sprse_mat = self.R
		self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
		self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
		
		# 初始化模型
		# model = UIG(self.R)
		# # 前向传播
		# self.UIPG, self.UING = model.forward()
		model = PNG(
			self.R,
			wavelet=self.png_wavelet,
			level=self.png_level,
			device=self.device,
			batch_size=self.png_graph_batch_size
		)
		self.UIPG, self.UING, self.quads = model.forward()
		# num_users, num_items = self.R.shape[0], self.R.shape[1]
		# model = UIG(self.norm_adj, num_users, num_items)
		# self.UIPG, self.UING, self.UIPPG, self.UIPNG, self.UINPG, self.UINNG = model.forward()
		# UIG_Model = UIG(self.norm_adj, wavelet='haar', level=1)
		# self.UIPG, self.UING = UIG_Model.forward() 
	
		# 初始化分解器
		#decomposer = GraphWaveletDecomposer(wavelet_name='db4', threshold=0.05)
		# 方法1: 基本分解
		#self.UIPG, self.UING  = decomposer.decompose_graph(self.norm_adj, levels=3)
		# 方法2: 自适应分解(考虑用户和物品流行度)
		#self.UIPG, self.UING = decomposer.adaptive_decompose(self.norm_adj, levels=3)
		# model = UIG(self.norm_adj, rank=300)
		# self.UIPG, self.UING = model.forward()

		# 初始化跨模态注意力
		self.cross_mm_attentoin = CrossModalAttention(self.embedding_dim)


		# self.png_model = WaveletInterestNet().to(self.device)

		if self.v_feat is not None:
			self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
			if os.path.exists(image_adj_file):
				image_adj = torch.load(image_adj_file, map_location=self.device)
			else:
				image_adj = build_sim(self.image_embedding.weight.detach())
				image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,
													   norm_type='sym')
				torch.save(image_adj.detach().cpu(), image_adj_file)
			self.image_original_adj = image_adj.to(self.device).coalesce()

		if self.t_feat is not None:
			self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
			if os.path.exists(text_adj_file):
				text_adj = torch.load(text_adj_file, map_location=self.device)
			else:
				text_adj = build_sim(self.text_embedding.weight.detach())
				text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')
				torch.save(text_adj.detach().cpu(), text_adj_file)
			self.text_original_adj = text_adj.to(self.device).coalesce()

		self.fusion_adj = self.max_pool_fusion()

		if self.v_feat is not None:
			self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
		if self.t_feat is not None:
			self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

		self.softmax = nn.Softmax(dim=-1)

		self.query_v = nn.Sequential(
			nn.Linear(self.embedding_dim, self.embedding_dim),
			nn.Tanh(),
			nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
		)
		self.query_t = nn.Sequential(
			nn.Linear(self.embedding_dim, self.embedding_dim),
			nn.Tanh(),
			nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
		)

		self.gate_v = nn.Sequential(
			nn.Linear(self.embedding_dim, self.embedding_dim),
			nn.Sigmoid()
		)

		self.gate_t = nn.Sequential(
			nn.Linear(self.embedding_dim, self.embedding_dim),
			nn.Sigmoid()
		)

		self.gate_f = nn.Sequential(
			nn.Linear(self.embedding_dim, self.embedding_dim),
			nn.Sigmoid()
		)

		self.gate_image_prefer = nn.Sequential(
			nn.Linear(self.embedding_dim, self.embedding_dim),
			nn.Sigmoid()
		)

		self.gate_text_prefer = nn.Sequential(
			nn.Linear(self.embedding_dim, self.embedding_dim),
			nn.Sigmoid()
		)
		self.gate_fusion_prefer = nn.Sequential(
			nn.Linear(self.embedding_dim, self.embedding_dim),
			nn.Sigmoid()
		)

		self.image_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
		self.text_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
		self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
		

	def pre_epoch_processing(self):
		pass

	def max_pool_fusion(self):
		image_adj = self.image_original_adj.coalesce()
		text_adj = self.text_original_adj.coalesce()

		image_indices = image_adj.indices().to(self.device)
		image_values = image_adj.values().to(self.device)
		text_indices = text_adj.indices().to(self.device)
		text_values = text_adj.values().to(self.device)

		combined_indices = torch.cat((image_indices, text_indices), dim=1)
		combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)

		combined_values_image = torch.full(
			(combined_indices.size(1),), float('-inf'), device=self.device, dtype=image_values.dtype
		)
		combined_values_text = torch.full(
			(combined_indices.size(1),), float('-inf'), device=self.device, dtype=text_values.dtype
		)

		combined_values_image[unique_idx[:image_indices.size(1)]] = image_values
		combined_values_text[unique_idx[image_indices.size(1):]] = text_values
		combined_values, _ = torch.max(torch.stack((combined_values_image, combined_values_text)), dim=0)

		fusion_adj = torch.sparse_coo_tensor(
			combined_indices, combined_values, image_adj.size(), device=self.device
		).coalesce()

		return fusion_adj

	def get_adj_mat(self):
		inter_M = self.interaction_matrix.tocoo()
		inter_M_t = inter_M.transpose().tocoo()
		rows = np.concatenate([inter_M.row, inter_M_t.row + self.n_users])
		cols = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
		data = np.ones(rows.shape[0], dtype=np.float32)
		adj_mat = sp.coo_matrix(
			(data, (rows, cols)),
			shape=(self.n_users + self.n_items, self.n_users + self.n_items)
		).tocsr()
		adj_mat.data[:] = 1.0

		rowsum = np.asarray(adj_mat.sum(1)).flatten()
		d_inv = np.power(rowsum, -0.5)
		d_inv[np.isinf(d_inv)] = 0.
		d_mat_inv = sp.diags(d_inv)
		norm_adj_mat = d_mat_inv.dot(adj_mat).dot(d_mat_inv).tocsr()
		self.R = norm_adj_mat[:self.n_users, self.n_users:]
		return norm_adj_mat

	def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
		"""Convert a scipy sparse matrix to a torch sparse tensor."""
		sparse_mx = sparse_mx.tocoo().astype(np.float32)
		indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
		values = torch.from_numpy(sparse_mx.data)
		shape = torch.Size(sparse_mx.shape)
		return torch.sparse_coo_tensor(indices, values, shape).coalesce()

	def spectrum_convolution(self, image_embeds, text_embeds):
		"""
		Modality Denoising & Cross-Modality Fusion
		"""
		image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')           
		text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')

		image_complex_weight = torch.view_as_complex(self.image_complex_weight)   
		text_complex_weight = torch.view_as_complex(self.text_complex_weight)
		fusion_complex_weight = torch.view_as_complex(self.fusion_complex_weight)

		#   Uni-modal Denoising
		image_conv = torch.fft.irfft(image_fft * image_complex_weight, n=image_embeds.shape[1], dim=1, norm='ortho')    
		text_conv = torch.fft.irfft(text_fft * text_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

		#   Cross-modality fusion
		fusion_conv = torch.fft.irfft(text_fft * image_fft * fusion_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho') 
		
		return image_conv, text_conv, fusion_conv
	
	# # 轻量优化版本
	# def forward_ui_gcn(self, adj):
	# 	item_embeds = self.item_id_embedding.weight
	# 	user_embeds = self.user_embedding.weight
	# 	# 初始嵌入（用户+物品）
	# 	ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
	# 	ego_embeddings2 = torch.cat([user_embeds, item_embeds], dim=0)
	# 	ego_embeddings3 = torch.cat([user_embeds, item_embeds], dim=0)
	# 	all_embeddings = [ego_embeddings]
		
	# 	# 合并邻接矩阵（确保在同一设备）
	# 	adj = adj 
	# 	adj = adj.to(ego_embeddings.device)  # 显式指定邻接矩阵设备
		
	# 	# 可学习的层权重（关键修复：创建时就放到嵌入所在设备）
	# 	if not hasattr(self, 'layer_weights'):  # 避免重复初始化
	# 		self.layer_weights = nn.Parameter(
	# 			torch.ones(self.n_ui_layers + 1, device=ego_embeddings.device)  # 直接在GPU上创建
	# 		)
		
	# 	# GCN消息传递
	# 	for i in range(self.n_ui_layers):
	# 		ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
	# 		all_embeddings.append(ego_embeddings)
	# 	for i in range(self.n_ui_layers):
	# 		ego_embeddings2 = torch.sparse.mm(self.UIPG, ego_embeddings2)
	# 		all_embeddings.append(ego_embeddings2)
	# 	for i in range(self.n_ui_layers):
	# 		ego_embeddings3 = torch.sparse.mm(self.UING, ego_embeddings3)
	# 		all_embeddings.append(ego_embeddings3)

		
		
	# 	# 动态加权融合（确保所有张量在同一设备）
	# 	all_embeddings = torch.stack(all_embeddings, dim=1)  # [N, L+1, D]
	# 	attn = torch.softmax(self.layer_weights, dim=0)  # 层权重归一化（已在GPU）
	# 	# 扩展维度时保持设备一致
	# 	attn = attn.unsqueeze(0).unsqueeze(-1)  # [1, L+1, 1]
	# 	content_embeds = torch.sum(all_embeddings * attn, dim=1)  # [N, D]
		
	# 	return content_embeds
	
	def forward_ui_gcn(self, adj):
		item_embeds = self.item_id_embedding.weight
		user_embeds = self.user_embedding.weight
		# 初始嵌入（用户+物品）
		ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
		ego_embeddings2 = ego_embeddings.clone()  # 使用克隆而不是重复拼接
		ego_embeddings3 = ego_embeddings.clone()
		
		# 为三个GCN分别创建嵌入列表
		all_embeddings1 = [ego_embeddings]
		all_embeddings2 = [ego_embeddings2]
		all_embeddings3 = [ego_embeddings3]
		
		# 合并邻接矩阵（确保在同一设备）
		adj = adj.to(ego_embeddings.device)  # 显式指定邻接矩阵设备
		
		# 可学习的层权重（关键修复：调整维度以匹配总层数）
		total_layers = (self.n_ui_layers + 1) * 3  # 三个GCN，每个有n_ui_layers+1层
		if not hasattr(self, 'layer_weights') or self.layer_weights.size(0) != total_layers:
			self.layer_weights = nn.Parameter(
				torch.ones(total_layers, device=ego_embeddings.device)
			)
		
		# GCN消息传递
		for i in range(self.n_ui_layers):
			ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
			all_embeddings1.append(ego_embeddings)
		
		for i in range(self.n_ui_layers):
			ego_embeddings2 = torch.sparse.mm(self.UIPG, ego_embeddings2)
			all_embeddings2.append(ego_embeddings2)
		
		for i in range(self.n_ui_layers):
			ego_embeddings3 = torch.sparse.mm(self.UING, ego_embeddings3)
			all_embeddings3.append(ego_embeddings3)
		
		# 合并所有嵌入
		all_embeddings = all_embeddings1 + all_embeddings2 + all_embeddings3
		
		# 动态加权融合（确保所有张量在同一设备）
		all_embeddings = torch.stack(all_embeddings, dim=1)  # [N, total_layers, D]
		attn = torch.softmax(self.layer_weights, dim=0)  # 层权重归一化
		attn = attn.unsqueeze(0).unsqueeze(-1)  # [1, total_layers, 1]
		content_embeds = torch.sum(all_embeddings * attn, dim=1)  # [N, D]
		
		return content_embeds

	def user_item_gcn_layer(self,
							adj: torch.Tensor, 
							n_ui_layers: int) -> torch.Tensor:
		"""
		用户-物品行为视图的嵌入传播函数
		
		参数:
			user_embedding: 用户ID嵌入层 (num_users x embed_dim)
			item_id_embedding: 物品ID嵌入层 (num_items x embed_dim)
			adj: 归一化的用户-物品邻接矩阵 (sparse tensor, (num_users + num_items) x (num_users + num_items))
			n_ui_layers: 图卷积层数
			
		返回:
			content_embeds: 传播后的融合嵌入 (num_users + num_items) x embed_dim)
		"""
		# 获取初始嵌入
		item_embeds = self.item_id_embedding.weight
		user_embeds = self.user_embedding.weight
		
		# 拼接用户和物品嵌入
		ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
		all_embeddings = [ego_embeddings]  # 存储各层嵌入
		
		# 多层图传播
		for _ in range(n_ui_layers):
			side_embeddings = torch.sparse.mm(adj, ego_embeddings)
			ego_embeddings = side_embeddings
			all_embeddings.append(ego_embeddings)
		
		# 合并各层嵌入
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
		content_embeds = all_embeddings
		
		return content_embeds

	def item_item_gcn_layer(self, original_adj, item_embeds, sparse, n_layers, R):
		if sparse:
			for i in range(n_layers):
				item_embeds = torch.sparse.mm(original_adj, item_embeds)
		else:
			for i in range(n_layers):
				item_embeds = torch.mm(original_adj, item_embeds)
		user_embeds = torch.sparse.mm(R, item_embeds)
		embeds = torch.cat([user_embeds, item_embeds], dim=0)
		return embeds

	def user_popular_niche_gcn_layer(self, R, item_embeds): 
		'''
			R: 用户-物品邻接矩阵 NxM
			item_embeds: 物品嵌入 MxD 
			n_layers: 图卷积层数
			返回: 用户-物品流行兴趣嵌入 NxD
			注意: 该函数用于计算用户对物品的流行兴趣,通过多层图卷积传播物品嵌入到用户最终得到用户对物品的流行兴趣嵌入
			fusion_user_embeds_UIPG = torch.sparse.mm(self.UIPG, fusion_item_embeds) #  torch.Size([19445, 7050]) x torch.Size([7050, 64]) -> torch.Size([19445, 64])
			fusion_embeds_UIPG = torch.cat([fusion_user_embeds_UIPG, fusion_item_embeds], dim=0) # fusion_embeds_uipg.shape: torch.Size([19445, 64]) + torch.Size([7050, 64]) -
		'''

		# for i in range(n_layers):
		# 	item_embeds = torch.sparse.mm(R, item_embeds)
		user_embeds = torch.sparse.mm(R, item_embeds)
		embeds = torch.cat([user_embeds, item_embeds], dim=0)
		return embeds

	def forward(self, adj, train=False):
		if self.v_feat is not None:
			image_feats = self.image_trs(self.image_embedding.weight)
		if self.t_feat is not None:
			text_feats = self.text_trs(self.text_embedding.weight)

		#   Spectrum Modality Fusion
		# image_conv, text_conv, fusion_conv = self.spectrum_convolution(image_feats, text_feats)
		image_conv, text_conv = image_feats, text_feats 
		# fusion_conv = torch.sqrt(image_conv *image_conv +  text_conv * text_conv) # 融合视图
		fusion_conv = self.cross_mm_attentoin(image_conv, text_conv)
		image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_conv))
		text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_conv))
		fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))

		# User-Item (Behavioral) View
		# item_embeds = self.item_id_embedding.weight
		# user_embeds = self.user_embedding.weight
		# ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
		# all_embeddings = [ego_embeddings]

		# for i in range(self.n_ui_layers):
		# 	side_embeddings = torch.sparse.mm(adj, ego_embeddings)
		# 	ego_embeddings = side_embeddings
		# 	all_embeddings += [ego_embeddings]
		# all_embeddings = torch.stack(all_embeddings, dim=1)
		# all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
		# content_embeds = all_embeddings

		# content_embeds_UIPG = self.forward_ui_gcn(self.UIPG)
		# content_embeds_UING = self.forward_ui_gcn(self.UING)

		# User-Item (Behavioral) View
		content_embeds = self.user_item_gcn_layer(adj=adj, n_ui_layers=self.n_ui_layers) # -> torch.Size([26495, 64])
		image_embeds = self.item_item_gcn_layer(self.image_original_adj, image_item_embeds, self.sparse, self.n_layers, self.R) # -> torch.Size([26495, 64])
		text_embeds = self.item_item_gcn_layer(self.text_original_adj, text_item_embeds, self.sparse, self.n_layers, self.R) #-> torch.Size([26495, 64])
		fusion_embeds = self.item_item_gcn_layer(self.fusion_adj, fusion_item_embeds, self.sparse, self.n_layers, self.R) #-> torch.Size([26495, 64])

		popular_behavior_embeds = self.user_popular_niche_gcn_layer(self.UIPG, fusion_item_embeds) # -> torch.Size([26495, 64]) # 大众兴趣嵌入
		niche_behavior_embeds = self.user_popular_niche_gcn_layer(self.UING, fusion_item_embeds) # -> torch.Size([26495, 64])   # 小众兴趣嵌入

		# 'pop_user_pop_item'  'pop_user_niche_item'  'niche_user_pop_item' 'niche_user_niche_item'
		pop_user_pop_item_embeds = self.user_popular_niche_gcn_layer(self.quads['pop_user_pop_item'] , fusion_item_embeds)  # torch.Size([19445, 64])
		pop_user_niche_item_embeds = self.user_popular_niche_gcn_layer(self.quads['pop_user_niche_item'] , fusion_item_embeds)  # torch.Size([19445, 64])
		niche_user_pop_item_embeds = self.user_popular_niche_gcn_layer(self.quads['niche_user_pop_item']  , fusion_item_embeds) # torch.Size([19445, 64])
		niche_user_niche_item_embeds = self.user_popular_niche_gcn_layer(self.quads['niche_user_niche_item'] , fusion_item_embeds) # torch.Size([19445, 64])

		popular_behavior_embeds = (popular_behavior_embeds - niche_user_pop_item_embeds - niche_user_niche_item_embeds) 
		niche_behavior_embeds = (niche_behavior_embeds - pop_user_pop_item_embeds - pop_user_niche_item_embeds) 


		#content_embeds = (content_embeds + content_embeds_UIPG +  content_embeds_UING)/ 3
		# behavior_embeds = self.png_model(content_embeds) 
		# popular_behavior_embeds = behavior_embeds['mass_emb']  # 大众兴趣嵌入
		# niche_behavior_embeds = behavior_embeds['niche_emb']   	

		# content_embeds_UIPG = self.propagate_user_item_behavior(adj=self.UIPG, n_ui_layers=1)
		# content_embeds_UING = self.propagate_user_item_behavior(adj=self.UING, n_ui_layers=1)

		#   Item-Item Modality Specific and Fusion views GCN Layer
		#   Image-view
		# if self.sparse:
		# 	for i in range(self.n_layers):
		# 		image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
		# else:
		# 	for i in range(self.n_layers):
		# 		image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
		# image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
		# image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

		#   Text-view
		# if self.sparse:
		# 	for i in range(self.n_layers):
		# 		text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
		# else:
		# 	for i in range(self.n_layers):
		# 		text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
		# text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
		# text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

		#   Fusion-view
		# if self.sparse:
		# 	for i in range(self.n_layers):
		# 		fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
		# else:
		# 	for i in range(self.n_layers):
		# 		fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
		# fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
		# fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)


		USE_Fusion_UIPG = False
		# 四象限非零元素: [torch.Size([19445, 7050]), torch.Size([19445, 7050]), torch.Size([19445, 7050]), torch.Size([19445, 7050])]
		if USE_Fusion_UIPG:
			# print("UIPG 类型:", self.UIPG.dtype)
			# print("fusion_item_embeds 类型:", fusion_item_embeds.dtype)
			fusion_user_embeds_UIPG = torch.sparse.mm(self.UIPG, fusion_item_embeds) #  torch.Size([19445, 7050]) x torch.Size([7050, 64]) -> torch.Size([19445, 64])
			fusion_embeds_UIPG = torch.cat([fusion_user_embeds_UIPG, fusion_item_embeds], dim=0) # fusion_embeds_uipg.shape: torch.Size([19445, 64]) + torch.Size([7050, 64]) -> torch.Size([26495, 64])
			fusion_user_embeds_UING = torch.sparse.mm(self.UING, fusion_item_embeds)
			fusion_embeds_UING = torch.cat([fusion_user_embeds_UING, fusion_item_embeds], dim=0)
			fusion_embeds = fusion_embeds * fusion_embeds_UIPG * fusion_embeds_UING + fusion_embeds # + is not good

		#   Modality-aware Preference Module
		fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
		fusion_soft_v = self.softmax(fusion_att_v)
		agg_image_embeds = fusion_soft_v * image_embeds

		fusion_soft_t = self.softmax(fusion_att_t)
		agg_text_embeds = fusion_soft_t * text_embeds

		image_prefer = self.gate_image_prefer(content_embeds)
		text_prefer = self.gate_text_prefer(content_embeds)
		fusion_prefer = self.gate_fusion_prefer(content_embeds)

		# popular_image_prefer = self.gate_image_prefer(popular_behavior_embeds)
		# popular_text_prefer = self.gate_text_prefer(popular_behavior_embeds) 
		# niche_image_prefer = self.gate_image_prefer(niche_behavior_embeds)
		# niche_text_prefer = self.gate_text_prefer(niche_behavior_embeds)
		popular_fusion_prefer = self.gate_fusion_prefer(popular_behavior_embeds)
		niche_fusion_prefer = self.gate_fusion_prefer(niche_behavior_embeds)

		# niche_fusion_prefer = self.gate_text_prefer(niche_behavior_embeds)


		# image_prefer, text_prefer, fusion_prefer = self.dropout(image_prefer), self.dropout(text_prefer), self.dropout(fusion_prefer)
		
		# fusion_prefer_UIPG = self.gate_fusion_prefer(content_embeds_UIPG)
		# fusion_prefer_UING = self.gate_fusion_prefer(content_embeds_UING)		
		#fusion_prefer_UIPG, fusion_prefer_UING = self.dropout(fusion_prefer_UIPG), self.dropout(fusion_prefer_UING)

		agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
		agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)
		fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds)

		popular_fusion_embeds = torch.multiply(popular_fusion_prefer, fusion_embeds)
		niche_fusion_embeds = torch.multiply(niche_fusion_prefer, fusion_embeds)
		# popular_image_embeds = torch.multiply(popular_image_prefer, agg_image_embeds)
		# popular_text_embeds = torch.multiply(popular_text_prefer, agg_text_embeds)
		# niche_image_embeds = torch.multiply(niche_image_prefer, agg_image_embeds)
		# niche_text_embeds = torch.multiply(niche_text_prefer, agg_text_embeds)
		# fusion_embeds_UIPG = torch.multiply(fusion_prefer_UIPG, content_embeds_UIPG)
		# fusion_embeds_UING = torch.multiply(fusion_prefer_UING, content_embeds_UING)

		# side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0) 
		USE_PNG = True
		if USE_PNG:
			side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds, popular_fusion_embeds, niche_fusion_embeds]), dim=0) 
		else:
			side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0) 
		# side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds, fusion_embeds_UIPG, fusion_embeds_UING]), dim=0) 

		all_embeds = content_embeds + side_embeds

		all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

		if train:
			return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds, popular_fusion_embeds, niche_fusion_embeds

		return all_embeddings_users, all_embeddings_items

	def bpr_loss(self, users, pos_items, neg_items):
		pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
		neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

		regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
		regularizer = regularizer / self.batch_size

		maxi = F.logsigmoid(pos_scores - neg_scores)
		mf_loss = -torch.mean(maxi)

		emb_loss = self.reg_weight * regularizer
		reg_loss = 0.0
		return mf_loss, emb_loss, reg_loss

	def InfoNCE(self, view1, view2, temperature):
		view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
		pos_score = (view1 * view2).sum(dim=-1)
		pos_score = torch.exp(pos_score / temperature)
		ttl_score = torch.matmul(view1, view2.transpose(0, 1))
		ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
		cl_loss = -torch.log(pos_score / ttl_score)
		return torch.mean(cl_loss)

	def calculate_loss(self, interaction):
		users = interaction[0]
		pos_items = interaction[1]
		neg_items = interaction[2]

		ua_embeddings, ia_embeddings, side_embeds, content_embeds, popular_fusion_embeds, niche_fusion_embeds = self.forward(
			self.norm_adj, train=True)

		u_g_embeddings = ua_embeddings[users]
		pos_i_g_embeddings = ia_embeddings[pos_items]
		neg_i_g_embeddings = ia_embeddings[neg_items]

		batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
																	  neg_i_g_embeddings)

		side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
		content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
		cl_loss = self.InfoNCE(
			side_embeds_items[pos_items], content_embeds_items[pos_items], self.temperature
		) + self.InfoNCE(side_embeds_users[users], content_embeds_user[users], self.temperature)


		# 2. 大众兴趣融合表征对比（popular_fusion_embeds）
		popular_users, popular_items = torch.split(
			popular_fusion_embeds, [self.n_users, self.n_items], dim=0
		)
		# 大众兴趣：结构融合表征 vs 内容模态表征（强化通用特征一致性）
		cl_popular_item = self.InfoNCE(popular_items[pos_items], content_embeds_items[pos_items], temperature=self.temperature)
		cl_popular_user = self.InfoNCE(popular_users[users], content_embeds_user[users], temperature=self.temperature)
		cl_popular_loss = cl_popular_item + cl_popular_user

		# 3. 小众兴趣融合表征对比（niche_fusion_embeds）
		niche_users, niche_items = torch.split(
			niche_fusion_embeds, [self.n_users, self.n_items], dim=0
		)
		# 小众兴趣：结构融合表征 vs 内容模态表征（强化独特特征一致性，用更低温度）
		niche_temperature = max(self.temperature * 0.5, 1e-8)
		cl_niche_item = self.InfoNCE(niche_items[pos_items], content_embeds_items[pos_items], temperature=niche_temperature)
		cl_niche_user = self.InfoNCE(niche_users[users], content_embeds_user[users], temperature=niche_temperature)
		cl_niche_loss = cl_niche_item + cl_niche_user

		# 4. 跨层次兴趣对比（大众 vs 小众，强化差异）
		# 同一物品的大众表征与小众表征应保持差异
		cl_cross_item = self.InfoNCE(popular_items[pos_items], niche_items[pos_items], temperature=self.temperature)
		# 同一用户的大众表征与小众表征应保持差异
		cl_cross_user = self.InfoNCE(popular_users[users], niche_users[users], temperature=self.temperature)
		cl_cross_loss = cl_cross_item + cl_cross_user

		# 总损失：基础损失 + 各类对比损失（权重可调整）
		total_loss = (
			batch_mf_loss + batch_emb_loss + batch_reg_loss +
			self.cl_loss * cl_loss +  # 原始对比损失
			0.001 * cl_popular_loss +  # 大众兴趣对比
			0.0005 * cl_niche_loss +  # 小众兴趣对比
			0.0001 * cl_cross_loss  # 跨层次差异对比
		)

		USE_CL = True
		if USE_CL:
			return total_loss
		else:
			return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss 
	
	def predict(self, interaction):
		user = interaction[0]
		item = interaction[1]
		restore_user_e, restore_item_e = self.forward(self.norm_adj)
		u_embeddings = restore_user_e[user]
		i_embeddings = restore_item_e[item]
		return torch.mul(u_embeddings, i_embeddings).sum(dim=1)

	def full_sort_predict(self, interaction):
		user = interaction[0]

		restore_user_e, restore_item_e = self.forward(self.norm_adj)
		u_embeddings = restore_user_e[user]

		# dot with all item embedding to accelerate
		scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
		return scores

class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)  # 以文本为查询
        self.key = nn.Linear(dim, dim)    # 图像作为键
        self.value = nn.Linear(dim, dim)  # 图像作为值
    
    def forward(self, text_feat, image_feat):
        q = self.query(text_feat)  # [N, D]
        k = self.key(image_feat).transpose(0, 1)  # [D, N]
        attn = F.softmax(torch.matmul(q, k) / math.sqrt(q.shape[1]), dim=1)  # [N, N]
        # 用文本查询聚焦图像特征，再与文本融合
        image_focused = torch.matmul(attn, self.value(image_feat))  # [N, D]
        return (text_feat + image_focused) / 2  # 融合结果
