import torch
import torch.nn as nn
import torch.nn.functional as F
from model.quantizer.vq import VectorQuantizer

class MultiVectorQuantizer(nn.Module):
    """
    Multi-Head Vector Quantizer (Product Quantization)
    
    将输入特征维度切分为 num_codebooks 份，每份独立使用一个 VectorQuantizer 进行量化。
    最后将量化后的特征拼接回原始维度。
    
    优点：
    1. 组合爆炸：N个大小为K的codebook可以表达 K^N 种状态。
    2. 计算效率：每个子空间的维度更小，计算距离更快。
    """
    def __init__(
        self, 
        num_embeddings, 
        embedding_dim, 
        num_codebooks, 
        commitment_cost=0.25, 
        ema_decay=0.99,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embedding_dim = embedding_dim
        
        assert embedding_dim % num_codebooks == 0, \
            f"embedding_dim {embedding_dim} must be divisible by num_codebooks {num_codebooks}"
        
        self.dim_per_codebook = embedding_dim // num_codebooks
        
        # 创建多个独立的 VectorQuantizer
        # 使用 nn.ModuleList 注册子模块
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                num_embeddings           = num_embeddings,
                embedding_dim            = self.dim_per_codebook,
                commitment_cost          = commitment_cost,
                ema_decay                = ema_decay,
            )
            for _ in range(num_codebooks)
        ])

    def forward(self, z):
        """
        :param z: (B, L, D) 输入特征
        :return: 
            z_q: (B, L, D) 量化后的特征 (拼接后)
            indices: (B, L, num_codebooks) 每个head的codebook索引
            vq_loss: 标量loss (所有head的平均或总和)
        """
        B, L, D = z.shape
        assert D == self.embedding_dim
        
        # 将特征切分为 (B, L, num_codebooks, dim_per_codebook)
        # 但为了方便分别送入 quantizers，我们先在最后一维 split
        z_splits = torch.split(z, self.dim_per_codebook, dim=-1)
        
        z_q_list = []
        indices_list = []
        total_vq_loss = 0.0
        
        # 对每个切片分别进行量化
        for i, quantizer in enumerate(self.quantizers):
            # z_chunk: (B, L, dim_per_codebook)
            z_chunk = z_splits[i]
            
            # z_q_chunk: (B, L, dim_per_codebook)
            # indices_chunk: (B, L)
            z_q_chunk, indices_chunk, loss_chunk = quantizer(z_chunk)
            
            z_q_list.append(z_q_chunk)
            indices_list.append(indices_chunk)
            total_vq_loss += loss_chunk
            
        # 拼接回原始维度
        z_q = torch.cat(z_q_list, dim=-1) # (B, L, D)
        
        # 堆叠索引 (B, L, num_codebooks)
        indices = torch.stack(indices_list, dim=-1)
        
        # Loss 取平均
        avg_vq_loss = total_vq_loss / self.num_codebooks
        
        return z_q, indices, avg_vq_loss

    def get_codebook_usage(self):
        """
        获取所有 codebook 的平均利用率指标
        """
        total_perplexity = 0.0
        total_utilization = 0.0
        total_active = 0.0
        
        stats_list = []
        
        for i, quantizer in enumerate(self.quantizers):
            stats = quantizer.get_codebook_usage()
            stats_list.append(stats)
            total_perplexity += stats["perplexity"]
            total_utilization += stats["utilization"]
            total_active += stats["active_codes"]
            
        return {
            "perplexity": total_perplexity / self.num_codebooks,
            "utilization": total_utilization / self.num_codebooks,
            "active_codes": total_active / self.num_codebooks,
            "per_codebook_stats": stats_list
        }

# 适配器类，用于替换原本的 VQ_MLP 或 VQ_ViT 中的 quantizer
# 如果你需要直接用 Multi-VQ 替换单 VQ，可以使用类似的 Wrapper

class VQ_MLP_MultiHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 确保配置里有 num_codebooks
        num_codebooks = getattr(config, "num_codebooks", 4) # 默认4个头

        self.down_proj = nn.Sequential(
            nn.Linear(config.input_feature_dim, 4 * config.input_feature_dim),
            nn.GELU(),
            nn.Linear(4 * config.input_feature_dim, config.embedding_dim),
        )

        self.quantizer = MultiVectorQuantizer(
            num_embeddings  = config.num_embeddings,
            embedding_dim   = config.embedding_dim,
            num_codebooks   = num_codebooks,
            commitment_cost = getattr(config, "commitment_cost", 0.25),
        )

        self.up_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.llm_hidden_size, config.llm_hidden_size),
        )

    def forward(self, x):
        z = self.down_proj(x)
        z_q, indices, vq_loss = self.quantizer(z)
        x_vq = self.up_proj(z_q)
        return x_vq, indices, vq_loss

def get_multi_vq_quantizer(config):
    # 这里可以根据 config 返回 MLP 或 ViT 版本的 Multi-VQ
    # 目前先返回 MLP 版本作为示例
    return VQ_MLP_MultiHead(config)
