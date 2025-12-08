import torch
import torch.nn as nn

class ParallelHead(nn.Module):
    """
    ParallelHead 将单个 hidden_state 映射到 K 个 vocab 的预测
    
    输入: hidden_state (B, L, D) 或 (B*L, D)
    输出: logits (B, L, K, V) 或 (B*L, K, V)
    其中 K 是 codebook 数量，V 是 vocab size (num_embeddings)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 创建 K 个独立的 Linear 层，每个层将 hidden_state 映射到一个 vocab
        self.heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_embeddings)
            for _ in range(config.num_codebooks)
        ])
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, L, D) 或 (B*L, D) 形状的 hidden states
        
        Returns:
            logits: (B, L, K, V) 或 (B*L, K, V) 形状的 logits
        """
        # 处理输入形状
        if hidden_states.dim() == 2:
            # (B*L, D) -> (B*L, K, V)
            logits_list = []
            for head in self.heads:
                logits_list.append(head(hidden_states))  # (B*L, V)
            logits = torch.stack(logits_list, dim=1)  # (B*L, K, V)
        elif hidden_states.dim() == 3:
            # (B, L, D) -> (B, L, K, V)
            B, L, D = hidden_states.shape
            hidden_states = hidden_states.reshape(B * L, D)  # (B*L, D)
            logits_list = []
            for head in self.heads:
                logits_list.append(head(hidden_states))  # (B*L, V)
            logits = torch.stack(logits_list, dim=1)  # (B*L, K, V)
            logits = logits.reshape(B, L, len(self.heads), -1)  # (B, L, K, V)
        else:
            raise ValueError(f"Expected hidden_states to be 2D or 3D, got {hidden_states.dim()}D")
        
        return logits

