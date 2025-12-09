import torch
import torch.nn as nn
from einops import rearrange
from .my_ar_head_basic import DecoderLayer, RMSNorm, precompute_freqs_cis_1d

class MyARHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList()
        for _ in range(self.config.num_codebooks):
            embedding = nn.Embedding(self.config.num_embeddings, self.config.hidden_size)
            self.embeddings.append(embedding)
        
        self.layers = nn.ModuleList()
        for _ in range(self.config.num_layers):
            layer = DecoderLayer(self.config)
            self.layers.append(layer)
        
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)
        self.head = nn.Linear(self.config.hidden_size, self.config.num_embeddings)
        
        # 1D RoPE 位置编码缓存
        self.precompute_pos = {}
    
    def get_freqs_cis(self, seq_len: int, device):
        """获取或计算指定长度的位置编码"""
        if seq_len not in self.precompute_pos:
            head_dim = self.config.hidden_size // self.config.num_heads
            freqs_cis = precompute_freqs_cis_1d(head_dim, seq_len)
            self.precompute_pos[seq_len] = freqs_cis
        return self.precompute_pos[seq_len].to(device)

    def forward(self, x):
        """
        x: (B, N, D) input embeddings
        returns: (B, N, V) logits
        """
        B, N, D = x.shape
        pos = self.get_freqs_cis(N, x.device)
        
        for layer in self.layers:
            x = layer(x, pos)
        
        x = self.norm(x)
        logits = self.head(x)

        return logits

    def _code_to_embeddings(self, code):
        """
        code: (B, L, K)
        return: (BxL, K, D)
        """
        B, L, K = code.shape
        code = rearrange(code, "B L K -> (B L) K")
        index_embeddings = []
        for i in range(K):
            index_embed = self.embeddings[i](code[:, i])
            index_embeddings.append(index_embed)
        index_embeddings = torch.stack(index_embeddings, dim=1)

        return index_embeddings


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/mcq_gen/dev_my_ar_head.yaml")

    model = MyARHead(config.model.head)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model = model.to(device, dtype)

    # x = torch.randn(2, 8, 2560).to(device, dtype)
    # logits = model(x)
    # print(logits.shape)

    code = torch.randint(0, 2048, (2, 256, 8)).to(device, torch.int32)
    index_embeddings = model._code_to_embeddings(code)
    print(index_embeddings.shape)