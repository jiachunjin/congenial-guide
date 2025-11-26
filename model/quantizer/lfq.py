import torch
import torch.nn as nn


def get_lfq_quantizer(config):
    if config.quantizer.type == "MLP":
        return LFQ_MLP(config.quantizer)
    elif config.quantizer.type == "ViT":
        raise NotImplementedError
    else:
        raise NotImplementedError


class LFQ_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.down_proj = nn.Sequential(
            nn.Linear(config.input_feature_dim, 4 * config.input_feature_dim),
            nn.GELU(),
            nn.Linear(4 * config.input_feature_dim, config.output_dim),
        )
        self.up_proj = nn.Sequential(
            nn.Linear(config.output_dim, 4 * config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.llm_hidden_size, config.llm_hidden_size),
        )
    
    def forward(self, x):
        """
        :param x: (B, 256, 4096), the input clip features
        """
        feature_low = self.down_proj(x) # (B, 256, output_dim)
        p = torch.sigmoid(feature_low)
        p_ = (p > 0.5).to(x.dtype)
        feature_bin = p + (p_ - p).detach()

        x_vq = self.up_proj(feature_bin) # (B, 256, llm_hidden_size)

        return x_vq, p_