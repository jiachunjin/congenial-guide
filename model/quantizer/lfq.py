import torch
import torch.nn as nn
from model.quantizer.vit import ViT

def get_lfq_quantizer(config):
    if config.type == "MLP":
        return LFQ_MLP(config)
    elif config.type == "ViT":
        return LFQ_ViT(config)
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

class LFQ_ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.down_proj = ViT(config) # 4096 -> output_dim

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

    @torch.no_grad()
    def get_bin_code(self, x):
        feature_low = self.down_proj(x)
        p = torch.sigmoid(feature_low)
        code = (p > 0.5).to(x.dtype)

        return code

    @torch.no_grad()
    def get_ids(self, x):
        raise NotImplementedError


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config/vq_distill/distill_vit.yaml")

    model = get_lfq_quantizer(config.model.quantizer)
    x = torch.randn(2, 256, 4096)
    x_vq, p_ = model(x)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:.2f}M")
    print(x_vq.shape, p_.shape)