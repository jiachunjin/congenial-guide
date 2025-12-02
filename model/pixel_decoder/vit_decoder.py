import torch
import torch.nn as nn
from model.quantizer.vit_basic import precompute_freqs_cis_2d, Block
from einops.layers.torch import Rearrange


class ViT_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.grid_size = config.grid_size
        self.patch_size = config.patch_size

        self.precompute_pos = dict()
        self.input_proj = nn.Linear(config.input_feature_dim, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList([Block(config.hidden_size, config.num_heads) for _ in range(config.depth)])
        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
            Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=self.patch_size, p2=self.patch_size)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def forward(self, x):
        """
        x: (B, 256, llm_hidden_size)
        """
        B, L, D = x.shape
        pos = self.fetch_pos(self.grid_size, self.grid_size, x.device)

        x = self.input_proj(x)

        x = self.norm1(x).to(x.dtype)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm2(x)
        x = x.permute(0, 2, 1).reshape(B, self.hidden_size, self.grid_size, self.grid_size).contiguous()

        x = self.output_proj(x)
        x = self.conv_out(x)

        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config/vq_distill/pixel_decoder_test.yaml")
    model = ViT_Decoder(config.model.pixel_decoder)

    x = torch.randn(2, 256, 2560)
    out = model(x)
    print(out.shape)