import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer, LlamaRotaryEmbedding

# 如果装上了flash_attention_2，则使用flash_attention_2，否则使用sdpa
try:
    import flash_attn
    ATTN_IMPLEMENTATION = "flash_attention_2"
    print(f"ARHead using flash_attention_2")
except:
    ATTN_IMPLEMENTATION = "sdpa"
    print(f"ARHead using sdpa")

class ARHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.codebooks = nn.ModuleList()
        for _ in range(self.config.num_codebooks - 1):
            codebook = nn.Embedding(self.config.num_embeddings, self.config.hidden_size)
            self.codebooks.append(codebook)

        llama_config = LlamaConfig(
            hidden_size         = self.config.hidden_size,
            num_attention_heads = 32,
            num_key_value_heads = 32,
            rms_norm_eps        = 1e-5,
            attention_dropout   = 0.0,
            attention_bias      = False,
            intermediate_size   = self.config.hidden_size * 4,
            mlp_bias            = False,
            hidden_act          = "silu",
            attn_implementation = ATTN_IMPLEMENTATION,
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(llama_config, layer_idx) for layer_idx in range(3)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_config)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=1e-5)

        self.linear_head = nn.Linear(self.config.hidden_size, self.config.num_embeddings)

    def forward(self, inputs_embeds):
        """
        inputs_embeds: (BxL, K, D)
        logits: (BxL, K, V)
        """
        batch_size, seq_len = inputs_embeds.shape[:2]
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask      = None,
                position_embeddings = position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        logits = self.linear_head(hidden_states)

        return logits


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config/mcq_gen/dev_ar_head.yaml")
    model = ARHead(config.model.ar_head)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    model = model.to(device, dtype)

    B, L, D = 2, 256, 1024
    K = 8

    code = torch.randint(0, 2048, (B, L, K)).to(device, torch.int32)
    code = code.permute(0, 2, 1) # (B, K, L)

    base_tokens = torch.randn(B, L, D).to(device, dtype)
    base_tokens = base_tokens.reshape(B * L, 1, D)

    targets = code.permute(0, 2, 1).reshape(B * L, K)[:, :-1] # (BxL, K-1)
    index_embeddings = []
    for i in range(K - 1):
        index_embed = model.codebooks[i](targets[:, i])
        index_embeddings.append(index_embed)
    index_embeddings = torch.stack(index_embeddings, dim=1)
    print(index_embeddings.shape)
    h = torch.cat((base_tokens, index_embeddings), dim=1)  # [B*L, K, C]
    print(h.shape)

    logits = model(h)
    print(logits.shape)
    logits = logits.reshape(B, L, K, -1).permute(0, 2, 1, 3)  # [B, K, L, sub_vocab_size]
    print(logits.shape)
    loss_fct = torch.nn.CrossEntropyLoss()
    logits = logits.reshape(-1, model.config.num_embeddings)
    print(logits.shape)

    # labels = code.view(-1)
    labels = code.contiguous().view(-1).long()
    # print(labels.shape)
    loss = loss_fct(logits, labels)

    print(loss.item())
