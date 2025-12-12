import torch
import torch.nn as nn
import copy
from typing import Optional
from einops import rearrange


def create_moe_forward(original_forward):
    """
    创建新的 forward 方法，根据 vision_token_mask 选择使用 mlp 还是 mlp_vision
    vision_token_mask 通过 kwargs 传入
    """
    def moe_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 从 kwargs 获取 vision_token_mask
        vision_token_mask = kwargs.get('vision_token_mask', None)
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected with MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 检查是否有 vision_token_mask 并且有 mlp_vision
        if hasattr(self, 'mlp_vision') and vision_token_mask is not None:
            # 获取 mask: (B, N) -> (B, N, 1)
            mask = vision_token_mask.unsqueeze(-1).to(hidden_states.dtype)
            
            # 分别计算两个 MLP 的输出
            mlp_output = self.mlp(hidden_states)
            mlp_vision_output = self.mlp_vision(hidden_states)
            
            # 根据 mask 混合输出：vision token 用 mlp_vision，text token 用 mlp
            hidden_states = mask * mlp_vision_output + (1 - mask) * mlp_output
        else:
            # 没有 mask 时使用原来的 mlp
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        return hidden_states
    
    return moe_forward


def make_internvl_moe(internvl):
    """
    将 pretrained InternVL 转换为 MoE 模型：
    1. 给每个 Qwen3DecoderLayer 添加一个 mlp_vision（与原本的 mlp 结构相同的 Qwen3MLP）
    2. 冻结所有原有参数，只有新添加的 mlp_vision 可训练
    3. 修改 forward 逻辑：视觉 token 使用 mlp_vision，文本 token 使用 mlp
    
    使用方式：在调用 language_model 时传入 vision_token_mask 参数
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    import types
    
    # 首先冻结所有原有参数
    for param in internvl.parameters():
        param.requires_grad = False
    
    # 获取 LLM 的配置（用于创建新的 Qwen3MLP）
    llm_config = internvl.language_model.config
    
    # 获取所有的 decoder layers
    layers = internvl.language_model.model.layers
    
    # 为每个 layer 添加 mlp_vision 并替换 forward
    for layer_idx, layer in enumerate(layers):
        # 创建新的 Qwen3MLP 作为 mlp_vision
        mlp_vision = Qwen3MLP(llm_config)
        
        # 用原有 mlp 的权重初始化 mlp_vision
        mlp_vision.load_state_dict(layer.mlp.state_dict())
        
        # 设置 mlp_vision 的参数为可训练
        for param in mlp_vision.parameters():
            param.requires_grad = True
        
        # 添加 mlp_vision 到 layer
        layer.mlp_vision = mlp_vision
        
        # 替换 forward 方法
        layer.forward = types.MethodType(create_moe_forward(layer.forward), layer)
    
    return internvl