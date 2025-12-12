import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from einops import rearrange

@torch.inference_mode()
def generate(args):
    from omegaconf import OmegaConf

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    exp_dir = args.exp_dir
    exp_name = args.exp_dir.split("/")[-1]
    step = args.step
    config = OmegaConf.load(os.path.join(exp_dir, f"config.yaml"))

    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.mixture_modality.moe import modify_internvl_to_mixture
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl_to_mixture(internvl, config.model)
    ckpt_path = os.path.join(exp_dir, f"model-mcq_gen-{step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    internvl.load_state_dict(ckpt, strict=False)
    internvl = internvl.to(device, dtype).eval()

    quantizer = get_multi_vq_quantizer(config.model.quantizer)
    ckpt_path = config.model.quantizer.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    quantizer.load_state_dict(ckpt, strict=True)
    quantizer = quantizer.to(device, dtype).eval()

    # ---------- generate from text prompt ----------
    from tqdm import trange
    from transformers import AutoTokenizer
    from util.sample import sample

    IMG_START_TOKEN = "<img>"
    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)
    prompts = [
        "a photo of a wine glass right of a hot dog",
        "a blackboard with words 'Visual thinking without pixels'.",
        "a photo of a tennis racket and a wine glass",
        "a photo of a tv and a bicycle",
        "A man in a white shirt and black pants is playing guitar on the street, with a crowd of people watching him. The background is a city street with buildings and trees.",
        "A photo of a purple backpack and a yellow unbrella.",
        "A whiteboard with words 'Hello World' on it.",
        "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair.",
        "A mother panda playing with her baby in a bamboo forest, with soft sunlight filtering through the leaves.",
        "A lion standing on a rocky cliff overlooking the savanna at sunset, with a golden sky and a herd of wildebeest in the distance.",
        "A woman with long black hair, wearing a red dress, standing in a sunlit field of wildflowers, with soft golden light casting gentle shadows on her face and the wind blowing her hair.",
        "A middle-aged man in a gray suit, sitting at a desk in a modern office, surrounded by bookshelves and a large window overlooking a city skyline at dusk.",
    ]
    cfg_scale = 3.0
    tau = 1.0
    topk = 2048
    topp = 1.0
    sampling_kwargs = {
        "temperature": tau,
        "top_k": topk,
        "top_p": topp,
        "sample_logits": True
    }
    all_generated_codes = []
    for idx, prompt_txt in enumerate(prompts):
        prompt = prompt_txt + IMG_START_TOKEN
        print(prompt)
        tokenizer_output = tokenizer(
            [prompt],
            padding        = True,
            padding_side   = "left",
            truncation     = True,
            return_tensors = "pt",
        )

        input_ids = torch.LongTensor(tokenizer_output["input_ids"]).to(device)
        text_embedding = internvl.language_model.get_input_embeddings()(input_ids)

        if cfg_scale > 1:
            # 创建无条件输入：将文本部分替换为pad token
            uncond_input_ids = input_ids.clone()
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            # 找到IMG_START_TOKEN的位置，将其之前的所有token替换为pad_token_id
            img_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
            for b in range(uncond_input_ids.shape[0]):
                for t in range(uncond_input_ids.shape[1]):
                    if uncond_input_ids[b, t] == img_token_id:
                        break
                    uncond_input_ids[b, t] = pad_token_id
            
            uncond_text_embedding = internvl.language_model.get_input_embeddings()(uncond_input_ids)

            text_embedding_cfg = torch.cat([text_embedding, uncond_text_embedding], dim=0)  # (2*B, L, D)
        else:
            text_embedding_cfg = text_embedding

        past_key_values = None
        generated_codes = []
        x_vq_list = []

        for i in trange(256):
            if i == 0:
                current_input = text_embedding_cfg
            else:
                if cfg_scale > 1:
                    # 条件和无条件分支使用相同的 img_embeds（因为采样结果相同）
                    # 拼接成 (2*B, 1, D) 的形式以匹配 KV cache 的 batch 维度
                    current_input = torch.cat([img_embeds_current, img_embeds_current], dim=0)
                else:
                    current_input = img_embeds_current

            # 构建 vision_token_mask: 第一次是文本(0)，后续是视觉(1)
            if i == 0:
                vision_token_mask = torch.zeros(current_input.shape[0], current_input.shape[1], device=device, dtype=dtype)
            else:
                vision_token_mask = torch.ones(current_input.shape[0], current_input.shape[1], device=device, dtype=dtype)

            outputs = internvl.language_model(
                inputs_embeds     = current_input,
                use_cache         = True,
                past_key_values   = past_key_values,
                vision_token_mask = vision_token_mask,
            )
            base_token = outputs.last_hidden_state[:, -1:, :]  # (B or 2*B, 1, D)
            past_key_values = outputs.past_key_values

            # generate_from_base_token 返回 (B, K)，无论是否使用 CFG
            generated_code = internvl.ar_head.generate_from_base_token(base_token, cfg_scale, sampling_kwargs)
            z_q_current, x_vq_current = quantizer.indices_to_feature(generated_code.unsqueeze(1))
            img_embeds_current = internvl.visual_projector(z_q_current)  # (B, 1, llm_hidden_size)
            x_vq_list.append(x_vq_current)
            generated_codes.append(generated_code)

        generated_code = torch.stack(generated_codes, dim=1) # (B, L, K)
        all_generated_codes.append(generated_code)
        all_codes = torch.cat(all_generated_codes, dim=0)  # (B, L, K) where B is number of prompts
        x_vq = torch.cat(x_vq_list, dim=1) # (B, L, embedding_dim)
        print(f"x_vq shape: {x_vq.shape}, all_codes shape: {all_codes.shape}")
    
        os.makedirs("asset/code", exist_ok=True)
        all_codes = torch.cat(all_generated_codes, dim=0)
        code_path = f"asset/code/code_{exp_name}_{step}_{cfg_scale}.pt"
        torch.save(all_codes, code_path)
        print(f"All codes saved to {code_path}, shape: {all_codes.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    generate(args)