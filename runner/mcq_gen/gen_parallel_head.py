import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from einops import rearrange

@torch.inference_mode()
def generate_and_describe(args, save_code=False):
    from omegaconf import OmegaConf

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    exp_dir = args.exp_dir
    step = args.step
    config = OmegaConf.load(os.path.join(exp_dir, f"config.yaml"))
    # config = OmegaConf.load("config/mcq_gen/dev_parallel_head_g1.yaml")

    # ---------- load internvl and quantizer ----------
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.mcq_gen.dev_parallel_head import modify_internvl_parallel_head
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl_parallel_head(internvl, config.model)
    ckpt_path = os.path.join(exp_dir, f"model-mcq_gen-{step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    internvl.load_state_dict(ckpt, strict=True)
    internvl = internvl.to(device, dtype).eval()

    internvl_original = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl_original = internvl_original.to(device, dtype).eval()

    print("load internvl_original done")

    quantizer = get_multi_vq_quantizer(config.model.quantizer)
    ckpt_path = config.model.quantizer.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    quantizer.load_state_dict(ckpt, strict=True)
    quantizer = quantizer.to(device, dtype).eval()
    print(f"load quantizer done")

    # ---------- generate from text prompt ----------
    from tqdm import trange
    from transformers import AutoTokenizer
    from util.sample import sample

    IMG_START_TOKEN = "<img>"
    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)
    prompts = [
        "A stunning sunset over the ocean, the sky is painted with shades of orange and pink, the ocean is calm and the waves are gentle.",
        "A beautiful building in a city, the building is made of glass and steel, the building is tall and has a modern design.",
        "A group of people playing soccer on a sunny day, the players are running and passing the ball, the ball is flying through the air, the players are smiling and having fun.",
        "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair."
    ]
    cfg_scale = 4.0
    tau = 0.9
    topk = 2048
    topp = 0.96
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
            
            # 合并条件和无条件输入
            batch_size = input_ids.shape[0]
            text_embedding_cfg = torch.cat([text_embedding, uncond_text_embedding], dim=0)  # (2*B, L, D)
        else:
            text_embedding_cfg = text_embedding
            batch_size = input_ids.shape[0]

        past_key_values = None
        pred_tokens = []
        x_vq_list = []

        for i in trange(256):
            if i == 0:
                current_input = text_embedding_cfg
            else:
                if cfg_scale > 1:
                    # 只取最后一个时间步的 embedding 用于下一轮输入
                    current_input = torch.cat([img_embeds[:, -1:, :], img_embeds_uncond[:, -1:, :]], dim=0)
                else:
                    current_input = img_embeds_current

            outputs = internvl.language_model.model(
                inputs_embeds   = current_input,
                use_cache       = True,
                past_key_values = past_key_values
            )
            L = outputs.last_hidden_state.shape[1]
            hidden_state = outputs.last_hidden_state[:, L-1:L, :] # (B or 2*B, 1, D)
            past_key_values = outputs.past_key_values

            if cfg_scale > 1:
                # 分离条件和无条件 hidden_state
                hidden_state_cond = hidden_state[:batch_size, :, :]  # (B, 1, D)
                hidden_state_uncond = hidden_state[batch_size:, :, :]  # (B, 1, D)
                
                # 分别计算条件和无条件 logits
                logits_cond = internvl.parallel_head(hidden_state_cond)  # (B, 1, K, V)
                logits_uncond = internvl.parallel_head(hidden_state_uncond)  # (B, 1, K, V)
                
                # 应用CFG公式: logits = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)  # (B, 1, K, V)
            else:
                logits = internvl.parallel_head(hidden_state) # (B, 1, K, V)
            
            logits = rearrange(logits, "B 1 K V -> (B K) V")
            next_token, _ = sample(logits, **sampling_kwargs) # (B*K, 1)
            next_token = rearrange(next_token, "(B K) 1 -> B K", B=batch_size)

            pred_tokens.append(next_token)

            z_q_current, x_vq_current = quantizer.indices_to_feature(next_token.unsqueeze(1))
            img_embeds_current = internvl.visual_projector(z_q_current)  # (B, 1, llm_hidden_size)
            x_vq_list.append(x_vq_current)
            
            # 累积 img_embeds 用于后续时间步
            if cfg_scale > 1:
                if i == 0:
                    img_embeds = img_embeds_current
                    # 对于无条件分支，也需要生成对应的img_embeds
                    img_embeds_uncond = img_embeds_current.clone()
                else:
                    img_embeds = torch.cat([img_embeds, img_embeds_current], dim=1)
                    img_embeds_uncond = torch.cat([img_embeds_uncond, img_embeds_current.clone()], dim=1)

        generated_code = torch.stack(pred_tokens, dim=1) # (B, L, K)
        all_generated_codes.append(generated_code)
        x_vq = torch.cat(x_vq_list, dim=1) # (B, L, embedding_dim)

        # ---------- understand the generated code ----------
        question_prime = "<image>\n" + "Describe this image in detail."
        generation_config = dict(max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generation_config["visual_features"] = x_vq
        pixel_values = torch.zeros((1, 3, 448, 448)).to(device, dtype)
        response_raw = internvl_original.chat(tokenizer, pixel_values, question_prime, generation_config)
        print(response_raw)
    
    if save_code and all_generated_codes:
        os.makedirs("asset/mcq_gen", exist_ok=True)
        all_codes = torch.cat(all_generated_codes, dim=0)  # (B, L, K) where B is number of prompts
        code_path = "asset/mcq_gen/code_all_prompts.pt"
        torch.save(all_codes, code_path)
        print(f"All codes saved to {code_path}, shape: {all_codes.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    generate_and_describe(args, save_code=True)