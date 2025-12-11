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

    # ---------- load internvl and quantizer ----------
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.mcq_gen.dev_abs_code import modify_internvl
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl(internvl, config.model)
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
        "a photo of a wine glass right of a hot dog",
        "a photo of 4 televisions.",
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
    cfg_scale = 5.0
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

        for i in trange(2048):
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
            hidden_state = outputs.last_hidden_state[:, -1:, :] # (B or 2*B, 1, D)
            past_key_values = outputs.past_key_values

            if cfg_scale > 1:
                # 分离条件和无条件 hidden_state
                hidden_state_cond = hidden_state[:batch_size, :, :]  # (B, 1, D)
                hidden_state_uncond = hidden_state[batch_size:, :, :]  # (B, 1, D)
                
                # 分别计算条件和无条件 logits
                logits_cond = internvl.head(hidden_state_cond)  # (B, 1, K*V)
                logits_uncond = internvl.head(hidden_state_uncond)  # (B, 1, K*V)
                
                # Reshape to (B, K, V) for CFG calculation
                logits_cond = logits_cond.squeeze(1)
                logits_uncond = logits_uncond.squeeze(1)
                
                # 应用CFG公式: logits = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)  # (B, K, V)
                # logits = logits.view(batch_size * K, V)  # (B*K, V)
            else:
                logits = internvl.head(hidden_state) # (B, 1, K*V)
                # reshape logits to (B*K, V) for sampling
                logits = logits.squeeze(1)  # (B, K*V)
                # logits = logits.view(batch_size, K, V)  # (B, K, V)
                # logits = logits.view(batch_size * K, V)  # (B*K, V)
            
            next_token, _ = sample(logits, **sampling_kwargs)  # (B*K, 1)

            pred_tokens.append(next_token)
            
            # Convert abs_code to embeddings: (B, K) -> (B, K, D)
            visual_embeds = internvl.visual_embeddings(next_token)  # (B, K, D)
            # Sum over K dimension to get (B, 1, D) for next iteration
            img_embeds_current = visual_embeds.sum(dim=1, keepdim=True)  # (B, 1, D)
            
            # 累积 img_embeds 用于后续时间步（CFG需要）
            if cfg_scale > 1:
                if i == 0:
                    img_embeds = img_embeds_current
                    # 对于无条件分支，也需要生成对应的img_embeds
                    img_embeds_uncond = img_embeds_current.clone()
                else:
                    img_embeds = torch.cat([img_embeds, img_embeds_current], dim=1)
                    img_embeds_uncond = torch.cat([img_embeds_uncond, img_embeds_current.clone()], dim=1)

        # Stack all generated tokens: (B, L, K)
        generated_code = torch.stack(pred_tokens, dim=1).squeeze(-1)
        code = quantizer.to_rel_code(generated_code)
        all_generated_codes.append(code)

        # z_q, x_vq = quantizer.indices_to_feature(code)
        # print(z_q.shape, x_vq.shape)
        # # ---------- understand the generated code ----------
        # question_prime = "<image>\n" + "Describe this image in detail."
        # generation_config = dict(max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        # generation_config["visual_features"] = x_vq
        # pixel_values = torch.zeros((1, 3, 448, 448)).to(device, dtype)
        # response_raw = internvl_original.chat(tokenizer, pixel_values, question_prime, generation_config)
        # print(response_raw)

    # Save all generated codes if requested
    if save_code and all_generated_codes:
        os.makedirs("asset/mcq_gen", exist_ok=True)
        all_codes = torch.cat(all_generated_codes, dim=0)  # (num_prompts, L, K)
        code_path = f"asset/mcq_gen/code_all_prompts_abs_code_{step}.pt"
        torch.save(all_codes, code_path)
        print(f"All codes saved to {code_path}, shape: {all_codes.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    generate_and_describe(args, save_code=True)