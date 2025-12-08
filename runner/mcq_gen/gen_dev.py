import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

@torch.inference_mode()
def generate_and_describe(save_code=False):
    from omegaconf import OmegaConf

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    exp_path = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/mcq_gen/1205_dev_ar_head"
    step = 200000
    # config = OmegaConf.load("config/mcq_gen/dev_ar_head_g1.yaml")
    config = OmegaConf.load(os.path.join(exp_path, f"config.yaml"))

    # ---------- load internvl and quantizer ----------
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.mcq_gen.dev_ar_head import modify_internvl
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl(internvl, config.model)
    ckpt_path = os.path.join(exp_path, f"model-mcq_gen-{step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    internvl.load_state_dict(ckpt, strict=True)
    internvl = internvl.to(device, dtype).eval()
    print("load internvl done")

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
        "A cute dog in a park.",
    ]
    cfg_scale = 1.0
    tau = 0.9
    topk = 2048
    topp = 0.96
    sampling_kwargs = {
        "temperature": tau,
        "top_k": topk,
        "top_p": topp,
        "sample_logits": True
    }

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

        # 准备CFG输入：条件输入和无条件输入
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
                # 只取最后一个时间步的 embedding 用于下一轮输入
                if cfg_scale > 1:
                    current_input = torch.cat([img_embeds[:, -1:, :], img_embeds_uncond[:, -1:, :]], dim=0)
                else:
                    current_input = img_embeds[:, -1:, :]
            
            outputs = internvl.language_model.model(
                inputs_embeds   = current_input,
                use_cache       = True,
                past_key_values = past_key_values
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            if cfg_scale > 1:
                z_cond = hidden_states[:batch_size, -1, :]  # (B, D)
                z_uncond = hidden_states[batch_size:, -1, :]  # (B, D)
            else:
                z = hidden_states[:, -1, :]

            num_codebooks = config.model.quantizer.num_codebooks
            if cfg_scale > 1:
                next_embed_cond = z_cond.unsqueeze(dim=1)  # (B, 1, D)
                next_embed_uncond = z_uncond.unsqueeze(dim=1)  # (B, 1, D)
            else:
                next_embed = z.unsqueeze(dim=1)  # (B, 1, D)

            indices_arhead = []
            for i_head in range(num_codebooks):
                if cfg_scale > 1:
                    # 同时计算条件和无条件logits
                    ar_next_logits_cond = internvl.ar_head(inputs_embeds=next_embed_cond)  # (B, K, V)
                    ar_next_logits_uncond = internvl.ar_head(inputs_embeds=next_embed_uncond)  # (B, K, V)
                    
                    # 取当前 codebook 的 logits
                    current_logits_cond = ar_next_logits_cond[:, i_head:i_head+1, :]  # (B, 1, V)
                    current_logits_uncond = ar_next_logits_uncond[:, i_head:i_head+1, :]  # (B, 1, V)
                    
                    # 应用CFG公式: logits = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
                    current_logits = current_logits_uncond + cfg_scale * (current_logits_cond - current_logits_uncond)
                    next_token, next_prob = sample(current_logits, **sampling_kwargs)
                else:
                    ar_next_logits = internvl.ar_head(inputs_embeds=next_embed)  # (B, K, V)
                    # 取当前 codebook 的 logits，而不是最后一个
                    current_logits = ar_next_logits[:, i_head:i_head+1, :]  # (B, 1, V)
                    next_token, next_prob = sample(current_logits, **sampling_kwargs)
                
                indices_arhead.append(next_token)

                if i_head < num_codebooks - 1:
                    if cfg_scale > 1:
                        predicted_embed_cond = internvl.ar_head.codebooks[i_head](next_token)
                        predicted_embed_uncond = internvl.ar_head.codebooks[i_head](next_token)
                        next_embed_cond = torch.cat([next_embed_cond, predicted_embed_cond], dim=1)
                        next_embed_uncond = torch.cat([next_embed_uncond, predicted_embed_uncond], dim=1)
                    else:
                        predicted_embed = internvl.ar_head.codebooks[i_head](next_token)
                        next_embed = torch.cat([next_embed, predicted_embed], dim=1)

            # 只保存当前时间步的 tokens
            pred_tokens.append(torch.cat(indices_arhead, dim=1))  # 每一个元素: (B, K)
            
            # 只计算当前时间步的 z_q 和 img_embeds，避免冗余计算
            current_multi_ids = torch.cat(indices_arhead, dim=1).unsqueeze(1)  # (B, 1, K)
            z_q_current, x_vq_current = quantizer.indices_to_feature(current_multi_ids)  # (B, 1, embedding_dim)
            img_embeds_current = internvl.visual_projector(z_q_current)  # (B, 1, llm_hidden_size)
            x_vq_list.append(x_vq_current)
            
            # 累积 img_embeds 用于后续时间步
            if cfg_scale > 1:
                if i == 0:
                    img_embeds = img_embeds_current
                    # 对于无条件分支，也需要生成对应的img_embeds（虽然不会被使用，但为了保持一致性）
                    img_embeds_uncond = img_embeds_current.clone()
                else:
                    img_embeds = torch.cat([img_embeds, img_embeds_current], dim=1)
                    img_embeds_uncond = torch.cat([img_embeds_uncond, img_embeds_current.clone()], dim=1)
            else:
                if i == 0:
                    img_embeds = img_embeds_current
                else:
                    img_embeds = torch.cat([img_embeds, img_embeds_current], dim=1)

        generated_code = torch.stack(pred_tokens, dim=1) # (B, L, K)
        if save_code:
            os.makedirs("asset/mcq_gen", exist_ok=True)
            code_path = f"asset/mcq_gen/code_{prompt_txt[:10]}.pt"
            torch.save(generated_code, code_path)
            print(f"Code saved to {code_path}")
        x_vq = torch.cat(x_vq_list, dim=1) # (B, L, embedding_dim)

        # ---------- understand the generated code ----------
        internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
        internvl = internvl.to(device, dtype).eval()
        print("load internvl done")
        print(x_vq.shape)

        question_prime = "<image>\n" + "Describe this image in detail."
        generation_config = dict(max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generation_config["visual_features"] = x_vq
        pixel_values = torch.zeros((1, 3, 448, 448)).to(device, dtype)
        response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)
        print(response_raw)

if __name__ == "__main__":
    generate_and_describe()