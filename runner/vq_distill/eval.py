import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from omegaconf import OmegaConf
from datasets import load_dataset
from transformers import AutoTokenizer
from util.dataloader_llava import load_image
from tqdm import tqdm

@torch.no_grad()
def img_describe():
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.vq_distill.distill import add_quantizer

    device = torch.device("cuda")
    dtype = torch.bfloat16

    exp_dir = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/vq_llava_distill/1128_lfq_vit_16_intern4B_desdata"
    step = 42000
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    internvl_path = config.model.internvl_path
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    internvl = add_quantizer(internvl, config.model.quantizer)
    
    ckpt_path = os.path.join(exp_dir, f"quantizer-vq_llava_distill-{step}")
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m, u = internvl.clip_quantizer.load_state_dict(ckpt, strict=True)
    print(f"missing keys: {m}, unmatched keys: {u}")    

    internvl = internvl.to(device, dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)

    # ---------- chat with the model ----------
    image = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/非常厉害.png"
    pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
    question_prime = "<image>\n" + "Describe this image in detail."
    generation_config = dict(max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    # extract visual features from pixel values
    vit_feature = internvl.get_vit_feature(pixel_values)
    # print(pixel_values.shape, vit_feature.shape)
    visual_features, code = internvl.clip_quantizer(vit_feature)
    generation_config["visual_features"] = visual_features
    response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)
    print(response_raw)

@torch.no_grad()
def test_mme(args):
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.vq_distill.distill import add_quantizer

    device = args.device
    dtype = torch.bfloat16

    # ---------- load trained internvl with new projector ----------
    exp_dir = args.exp_dir
    step = args.step
    exp_name = exp_dir.split("/")[-1]
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    internvl_path = config.model.internvl_path
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    internvl = add_quantizer(internvl, config.model.quantizer)
    
    ckpt_path = os.path.join(exp_dir, f"quantizer-vq_llava_distill-{step}")
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m, u = internvl.clip_quantizer.load_state_dict(ckpt, strict=True)
    print(f"missing keys: {m}, unmatched keys: {u}")    

    internvl = internvl.to(device, dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)

    data_files = {
        "test": "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/dataset/darkyarding/MME/data/test-*-of-*.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)

    for data in tqdm(dataset["test"]):
        img_name = data["question_id"].split("/")[-1]
        category = data["category"]
        image = data["image"].convert("RGB")
        question = data["question"] + "Directly answer yes or no, with no other words."
        gt_answer = data["answer"]

        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).to(device)

        question_prime = '<image>\n' + question

        generation_config = dict(max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        # construct visual features
        vit_feature = internvl.get_vit_feature(pixel_values)
        visual_features, code = internvl.clip_quantizer(vit_feature)
        generation_config["visual_features"] = visual_features

        response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)

        answer = extract_yes_no_answer(response_raw)
        # print(response_raw, answer)
        model_name = ckpt_path.split("/")[-1]
        os.makedirs(f"evaluation/understanding/mme/{exp_name}_{step}", exist_ok=True)
        with open(f"evaluation/understanding/mme/{exp_name}_{step}/{category}.txt", "a") as f:
            line = f"{img_name}\t{question}\t{gt_answer}\t{answer}\n"
            f.write(line)

def extract_yes_no_answer(response_raw):
    import re
    response_lower = response_raw.lower().strip()
    if re.search(r'\byes\b', response_lower):
        response = "yes"
    elif re.search(r'\bno\b', response_lower):
        response = "no"
    else:
        # 如果没有找到yes/no，取第一个词
        response = response_raw.split()[0].strip() if response_raw.split() else "unknown"
    return response

@torch.no_grad()
def test_mme_original():
    from model.internvl.modeling_internvl_chat import InternVLChatModel

    device = "cuda:0"
    dtype = torch.bfloat16

    # ---------- load trained internvl with new projector ----------
    exp_name = "8B_continuous"
    internvl_path = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/ckpt/OpenGVLab/InternVL3_5-8B"
    internvl = InternVLChatModel.from_pretrained(internvl_path)

    internvl = internvl.to(device, dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)

    data_files = {
        "test": "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/dataset/darkyarding/MME/data/test-*-of-*.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)

    for i, data in enumerate(dataset["test"]):
        img_name = data["question_id"].split("/")[-1]
        category = data["category"]
        image = data["image"].convert("RGB")
        question = data["question"] + "Directly answer yes or no, with no other words."
        gt_answer = data["answer"]

        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).to(device)

        question_prime = '<image>\n' + question

        generation_config = dict(max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)

        answer = extract_yes_no_answer(response_raw)
        print(response_raw, answer)
        os.makedirs(f"evaluation/understanding/mme/{exp_name}", exist_ok=True)
        with open(f"evaluation/understanding/mme/{exp_name}/{category}.txt", "a") as f:
            line = f"{img_name}\t{question}\t{gt_answer}\t{answer}\n"
            f.write(line)

def test_mme_original_llamagen_reconstruction():
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from model.llamagen.tokenizer import VQModel, ModelArgs
    from util.dataloader_llava import load_image_llamagen_recon

    device = "cuda:0"
    dtype = torch.bfloat16
    exp_name = "4B_llamagen_recon_input"
    internvl_path = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/ckpt/OpenGVLab/InternVL3_5-4B"
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    internvl = internvl.to(device, dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)

    llamagen_path = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/ckpt/LlamaGen/vq_ds16_t2i.pt"
    tok = VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4]))
    tok_ckpt = torch.load(llamagen_path, map_location="cpu", weights_only=True)["model"]
    tok.load_state_dict(tok_ckpt, strict=True)
    tok = tok.to(device, torch.float32).eval()

    data_files = {
        "test": "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/dataset/darkyarding/MME/data/test-*-of-*.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)

    for data in tqdm(dataset["test"]):
        img_name = data["question_id"].split("/")[-1]
        category = data["category"]
        image = data["image"].convert("RGB")
        question = data["question"] + "Directly answer yes or no, with no other words."
        gt_answer = data["answer"]

        pixel_values = load_image_llamagen_recon(image, tok, max_num=12).to(torch.bfloat16).to(device)

        question_prime = '<image>\n' + question

        generation_config = dict(max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        # construct visual features
        # vit_feature = internvl.get_vit_feature(pixel_values)
        # visual_features, code = internvl.clip_quantizer(vit_feature)
        # generation_config["visual_features"] = visual_features

        response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)

        answer = extract_yes_no_answer(response_raw)
        os.makedirs(f"evaluation/understanding/mme/{exp_name}", exist_ok=True)
        with open(f"evaluation/understanding/mme/{exp_name}/{category}.txt", "a") as f:
            line = f"{img_name}\t{question}\t{gt_answer}\t{answer}\n"
            f.write(line)


if __name__ == "__main__":
    # img_describe()
    # test_mme_original()
    test_mme_original_llamagen_reconstruction()
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_dir", required=True)
    # parser.add_argument("--step", required=True)
    # parser.add_argument("--device", default="cuda:0")
    # args = parser.parse_args()
    # test_mme(args)