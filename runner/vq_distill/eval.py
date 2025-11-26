import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from omegaconf import OmegaConf
from datasets import load_dataset
from transformers import AutoTokenizer
from util.dataloader_llava import load_image

@torch.no_grad()
def img_describe():
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.vq_distill.distill import add_quantizer

    device = torch.device("cuda")
    dtype = torch.bfloat16

    exp_dir = "/data/phd/jinjiachun/experiment/vq_llava_distill/1126_lfq_mlp_14_intern1B"
    step = 8000
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    internvl_path = "/data/phd/hf_models/ckpt/OpenGVLab/InternVL3_5-1B"
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    internvl = add_quantizer(internvl, config.model.quantizer)
    
    ckpt_path = os.path.join(exp_dir, f"internvl-vq_llava_distill-{step}")
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m, u = internvl.load_state_dict(ckpt, strict=False)
    print(f"missing keys: {m}, unmatched keys: {u}")    

    internvl = internvl.to(device, dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)

    # ---------- chat with the model ----------
    image = "/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi_1.jpg"
    pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()

    question_prime = "<image>\n" + "Describe this image in great detail."
    generation_config = dict(max_new_tokens=128, do_sample=True)
    
    # extract visual features from pixel values
    vit_feature = internvl.get_vit_feature(pixel_values)
    print(pixel_values.shape, vit_feature.shape)

    visual_features, code = internvl.lfq(vit_feature)

    generation_config["visual_features"] = visual_features

    response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)
    
    print(response_raw)


if __name__ == "__main__":
    img_describe()