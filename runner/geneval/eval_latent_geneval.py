import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import json
from omegaconf import OmegaConf
from datasets import load_dataset
from transformers import AutoTokenizer
from util.dataloader_llava import load_image
from tqdm import tqdm
from util.misc import disable_torch_init

disable_torch_init()

@torch.no_grad()
def eval_latent_geneval(args):
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.vq_distill.distill import add_quantizer

    device = torch.device("cuda")
    dtype = torch.bfloat16
    exp_dir = args.exp_dir
    # exp_name = args.exp_dir.split("/")[-1]
    # step = args.step
    geneval_path = "asset/geneval/1224_new_save/45000"

    # load latent geneval questions
    with open("evaluation/generation/geneval/correct_answers.jsonl", "r") as f:
        json_lines = [json.loads(line) for line in f]

    # load pretrained internvl and quantizer, quantizer is used to reproduce the vq visual features
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    config = OmegaConf.load(os.path.join(exp_dir, f"config.yaml"))

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = internvl.to(device, dtype).eval()

    quantizer = get_multi_vq_quantizer(config.model.quantizer)
    ckpt_path = config.model.quantizer.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    quantizer.load_state_dict(ckpt, strict=True)
    quantizer = quantizer.to(device, dtype).eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)

    with open(os.path.join(geneval_path, "results.jsonl"), "w") as results_file:
        for json_line in json_lines:
            index = json_line["index"]
            question = json_line["question"]
            correct_answer = json_line["correct_answer"]
            code_path = os.path.join(geneval_path, index, "code", "code.pt")
            code = torch.load(code_path, map_location="cpu").to(device)
            z_q, x_vq = quantizer.indices_to_feature(code)
            pixel_values = torch.zeros((1, 3, 448, 448)).to(device, dtype)
            print(question)

            answer_list = []
            for idx in range(x_vq.shape[0]):
                question_prime = '<image>\n' + question
                generation_config = dict(max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                generation_config["visual_features"] = x_vq[idx]
                response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)
                print(response_raw)
                answer_list.append(response_raw)
                print('='*16)
            results_file.write(json.dumps({
                "index": index,
                "type": json_line["type"],
                "correct_answer": correct_answer,
                "answers": answer_list,
            }) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    eval_latent_geneval(args)