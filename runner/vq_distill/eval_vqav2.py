import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import math
import torch
import shortuuid
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from util.dataloader_llava import load_image


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        question_id = line["question_id"]

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        pixel_values = load_image(image, max_num=12)
        return qs, pixel_values, question_id

    def __len__(self):
        return len(self.questions)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_model(exp_dir, step):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.vq_distill.distill import add_quantizer
    from util.misc import disable_torch_init

    disable_torch_init()

    exp_name = exp_dir.split("/")[-1]
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    internvl_path = config.model.internvl_path
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    # internvl = add_quantizer(internvl, config.model.quantizer)
    # ckpt_path = os.path.join(exp_dir, f"quantizer-vq_llava_distill-{step}")
    # print(ckpt_path)
    # ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # m, u = internvl.clip_quantizer.load_state_dict(ckpt, strict=False)
    # print(f"missing keys: {m}, unmatched keys: {u}")

    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)

    return internvl, tokenizer, exp_name, config

def load_data(num_chunks, chunk_idx, exp_name, step):
    question_file = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/dataset/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl"
    # answers_file  = f"/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/evaluation/understanding/vqa/{exp_name}_{step}/vqa_{chunk_idx}.jsonl"
    answers_file  = f"/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/evaluation/understanding/vqa/4B_continuous/vqa_{chunk_idx}.jsonl"
    image_folder = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/dataset/vqav2/test2015"
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions, num_chunks, chunk_idx)
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    dataset = CustomDataset(questions, image_folder)

    return dataset, ans_file

@torch.no_grad()
def eval_model(args):

    device = torch.device("cuda")
    dtype = torch.bfloat16

    internvl, tokenizer, exp_name, config = load_model(args.exp_dir, args.step)
    internvl = internvl.to(device, dtype).eval()

    dataset, ans_file = load_data(args.num_chunks, args.chunk_idx, exp_name, args.step)
    print(f"len(dataset): {len(dataset)}")

    for data in tqdm(dataset):
        question, pixel_values, question_id = data

        pixel_values = pixel_values.to(device, dtype)
        question_prime = '<image>\n' + question

        generation_config = dict(max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        # construct visual features
        # vq_type = getattr(config.model.quantizer, "vq_type", "lfq")
        # vit_feature = internvl.get_vit_feature(pixel_values)
        # if vq_type == "lfq":
        #     visual_features, code_bin = internvl.clip_quantizer(vit_feature)
        #     D = code_bin.shape[-1]
        #     powers = 2 ** torch.arange(D, device=code_bin.device)
        #     code = (code_bin.long() * powers).sum(dim=-1)
        # elif vq_type == "vq":
        #     visual_features, code, _ = internvl.clip_quantizer(vit_feature)
        # elif vq_type == "multi_vq":
        #     visual_features, code, _ = internvl.clip_quantizer(vit_feature)

        # generation_config["visual_features"] = visual_features

        response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)

        ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": question_id,
        #                            "prompt": question,
        #                            "text": response_raw,
        #                            "answer_id": ans_id,
        #                            "model_id": f"{exp_name}_{args.step}",
        #                            "metadata": {}}) + "\n")
        ans_file.write(json.dumps({"question_id": question_id,
                                   "prompt": question,
                                   "text": response_raw,
                                   "answer_id": ans_id,
                                   "model_id": "4B_intern",
                                   "metadata": {}}) + "\n")
    ans_file.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default="/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/vq_llava_distill/1203_multivq_mlp_4B_256_8x2048")
    parser.add_argument("--step", type=int, default=85000)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()
    eval_model(args)