import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from accelerate.utils import set_seed
from accelerate import Accelerator

@torch.inference_mode()
def generate(args):
    from runner.mixture_modality.moe_gen import intern_gen
    from util.misc import disable_torch_init
    # Load prompts
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # Load model
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.bfloat16
    exp_dir = args.exp_dir
    exp_name = args.exp_dir.split("/")[-1]
    step = args.step
    config = OmegaConf.load(os.path.join(exp_dir, f"config.yaml"))

    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from runner.mixture_modality.moe import modify_internvl_to_mixture
    from model.quantizer.multi_vq import get_multi_vq_quantizer

    disable_torch_init()

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl = modify_internvl_to_mixture(internvl, config.model)
    ckpt_path = os.path.join(exp_dir, f"model-mcq_gen-{step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    internvl.load_state_dict(ckpt, strict=False)
    internvl = internvl.to(device, dtype).eval()

    quantizer = get_multi_vq_quantizer(config.model.quantizer)
    ckpt_path = config.model.quantizer.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    quantizer.load_state_dict(ckpt, strict=True)
    quantizer = quantizer.to(device, dtype).eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)

    B = 4
    cfg_scale = 3.0
    tau = 0.5
    topk = 50
    topp = 0.95
    sampling_kwargs = {
        "temperature": tau,
        "top_k": topk,
        "top_p": topp,
        "sample_logits": True
    }

    # 将任务分配到不同的GPU
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    
    for index, metadata in enumerate(metadatas):
        # 只处理分配给当前进程的任务
        if index % num_processes != process_index:
            continue
            
        set_seed(args.seed)

        outpath = os.path.join(args.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        accelerator.print(f"Process {process_index}: Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        code_path = os.path.join(outpath, "code")
        os.makedirs(sample_path, exist_ok=True)
        os.makedirs(code_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        
        generated_code = intern_gen(internvl, quantizer, tokenizer, prompt, B, cfg_scale, sampling_kwargs, device)
        accelerator.print(f"Process {process_index}: Generated code shape: {generated_code.shape}")
        torch.save(generated_code, os.path.join(code_path, "code.pt"))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="asset/geneval_generation")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    generate(args)