import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import hashlib
from einops import rearrange
from util.trainer import Trainer

def modify_internvl(internvl, config):
    vocab_size = 2 ** config.output_dim
    internvl.visual_embeddings = torch.nn.Embedding(vocab_size, config.llm_hidden_size)
    internvl.visual_aligner = torch.nn.Linear(config.llm_hidden_size, config.llm_hidden_size)
    internvl.visual_head = torch.nn.Linear(config.llm_hidden_size, vocab_size, bias=False)

    return internvl

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _load_models(self):
        from transformers import AutoTokenizer
        from model.llamagen.tokenizer import VQModel, ModelArgs
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        tok = VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4]))
        tok_ckpt = torch.load(self.config.model.quantizer.ckpt_path, map_location="cpu", weights_only=True)["model"]
        tok.load_state_dict(tok_ckpt, strict=True)
        tok = tok.eval()
        
        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        internvl = modify_internvl(internvl, self.config.model.quantizer)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model.internvl_path, trust_remote_code=True, use_fast=False)

        self.model = internvl
        self.tok = tok.to(self.device, self.dtype).eval()
        self.tokenizer = tokenizer
    
    def _load_dataloader(self):
        from util.dataloader import get_blip3o_dataloader
        self.dataloader = get_blip3o_dataloader(self.config.data, self.tokenizer, self.accelerator)
    
    def train(self):
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        training_done = False
        while not training_done:
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    self.model.train()

                    pixel_values = batch["pixel_values"].to(self.device, self.dtype)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    x_gen = pixel_values * 2 - 1
                    B = x_gen.shape[0]

                    with torch.no_grad():
                        _, _, info = self.tok.encode(x_gen)
                        code = rearrange(info[2], "(B L) -> B L", B=B)
                    
                    text_embedding = self.model.language_model.get_input_embeddings()(input_ids)
                    visual_embedding = self.model.visual_aligner(self.model.visual_embeddings(code))
                    joint_embedding = torch.cat([text_embedding, visual_embedding], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((B, self.config.data.num_img_token), dtype=torch.bool, device=self.device)], dim=1)

                    visual_hidden_states = self.model.language_model(
                        inputs_embeds        = joint_embedding,
                        attention_mask       = attention_mask,
                        output_hidden_states = True,
                    ).hidden_states[-1][:, -self.config.data.num_img_token-1:-1, :]

                    visual_token_logits = self.model.visual_head(visual_hidden_states)

                    loss = torch.nn.functional.cross_entropy(visual_token_logits.contiguous().view(-1, visual_token_logits.size(-1)), code.contiguous().view(-1))

                    self.accelerator.backward(loss)
                        
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self.global_step += 1
                        self.progress_bar.update(1)
                        logs = dict(
                            loss_CE = self.accelerator.gather(loss.detach()).mean().item(),
                        )
                        self.accelerator.log(logs, step=self.global_step)
                        self.progress_bar.set_postfix(**logs)

                        if self.global_step > 0 and self.global_step % self.config.train.save_every == 0 and self.accelerator.is_main_process:
                            self.model.eval()
                            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                            save_path = os.path.join(self.output_dir, f"model-{self.config.train.exp_name}-{self.global_step}")
                            torch.save(state_dict, save_path)
                            print(f"model saved to {save_path}")

                        self.accelerator.wait_for_everyone()

                        if self.global_step >= self.config.train.num_iter:
                            training_done = True
                            break

            self.epoch += 1
            self.accelerator.print(f"epoch {self.epoch}: finished")
            self.accelerator.log({"epoch": self.epoch}, step=self.global_step)

        self.accelerator.end_training()


    def dataset_check(self):
        # self.dataloader = self.accelerator.prepare(self.dataloader)
        # self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # 初始化一个全局set来累积整个epoch的不同input_ids的hash值（使用hash减少内存占用）
        epoch_unique_hashes = set()
        
        try:
            for batch in self.dataloader:
                pixel_values = batch["pixel_values"].to(self.device, self.dtype)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # 将当前batch的input_ids序列的hash值添加到全局set中
                for seq in input_ids:
                    # 将tensor转换为bytes以便计算hash
                    seq_bytes = seq.cpu().numpy().tobytes()
                    seq_hash = hashlib.sha256(seq_bytes).hexdigest()
                    epoch_unique_hashes.add(seq_hash)
        
        except StopIteration:
            # epoch结束
            pass
        
        # 统计当前GPU上整个epoch的不同input_ids个数
        local_epoch_unique_count = len(epoch_unique_hashes)
        
        # 将hash值列表转换为可传输的格式
        # 将每个hash值（64字符的hex字符串）转换为整数列表以便传输
        hash_list = list(epoch_unique_hashes)
        
        # 为了跨GPU传输，我们需要将hash值编码
        # 方法：将64字符的hex字符串分成8部分，每部分8字符（32 bits），转换为整数
        # 这样每个hash需要8个int32/int64，总共需要8*len(hash_list)个整数
        # 使用8字符（32 bits）确保不会溢出int64范围
        if len(hash_list) > 0:
            # 将hash字符串转换为整数列表
            hash_ints = []
            for h in hash_list:
                # 将64字符的hex字符串分成8部分，每部分8字符（32 bits）
                int1 = int(h[0:8], 16)
                int2 = int(h[8:16], 16)
                int3 = int(h[16:24], 16)
                int4 = int(h[24:32], 16)
                int5 = int(h[32:40], 16)
                int6 = int(h[40:48], 16)
                int7 = int(h[48:56], 16)
                int8 = int(h[56:64], 16)
                hash_ints.extend([int1, int2, int3, int4, int5, int6, int7, int8])
            
            # 创建tensor来存储hash值（每个hash用8个int64表示）
            # 第一个元素存储hash的数量
            hash_tensor = torch.tensor([len(hash_list)] + hash_ints, device=self.device, dtype=torch.long)
        else:
            # 如果没有hash值，创建一个只包含0的tensor
            hash_tensor = torch.tensor([0], device=self.device, dtype=torch.long)
        
        # 使用all_gather收集所有GPU的hash值
        # 由于不同GPU可能有不同数量的hash值，我们需要先收集数量信息
        local_count_tensor = torch.tensor([local_epoch_unique_count], device=self.device, dtype=torch.long)
        gathered_counts = self.accelerator.gather(local_count_tensor)
        
        # 收集所有hash值（需要处理不同长度的情况）
        # 先找到最大长度
        max_hash_count = max(len(hash_list), 1) if len(hash_list) > 0 else 1
        # 使用all_reduce找到全局最大值（如果distributed可用）
        if self.accelerator.num_processes > 1:
            max_tensor = torch.tensor([max_hash_count], device=self.device, dtype=torch.long)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(max_tensor, op=torch.distributed.ReduceOp.MAX)
                max_hash_count = max_tensor.item()
            else:
                # 如果distributed未初始化，使用gather然后找最大值
                gathered_max = self.accelerator.gather(max_tensor)
                max_hash_count = gathered_max.max().item()
        else:
            max_hash_count = len(hash_list) if len(hash_list) > 0 else 1
        
        # 填充hash_tensor到统一长度（每个hash8个int，加上1个长度信息）
        target_length = 1 + max_hash_count * 8
        if hash_tensor.size(0) < target_length:
            padding = torch.zeros(target_length - hash_tensor.size(0), device=self.device, dtype=torch.long)
            hash_tensor = torch.cat([hash_tensor, padding])
        
        # 收集所有GPU的hash tensor
        gathered_hash_tensors = self.accelerator.gather(hash_tensor.unsqueeze(0))
        
        # 在主进程上计算全局唯一数
        if self.accelerator.is_main_process:
            # 合并所有GPU的hash值并去重
            global_unique_hashes = set()
            for gpu_hash_tensor in gathered_hash_tensors:
                num_hashes = gpu_hash_tensor[0].item()
                if num_hashes > 0:
                    for i in range(num_hashes):
                        int1 = gpu_hash_tensor[1 + i * 8].item()
                        int2 = gpu_hash_tensor[1 + i * 8 + 1].item()
                        int3 = gpu_hash_tensor[1 + i * 8 + 2].item()
                        int4 = gpu_hash_tensor[1 + i * 8 + 3].item()
                        int5 = gpu_hash_tensor[1 + i * 8 + 4].item()
                        int6 = gpu_hash_tensor[1 + i * 8 + 5].item()
                        int7 = gpu_hash_tensor[1 + i * 8 + 6].item()
                        int8 = gpu_hash_tensor[1 + i * 8 + 7].item()
                        # 将8个整数转换回hex字符串（每个8字符）
                        h = f"{int1:08x}{int2:08x}{int3:08x}{int4:08x}{int5:08x}{int6:08x}{int7:08x}{int8:08x}"
                        global_unique_hashes.add(h)
            
            global_unique_count = len(global_unique_hashes)
            total_sum = gathered_counts.sum().item()
            # 不同GPU之间的重复hash数量
            duplicate_count = total_sum - global_unique_count
            duplicate_rate = duplicate_count / total_sum * 100 if total_sum > 0 else 0
            # 平均每个唯一序列在多少个GPU上出现（近似值）
            avg_appearances = total_sum / global_unique_count if global_unique_count > 0 else 0
            
            self.accelerator.print(f"一个epoch后不同input_ids的总数（全局去重后）: {global_unique_count}")
            self.accelerator.print(f"各GPU唯一数之和: {total_sum}")
            self.accelerator.print(f"各GPU上的唯一数: {gathered_counts.cpu().tolist()}")
            self.accelerator.print(f"不同GPU之间的重复hash数量: {duplicate_count}")
            self.accelerator.print(f"重复率: {duplicate_rate:.2f}%")
            self.accelerator.print(f"平均每个唯一序列约在 {avg_appearances:.2f} 个GPU上出现（近似值）")
            self.accelerator.print(f"说明: 由于数据随机shuffle，不同GPU处理的数据有重叠，导致很多序列在多个GPU上重复出现")


def main(args):
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    trainer = MyTrainer(config)
    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)