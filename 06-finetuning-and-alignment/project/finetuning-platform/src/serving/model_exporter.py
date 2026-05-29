"""
模型导出器：支持多种导出格式
"""

import os
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class ModelExporter:
    """模型导出器"""

    def export_merged(self, base_model_path: str, lora_path: str,
                     output_dir: str, dtype: str = "bfloat16"):
        """合并 LoRA 并导出完整模型"""
        print(f"合并导出: {base_model_path} + {lora_path}")

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        # 加载
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch_dtype)

        # 合并
        merged = model.merge_and_unload()

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        merged.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        )
        print(f"导出完成: {output_dir} ({size/1e9:.2f} GB)")
        return output_dir

    def export_lora_only(self, model, tokenizer, output_dir: str):
        """只导出 LoRA 权重"""
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"LoRA 权重已导出: {output_dir}")

    def export_for_vllm(self, model_path: str, output_dir: Optional[str] = None):
        """为 vLLM 准备模型"""
        output_dir = output_dir or model_path
        # vLLM 直接支持 HF 格式，确认文件完整即可
        required_files = ["config.json", "tokenizer.json"]
        missing = [f for f in required_files
                  if not os.path.exists(os.path.join(output_dir, f))]
        if missing:
            print(f"警告: 缺少文件 {missing}")
        else:
            print(f"模型已准备好用于 vLLM: {output_dir}")
            print(f"  启动命令: vllm serve {output_dir}")
        return output_dir
