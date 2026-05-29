"""模型格式转换器"""

import os
import time
from typing import Optional, Dict


class FormatConverter:
    """模型格式转换

    支持的转换：
    - bin → safetensors
    - safetensors → bin
    - bin → gguf (需要外部工具)
    """

    SUPPORTED_FORMATS = {"bin", "safetensors", "gguf", "onnx"}

    @staticmethod
    def convert(input_path: str, output_path: str,
                target_format: str) -> Dict:
        """转换模型格式"""
        if target_format not in FormatConverter.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {target_format}")

        source_format = os.path.splitext(input_path)[1].lstrip(".")

        t_start = time.perf_counter()

        if source_format == "bin" and target_format == "safetensors":
            result = FormatConverter._bin_to_safetensors(input_path, output_path)
        elif source_format == "safetensors" and target_format == "bin":
            result = FormatConverter._safetensors_to_bin(input_path, output_path)
        else:
            raise ValueError(f"Conversion {source_format} → {target_format} "
                           f"not supported")

        duration = time.perf_counter() - t_start
        result["duration_s"] = duration
        return result

    @staticmethod
    def _bin_to_safetensors(input_path: str, output_path: str) -> Dict:
        """PyTorch .bin → safetensors"""
        try:
            import torch
            from safetensors.torch import save_file

            state_dict = torch.load(input_path, map_location="cpu",
                                   weights_only=True)
            # 只保留 tensor 类型
            tensor_dict = {
                k: v for k, v in state_dict.items()
                if isinstance(v, torch.Tensor)
            }
            save_file(tensor_dict, output_path)

            return {
                "status": "success",
                "input_size": os.path.getsize(input_path),
                "output_size": os.path.getsize(output_path),
                "num_tensors": len(tensor_dict),
            }
        except ImportError as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def _safetensors_to_bin(input_path: str, output_path: str) -> Dict:
        """safetensors → PyTorch .bin"""
        try:
            import torch
            from safetensors.torch import load_file

            state_dict = load_file(input_path)
            torch.save(state_dict, output_path)

            return {
                "status": "success",
                "input_size": os.path.getsize(input_path),
                "output_size": os.path.getsize(output_path),
                "num_tensors": len(state_dict),
            }
        except ImportError as e:
            return {"status": "error", "message": str(e)}
