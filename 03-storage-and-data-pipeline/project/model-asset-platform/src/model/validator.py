"""模型校验器"""

import os
import hashlib
from typing import Dict, List


class ModelValidator:
    """模型文件校验"""

    @staticmethod
    def validate_checksum(filepath: str, expected_sha256: str) -> bool:
        """验证文件 SHA256"""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_sha256

    @staticmethod
    def compute_checksum(filepath: str) -> str:
        """计算文件 SHA256"""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def validate_safetensors(filepath: str) -> Dict:
        """验证 safetensors 文件完整性"""
        try:
            from safetensors import safe_open
            result = {"valid": True, "tensors": []}
            with safe_open(filepath, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    result["tensors"].append({
                        "name": key,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    })
            result["num_tensors"] = len(result["tensors"])
            return result
        except Exception as e:
            return {"valid": False, "error": str(e)}

    @staticmethod
    def validate_pytorch_bin(filepath: str) -> Dict:
        """验证 PyTorch .bin 文件"""
        try:
            import torch
            state_dict = torch.load(filepath, map_location="cpu",
                                   weights_only=True)
            tensors = []
            for key, value in state_dict.items():
                if hasattr(value, "shape"):
                    tensors.append({
                        "name": key,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    })
            return {"valid": True, "num_tensors": len(tensors),
                    "tensors": tensors}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    @staticmethod
    def validate(filepath: str) -> Dict:
        """自动检测格式并验证"""
        ext = os.path.splitext(filepath)[1].lower()
        result = {
            "filepath": filepath,
            "size_bytes": os.path.getsize(filepath),
            "checksum": ModelValidator.compute_checksum(filepath),
        }

        if ext == ".safetensors":
            result.update(ModelValidator.validate_safetensors(filepath))
        elif ext in (".bin", ".pt", ".pth"):
            result.update(ModelValidator.validate_pytorch_bin(filepath))
        else:
            result["valid"] = True
            result["note"] = f"No specific validation for {ext}"

        return result
