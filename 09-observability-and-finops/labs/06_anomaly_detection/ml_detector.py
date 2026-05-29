"""
ML 异常检测器 — GPU 推理多维指标异常检测
==========================================

统计方法适合单指标检测, ML 方法适合:
1. 多维协同异常: TTFT 正常 + TPOT 正常 + KV Cache 正常, 但三者组合异常
2. 复杂时序模式: 非线性、非平稳、多模态分布
3. 自动特征学习: 不需要人工定义 "什么是异常"

实现:
- Isolation Forest: 无监督, 多维点异常检测
- Autoencoder: 学习正常模式, 重构误差检测异常
- DBSCAN: 密度聚类, 发现行为模式变化

依赖:
    pip install numpy pandas scikit-learn torch
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 特征工程: 将原始指标转化为 ML 特征
# ============================================================

class InferenceMetricFeatureExtractor:
    """GPU 推理指标特征提取器

    原始指标 → 特征向量:
    - 当前值 (level)
    - 变化率 (derivative)
    - 波动性 (rolling std)
    - 相对位置 (percentile)
    - 交互特征 (ratio between metrics)

    特征示例:
    ┌────────────────────────────────────────────────────────┐
    │ Raw: ttft=0.5s, tpot=30ms, kv=0.85, throughput=1200  │
    │                                                        │
    │ Features:                                              │
    │   ttft_value: 0.5                                      │
    │   ttft_delta: +0.05 (较上一点增加 50ms)               │
    │   ttft_rolling_std: 0.08                               │
    │   ttft_percentile: 0.72 (在历史中的位置)               │
    │   tpot_value: 0.03                                     │
    │   kv_value: 0.85                                       │
    │   kv_delta: +0.02                                      │
    │   throughput_value: 1200                                │
    │   throughput_delta: -100                                │
    │   ttft_tpot_ratio: 16.7                                │
    │   kv_throughput_product: 1020                           │
    │   ...                                                  │
    └────────────────────────────────────────────────────────┘
    """

    def __init__(self, window_size: int = 60, feature_names: List[str] = None):
        self.window_size = window_size
        self.feature_names = feature_names or [
            "ttft_p99", "tpot_p99", "kv_cache_usage",
            "throughput", "queue_length", "preemption_rate",
            "gpu_sm_active", "gpu_temp",
        ]
        # 每个原始指标的历史缓冲
        self._buffers: Dict[str, deque] = {
            name: deque(maxlen=window_size) for name in self.feature_names
        }
        self._feature_dim = None  # 特征维度 (首次提取后确定)

    def extract(self, raw_metrics: Dict[str, float]) -> Optional[np.ndarray]:
        """从原始指标提取特征向量

        Args:
            raw_metrics: {metric_name: value}

        Returns:
            特征向量 np.ndarray, 或 None (数据不足)
        """
        # 更新缓冲
        for name in self.feature_names:
            value = raw_metrics.get(name, 0.0)
            self._buffers[name].append(value)

        # 检查数据是否充足
        min_len = min(len(buf) for buf in self._buffers.values())
        if min_len < 10:  # 至少需要 10 个点
            return None

        features = []

        for name in self.feature_names:
            buf = np.array(self._buffers[name])

            # 当前值 (归一化到 [0, 1])
            current = buf[-1]
            features.append(current)

            # 一阶差分 (变化率)
            delta = buf[-1] - buf[-2] if len(buf) >= 2 else 0
            features.append(delta)

            # 滚动标准差 (波动性)
            if len(buf) >= 10:
                rolling_std = np.std(buf[-10:])
            else:
                rolling_std = np.std(buf)
            features.append(rolling_std)

            # 在历史分布中的百分位
            percentile = np.searchsorted(np.sort(buf), current) / len(buf)
            features.append(percentile)

            # 偏离均值的程度
            mean = np.mean(buf)
            std = np.std(buf)
            z_score = (current - mean) / std if std > 0 else 0
            features.append(z_score)

        # 交互特征 (指标间关系)
        ttft = raw_metrics.get("ttft_p99", 0)
        tpot = raw_metrics.get("tpot_p99", 0)
        kv = raw_metrics.get("kv_cache_usage", 0)
        throughput = raw_metrics.get("throughput", 0)
        queue = raw_metrics.get("queue_length", 0)

        # TTFT / TPOT 比值 (正常时应该比较稳定)
        features.append(ttft / max(tpot, 1e-6))
        # KV × Queue (两者同时高 = 严重问题)
        features.append(kv * queue)
        # 吞吐 / (1 + Queue) (排队多但吞吐不变 = 容量不足)
        features.append(throughput / (1 + queue))

        feature_vector = np.array(features, dtype=np.float64)
        # 处理 NaN 和 Inf
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

        self._feature_dim = len(feature_vector)
        return feature_vector

    @property
    def feature_dim(self) -> int:
        return self._feature_dim or 0


# ============================================================
# Isolation Forest 异常检测
# ============================================================

class IsolationForestDetector:
    """Isolation Forest 多维异常检测

    原理:
    - 随机选择特征和分割点构建二叉树
    - 异常点更容易被 "隔离" (路径更短)
    - 路径长度 → 异常分数

    为什么适合 GPU 推理:
    - 不需要标注数据 (无监督)
    - 对多维数据有效 (同时考虑多个指标)
    - 计算效率高 (适合实时检测)
    - 对离群值类型不敏感

    ┌──────────────────────────────────────────────┐
    │         Isolation Forest 示意                 │
    │                                              │
    │   特征空间中:                                 │
    │   ○ ○ ○ ○ ○ ← 正常点 (聚集, 路径长)         │
    │   ○ ○ ○ ○                                    │
    │   ○ ○ ○                                      │
    │                                              │
    │           ★  ← 异常点 (孤立, 路径短)         │
    │                                              │
    │   平均路径长度越短 → 异常分数越高             │
    └──────────────────────────────────────────────┘
    """

    def __init__(
        self,
        contamination: float = 0.05,   # 预估异常比例 (5%)
        n_estimators: int = 100,        # 树的数量
        min_training_samples: int = 200, # 最少训练样本
        retrain_interval: int = 1000,    # 每 N 个样本重训
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.min_training_samples = min_training_samples
        self.retrain_interval = retrain_interval

        self._model = None
        self._training_buffer: List[np.ndarray] = []
        self._sample_count = 0
        self._is_trained = False
        self._feature_scaler_mean = None
        self._feature_scaler_std = None

    def _train(self, data: np.ndarray):
        """训练 Isolation Forest"""
        from sklearn.ensemble import IsolationForest

        # 特征标准化
        self._feature_scaler_mean = np.mean(data, axis=0)
        self._feature_scaler_std = np.std(data, axis=0)
        self._feature_scaler_std[self._feature_scaler_std == 0] = 1.0

        scaled_data = (data - self._feature_scaler_mean) / self._feature_scaler_std

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,  # 使用所有 CPU 核心
        )
        self._model.fit(scaled_data)
        self._is_trained = True

        logger.info(
            f"Isolation Forest trained on {len(data)} samples, "
            f"feature_dim={data.shape[1]}"
        )

    def detect(self, features: np.ndarray) -> Tuple[bool, float]:
        """检测特征向量是否异常

        Args:
            features: 特征向量

        Returns:
            (is_anomaly, anomaly_score)
            anomaly_score: -1 到 0 之间, 越接近 -1 越异常
                          转换为 0-1: score = -decision_function_value
        """
        self._sample_count += 1
        self._training_buffer.append(features)

        # 首次训练
        if not self._is_trained:
            if len(self._training_buffer) >= self.min_training_samples:
                training_data = np.array(self._training_buffer)
                self._train(training_data)
            else:
                return False, 0.0

        # 定期重训 (适应数据分布变化)
        if self._sample_count % self.retrain_interval == 0:
            recent_data = np.array(self._training_buffer[-self.retrain_interval:])
            self._train(recent_data)
            logger.info(f"Isolation Forest retrained at sample {self._sample_count}")

        # 标准化
        scaled = (features - self._feature_scaler_mean) / self._feature_scaler_std
        scaled = scaled.reshape(1, -1)

        # 预测
        prediction = self._model.predict(scaled)[0]  # 1=normal, -1=anomaly
        raw_score = self._model.decision_function(scaled)[0]

        # 转换分数: decision_function 输出越负越异常
        # 转为 0-1 范围: 0=正常, 1=极度异常
        anomaly_score = max(0.0, min(1.0, -raw_score))
        is_anomaly = prediction == -1

        return is_anomaly, round(anomaly_score, 4)


# ============================================================
# Autoencoder 异常检测 (PyTorch)
# ============================================================

class AutoencoderDetector:
    """基于 Autoencoder 的时序异常检测

    原理:
    - 用正常数据训练 Autoencoder (encoder → latent → decoder)
    - 正常数据: 重构误差小
    - 异常数据: 重构误差大 (模型没见过这种模式)

    架构:
    Input (feature_dim × window)
       ↓
    Encoder: Linear(128) → ReLU → Linear(64) → ReLU → Linear(16)
       ↓
    Latent Space (16 维)
       ↓
    Decoder: Linear(64) → ReLU → Linear(128) → ReLU → Linear(feature_dim × window)
       ↓
    Reconstruction

    Loss = MSE(input, reconstruction)
    Anomaly = Loss > threshold (基于训练集分布)

    为什么适合 GPU 推理:
    - 学习时序形态模式 (不只是点值)
    - 可以捕捉 "形状异常" (如周期性被打破)
    - 潜在空间可做可视化
    """

    def __init__(
        self,
        feature_dim: int = 43,
        sequence_length: int = 10,
        latent_dim: int = 16,
        threshold_percentile: float = 95,
        min_training_epochs: int = 50,
        min_training_samples: int = 500,
    ):
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.min_training_epochs = min_training_epochs
        self.min_training_samples = min_training_samples

        self._model = None
        self._threshold = None
        self._is_trained = False
        self._sequence_buffer = deque(maxlen=sequence_length)
        self._training_sequences: List[np.ndarray] = []
        self._loss_history: List[float] = []

    def _build_model(self):
        """构建 Autoencoder 模型"""
        try:
            import torch
            import torch.nn as nn

            input_dim = self.feature_dim * self.sequence_length

            class InferenceAutoencoder(nn.Module):
                def __init__(self, input_dim, latent_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 128),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, latent_dim),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 64),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Linear(128, input_dim),
                    )

                def forward(self, x):
                    latent = self.encoder(x)
                    reconstructed = self.decoder(latent)
                    return reconstructed, latent

            self._model = InferenceAutoencoder(input_dim, self.latent_dim)
            logger.info(f"Autoencoder built: input_dim={input_dim}, latent_dim={self.latent_dim}")
            return True

        except ImportError:
            logger.warning("PyTorch not available, Autoencoder detector disabled")
            return False

    def _train(self, sequences: List[np.ndarray]):
        """训练 Autoencoder"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            if not self._build_model():
                return

        # 准备数据
        data = np.array([s.flatten() for s in sequences], dtype=np.float32)

        # 标准化
        self._data_mean = np.mean(data, axis=0)
        self._data_std = np.std(data, axis=0)
        self._data_std[self._data_std == 0] = 1.0
        data_normalized = (data - self._data_mean) / self._data_std

        tensor_data = torch.FloatTensor(data_normalized)
        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 训练
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.min_training_epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                reconstructed, _ = self._model(batch_x)
                loss = criterion(reconstructed, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # 计算阈值: 基于训练数据的重构误差分布
        self._model.eval()
        with torch.no_grad():
            all_reconstructed, _ = self._model(tensor_data)
            reconstruction_errors = torch.mean(
                (tensor_data - all_reconstructed) ** 2, dim=1
            ).numpy()

        self._threshold = np.percentile(
            reconstruction_errors, self.threshold_percentile
        )
        self._is_trained = True

        logger.info(
            f"Autoencoder trained: {len(sequences)} sequences, "
            f"threshold={self._threshold:.6f}"
        )

    def detect(self, features: np.ndarray) -> Tuple[bool, float]:
        """检测当前时间步是否异常"""
        self._sequence_buffer.append(features)

        # 收集训练数据
        if len(self._sequence_buffer) == self.sequence_length:
            sequence = np.array(self._sequence_buffer)
            self._training_sequences.append(sequence.copy())

        # 首次训练
        if not self._is_trained:
            if len(self._training_sequences) >= self.min_training_samples:
                self._train(self._training_sequences)
            return False, 0.0

        # 检测
        if len(self._sequence_buffer) < self.sequence_length:
            return False, 0.0

        import torch
        sequence = np.array(self._sequence_buffer).flatten().astype(np.float32)
        normalized = (sequence - self._data_mean) / self._data_std

        self._model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(normalized).unsqueeze(0)
            reconstructed, latent = self._model(input_tensor)
            error = torch.mean((input_tensor - reconstructed) ** 2).item()

        self._loss_history.append(error)

        is_anomaly = error > self._threshold
        # 归一化分数
        score = min(1.0, error / (self._threshold * 3)) if self._threshold > 0 else 0

        return is_anomaly, round(score, 4)


# ============================================================
# DBSCAN 行为模式检测
# ============================================================

class BehaviorPatternDetector:
    """基于 DBSCAN 的行为模式变化检测

    不是检测 "单个点是否异常", 而是检测 "行为模式是否变化":
    - 正常模式 A: 低延迟 + 高吞吐 (正常业务时段)
    - 正常模式 B: 中延迟 + 中吞吐 (高峰时段)
    - 异常: 出现了训练阶段未见过的模式

    ┌─────────────────────────────────────────────┐
    │  TPOT                                       │
    │   ↑                                         │
    │   │    Cluster A (正常-低负载)               │
    │   │    ●●●●                                 │
    │   │    ●●●●●                                │
    │   │                                         │
    │   │         Cluster B (正常-高峰)            │
    │   │         ■■■■                             │
    │   │         ■■■■■                            │
    │   │                                         │
    │   │                    ★ 异常点 (新模式!)    │
    │   │                                         │
    │   └──────────────────────────→ Throughput    │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 10):
        self.eps = eps
        self.min_samples = min_samples
        self._training_data: List[np.ndarray] = []
        self._model = None
        self._is_trained = False
        self._cluster_centers = None

    def train(self, feature_vectors: List[np.ndarray]):
        """训练阶段: 学习正常行为模式"""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        data = np.array(feature_vectors)
        self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(data)

        self._model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self._model.fit_predict(scaled_data)

        # 记录聚类中心
        unique_labels = set(labels) - {-1}  # -1 是噪声点
        self._cluster_centers = {}
        for label in unique_labels:
            mask = labels == label
            self._cluster_centers[label] = np.mean(scaled_data[mask], axis=0)

        n_clusters = len(unique_labels)
        n_noise = np.sum(labels == -1)
        self._is_trained = True

        logger.info(
            f"DBSCAN found {n_clusters} behavior patterns, "
            f"{n_noise} noise points in {len(data)} samples"
        )

    def detect(self, features: np.ndarray) -> Tuple[bool, float, int]:
        """检测当前行为是否属于已知模式

        Returns:
            (is_novel, distance_score, nearest_cluster)
        """
        if not self._is_trained or self._cluster_centers is None:
            return False, 0.0, -1

        scaled = self._scaler.transform(features.reshape(1, -1))[0]

        # 计算到最近聚类中心的距离
        min_dist = float('inf')
        nearest_cluster = -1
        for label, center in self._cluster_centers.items():
            dist = np.linalg.norm(scaled - center)
            if dist < min_dist:
                min_dist = dist
                nearest_cluster = label

        # 如果距离超过阈值, 认为是新模式 (异常)
        is_novel = min_dist > self.eps * 3  # 3 倍 eps 作为阈值
        score = min(1.0, min_dist / (self.eps * 6))

        return is_novel, round(score, 4), nearest_cluster


# ============================================================
# 集成 ML 检测器
# ============================================================

class EnsembleMLDetector:
    """集成 ML 异常检测器

    融合 Isolation Forest + Autoencoder + DBSCAN:
    - IF: 多维点异常
    - AE: 时序模式异常
    - DBSCAN: 行为模式变化

    决策逻辑:
    - 任何一个检测器的分数 > 0.8 → CRITICAL
    - 两个以上检测器认为异常 → WARNING
    - 只有一个认为异常 → INFO (可能是误报)
    """

    def __init__(self, feature_dim: int = 43):
        self.feature_extractor = InferenceMetricFeatureExtractor()
        self.isolation_forest = IsolationForestDetector(contamination=0.05)
        self.autoencoder = AutoencoderDetector(feature_dim=feature_dim)
        self.dbscan = BehaviorPatternDetector()
        self._detection_count = 0

    def detect(self, raw_metrics: Dict[str, float]) -> Dict:
        """运行完整 ML 检测流程"""
        self._detection_count += 1

        # 特征提取
        features = self.feature_extractor.extract(raw_metrics)
        if features is None:
            return {"is_anomaly": False, "reason": "insufficient_data"}

        results = {}

        # Isolation Forest
        if_anomaly, if_score = self.isolation_forest.detect(features)
        results["isolation_forest"] = {"is_anomaly": if_anomaly, "score": if_score}

        # Autoencoder
        ae_anomaly, ae_score = self.autoencoder.detect(features)
        results["autoencoder"] = {"is_anomaly": ae_anomaly, "score": ae_score}

        # 融合决策
        scores = [if_score, ae_score]
        anomaly_count = sum(1 for s in [if_anomaly, ae_anomaly] if s)
        max_score = max(scores) if scores else 0
        avg_score = np.mean(scores) if scores else 0

        is_anomaly = anomaly_count >= 1 and max_score > 0.5
        severity = "info"
        if max_score > 0.8:
            severity = "critical"
        elif anomaly_count >= 2:
            severity = "warning"
        elif anomaly_count >= 1:
            severity = "info"

        return {
            "is_anomaly": is_anomaly,
            "score": round(float(avg_score), 4),
            "max_score": round(float(max_score), 4),
            "severity": severity,
            "anomaly_count": anomaly_count,
            "detectors": results,
            "detection_id": self._detection_count,
        }


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=== ML Anomaly Detection Demo ===\n")

    extractor = InferenceMetricFeatureExtractor()
    if_detector = IsolationForestDetector(min_training_samples=100)

    # 生成正常数据训练
    print("Phase 1: Training on normal data...")
    for i in range(250):
        metrics = {
            "ttft_p99": 0.25 + random.gauss(0, 0.03),
            "tpot_p99": 0.025 + random.gauss(0, 0.003),
            "kv_cache_usage": 0.65 + random.gauss(0, 0.05),
            "throughput": 1500 + random.gauss(0, 100),
            "queue_length": max(0, random.gauss(2, 1)),
            "preemption_rate": max(0, random.gauss(0.01, 0.005)),
            "gpu_sm_active": 0.75 + random.gauss(0, 0.05),
            "gpu_temp": 65 + random.gauss(0, 2),
        }
        features = extractor.extract(metrics)
        if features is not None:
            is_anomaly, score = if_detector.detect(features)
            if i % 50 == 0:
                print(f"  Sample {i}: anomaly={is_anomaly}, score={score:.4f}")

    # 注入异常
    print("\nPhase 2: Injecting anomalies...")
    for i in range(20):
        metrics = {
            "ttft_p99": 5.0 + random.gauss(0, 0.5),    # 异常高 TTFT
            "tpot_p99": 0.12 + random.gauss(0, 0.02),   # 异常高 TPOT
            "kv_cache_usage": 0.95 + random.gauss(0, 0.02),  # KV Cache 满
            "throughput": 200 + random.gauss(0, 50),     # 吞吐骤降
            "queue_length": 30 + random.gauss(0, 5),     # 排队严重
            "preemption_rate": 5.0 + random.gauss(0, 1), # 频繁 preemption
            "gpu_sm_active": 0.95 + random.gauss(0, 0.02),
            "gpu_temp": 82 + random.gauss(0, 1),         # GPU 过热
        }
        features = extractor.extract(metrics)
        if features is not None:
            is_anomaly, score = if_detector.detect(features)
            status = "ANOMALY" if is_anomaly else "normal"
            print(f"  Anomaly sample {i}: {status}, score={score:.4f}")
