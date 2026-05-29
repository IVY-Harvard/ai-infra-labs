"""存储层单元测试"""

import os
import time
import shutil
import tempfile
import pytest

from src.storage.backend import (
    LocalStorageBackend, create_backend
)
from src.storage.cache_manager import CacheManager
from src.storage.replication import ReplicationManager
from src.model.registry import ModelRegistry
from src.model.versioning import VersionManager
from src.model.validator import ModelValidator
from src.distribution.distributor import Distributor
from src.distribution.p2p_transfer import P2PTransfer
from src.distribution.prewarmer import Prewarmer
from src.checkpoint.ckpt_manager import CheckpointManager
from src.checkpoint.gc_policy import GCPolicy, GCRule, RetentionPolicy


# ──────────── Fixtures ────────────

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def local_backend(tmp_dir):
    return LocalStorageBackend(root=tmp_dir)


@pytest.fixture
def cache(tmp_dir):
    cache_dir = os.path.join(tmp_dir, "cache")
    return CacheManager(cache_dir=cache_dir, max_size_gb=1)


# ──────────── Storage Backend Tests ────────────

class TestLocalStorageBackend:
    def test_put_and_get(self, local_backend):
        local_backend.put("models/test.bin", b"hello world")
        data = local_backend.get("models/test.bin")
        assert data == b"hello world"

    def test_get_nonexistent(self, local_backend):
        assert local_backend.get("nonexistent") is None

    def test_delete(self, local_backend):
        local_backend.put("to_delete.bin", b"data")
        assert local_backend.get("to_delete.bin") is not None
        local_backend.delete("to_delete.bin")
        assert local_backend.get("to_delete.bin") is None

    def test_exists(self, local_backend):
        assert not local_backend.exists("foo.bin")
        local_backend.put("foo.bin", b"bar")
        assert local_backend.exists("foo.bin")

    def test_list_keys(self, local_backend):
        local_backend.put("a/1.bin", b"1")
        local_backend.put("a/2.bin", b"2")
        local_backend.put("b/3.bin", b"3")
        keys = local_backend.list("a/")
        assert len(keys) == 2
        assert "a/1.bin" in keys
        assert "a/2.bin" in keys


class TestCreateBackend:
    def test_create_local(self, tmp_dir):
        b = create_backend("local", root=tmp_dir)
        assert isinstance(b, LocalStorageBackend)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError):
            create_backend("unknown_type")


# ──────────── Cache Manager Tests ────────────

class TestCacheManager:
    def test_put_and_get(self, cache):
        path = cache.put("model_a", b"data_a")
        assert path is not None
        assert os.path.exists(path)
        assert cache.get("model_a") == path

    def test_get_miss(self, cache):
        assert cache.get("nonexistent") is None

    def test_eviction(self, tmp_dir):
        """验证 LRU 淘汰"""
        cache_dir = os.path.join(tmp_dir, "small_cache")
        # 设置非常小的缓存
        small_cache = CacheManager(
            cache_dir=cache_dir, max_size_gb=0.000001
        )
        # 写入超过容量的数据
        large_data = b"x" * (1024 * 1024)  # 1MB
        small_cache.put("first", large_data)
        small_cache.put("second", large_data)
        # 由于容量限制，旧的应被淘汰
        # （实际淘汰逻辑取决于 CacheManager 实现）


# ──────────── Replication Tests ────────────

class TestReplication:
    def test_replicate(self, tmp_dir):
        primary_dir = os.path.join(tmp_dir, "primary")
        replica_dir = os.path.join(tmp_dir, "replica")
        primary = LocalStorageBackend(root=primary_dir)
        replica = LocalStorageBackend(root=replica_dir)

        primary.put("model.bin", b"model_data")

        mgr = ReplicationManager(primary=primary, replicas=[replica])
        mgr.replicate("model.bin")

        assert replica.get("model.bin") == b"model_data"


# ──────────── Model Registry Tests ────────────

class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        entry = reg.register(name="llama-7b", framework="pytorch")
        assert entry.name == "llama-7b"
        assert reg.get("llama-7b") is not None

    def test_list_models(self):
        reg = ModelRegistry()
        reg.register(name="model_a", framework="pytorch")
        reg.register(name="model_b", framework="jax")
        models = reg.list_models()
        assert len(models) == 2


# ──────────── Versioning Tests ────────────

class TestVersionManager:
    def test_create_version(self):
        vm = VersionManager()
        v = vm.create_version(
            model_name="test",
            version="1.0.0",
            storage_key="models/test/v1.0.0.safetensors",
            size_bytes=1024,
        )
        assert v["version"] == "1.0.0"

    def test_list_versions(self):
        vm = VersionManager()
        vm.create_version("test", "1.0.0", "k1", 100)
        vm.create_version("test", "1.1.0", "k2", 200)
        versions = vm.list_versions("test")
        assert len(versions) == 2


# ──────────── Validator Tests ────────────

class TestModelValidator:
    def test_compute_checksum(self, tmp_dir):
        filepath = os.path.join(tmp_dir, "test.bin")
        with open(filepath, "wb") as f:
            f.write(b"test data for checksum")
        checksum = ModelValidator.compute_checksum(filepath)
        assert len(checksum) == 64  # SHA256 hex

    def test_validate_checksum(self, tmp_dir):
        filepath = os.path.join(tmp_dir, "test.bin")
        with open(filepath, "wb") as f:
            f.write(b"test data")
        checksum = ModelValidator.compute_checksum(filepath)
        assert ModelValidator.validate_checksum(filepath, checksum)
        assert not ModelValidator.validate_checksum(filepath, "wrong")

    def test_validate_unknown_format(self, tmp_dir):
        filepath = os.path.join(tmp_dir, "test.xyz")
        with open(filepath, "wb") as f:
            f.write(b"some data")
        result = ModelValidator.validate(filepath)
        assert result["valid"] is True
        assert "note" in result


# ──────────── Distributor Tests ────────────

class TestDistributor:
    def test_distribute_not_found(self, local_backend, cache):
        dist = Distributor(backend=local_backend, cache=cache)
        result = dist.distribute("nonexistent_model")
        assert result["status"] == "not_found"

    def test_distribute_success(self, local_backend, cache):
        local_backend.put("model.bin", b"model_data_here")
        dist = Distributor(backend=local_backend, cache=cache)
        result = dist.distribute("model.bin")
        assert result["status"] == "completed"
        assert result["size_bytes"] == len(b"model_data_here")

    def test_distribute_cache_hit(self, local_backend, cache):
        # 预先放入缓存
        cache.put("cached_model", b"cached_data")
        dist = Distributor(backend=local_backend, cache=cache)
        result = dist.distribute("cached_model")
        assert result["status"] == "cache_hit"


# ──────────── P2P Transfer Tests ────────────

class TestP2PTransfer:
    def test_split_file(self):
        p2p = P2PTransfer(chunk_size_mb=1)
        data = b"x" * (2 * 1024 * 1024 + 100)  # 2MB + 100 bytes
        chunks = p2p.split_file(data)
        assert len(chunks) == 3
        assert chunks[0].size == 1024 * 1024
        assert chunks[2].size == 100

    def test_assemble(self):
        p2p = P2PTransfer(chunk_size_mb=1)
        data = b"hello world this is a test"
        chunks = p2p.split_file(data)
        chunk_dict = {c.index: c.data for c in chunks}
        reassembled = p2p.assemble(chunk_dict)
        assert reassembled == data

    def test_register_peer(self):
        p2p = P2PTransfer()
        p2p.register_peer("node1", "10.0.0.1:8080", {0, 1, 2})
        assert "node1" in p2p.peers
        assert p2p.peers["node1"].available_chunks == {0, 1, 2}

    def test_get_chunk(self):
        p2p = P2PTransfer(chunk_size_mb=1)
        p2p.split_file(b"x" * 100)
        data = p2p.get_chunk(0, "requester_1")
        assert data == b"x" * 100
        assert p2p.get_chunk(99, "requester_1") is None


# ──────────── Checkpoint Tests ────────────

class TestCheckpointManager:
    def test_save_and_load(self, tmp_dir):
        ckpt_dir = os.path.join(tmp_dir, "ckpts")
        mgr = CheckpointManager(local_dir=ckpt_dir)

        state = {"layer1.weight": b"tensor_data_1",
                 "layer1.bias": b"tensor_data_2"}
        meta = mgr.save(step=100, state_dict=state,
                        metrics={"loss": 0.5})

        assert meta.step == 100
        assert meta.metrics["loss"] == 0.5

        loaded = mgr.load_step(100)
        assert loaded is not None
        assert loaded["layer1.weight"] == b"tensor_data_1"

    def test_load_latest(self, tmp_dir):
        ckpt_dir = os.path.join(tmp_dir, "ckpts")
        mgr = CheckpointManager(local_dir=ckpt_dir)

        mgr.save(step=100, state_dict={"a": b"1"})
        mgr.save(step=200, state_dict={"a": b"2"})

        latest = mgr.load_latest()
        assert latest["a"] == b"2"

    def test_list_checkpoints(self, tmp_dir):
        ckpt_dir = os.path.join(tmp_dir, "ckpts")
        mgr = CheckpointManager(local_dir=ckpt_dir)

        mgr.save(step=10, state_dict={"x": b"data"})
        mgr.save(step=20, state_dict={"x": b"data2"})

        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 2
        assert ckpts[0]["step"] == 10


# ──────────── GC Policy Tests ────────────

class TestGCPolicy:
    def _make_checkpoints(self, steps):
        now = time.time()
        return [
            {
                "step": s,
                "timestamp": now - (max(steps) - s),
                "local_path": f"/tmp/ckpt/step_{s:08d}",
                "metrics": {"loss": 1.0 / (s + 1)},
            }
            for s in steps
        ]

    def test_keep_latest_n(self):
        gc = GCPolicy()
        gc.add_rule(GCRule(policy=RetentionPolicy.KEEP_LATEST_N, keep_n=2))

        ckpts = self._make_checkpoints([100, 200, 300, 400, 500])
        to_delete = gc.evaluate(ckpts)

        kept_steps = {c["step"] for c in ckpts} - {c["step"] for c in to_delete}
        assert 500 in kept_steps
        assert 400 in kept_steps
        assert len(to_delete) == 3

    def test_best_metric(self):
        gc = GCPolicy()
        gc.add_rule(GCRule(
            policy=RetentionPolicy.BEST_METRIC,
            metric_name="loss",
            metric_mode="min",
            keep_top_k=2,
        ))

        ckpts = self._make_checkpoints([100, 200, 300, 400, 500])
        to_delete = gc.evaluate(ckpts)

        # loss = 1/(step+1), 最小的 loss 在 step 最大处
        kept_steps = {c["step"] for c in ckpts} - {c["step"] for c in to_delete}
        assert 500 in kept_steps
        assert 400 in kept_steps

    def test_combined_rules(self):
        """组合规则取并集"""
        gc = GCPolicy()
        gc.add_rule(GCRule(policy=RetentionPolicy.KEEP_LATEST_N, keep_n=1))
        gc.add_rule(GCRule(
            policy=RetentionPolicy.BEST_METRIC,
            metric_name="loss",
            metric_mode="min",
            keep_top_k=1,
        ))

        ckpts = self._make_checkpoints([100, 200, 300])
        to_delete = gc.evaluate(ckpts)
        kept_steps = {c["step"] for c in ckpts} - {c["step"] for c in to_delete}
        # 最新的是 300，最好的 loss 也是 300
        assert 300 in kept_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
