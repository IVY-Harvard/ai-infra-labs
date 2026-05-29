"""
存储后端抽象层

统一 S3/MinIO/本地文件系统的访问接口。
"""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, BinaryIO
from dataclasses import dataclass


@dataclass
class StorageObject:
    """存储对象元信息"""
    key: str
    size: int
    last_modified: float
    checksum: str = ""
    content_type: str = "application/octet-stream"


class StorageBackend(ABC):
    """存储后端抽象基类"""

    @abstractmethod
    def put(self, key: str, data: bytes) -> bool:
        """上传对象"""
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """下载对象"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除对象"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查对象是否存在"""
        pass

    @abstractmethod
    def list_objects(self, prefix: str = "") -> list:
        """列出对象"""
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Optional[StorageObject]:
        """获取对象元信息"""
        pass


class LocalStorageBackend(StorageBackend):
    """本地文件系统存储后端"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _full_path(self, key: str) -> Path:
        return self.root_dir / key

    def put(self, key: str, data: bytes) -> bool:
        path = self._full_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "wb") as f:
                f.write(data)
            return True
        except IOError:
            return False

    def get(self, key: str) -> Optional[bytes]:
        path = self._full_path(key)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str) -> bool:
        path = self._full_path(key)
        if path.exists():
            os.remove(path)
            return True
        return False

    def exists(self, key: str) -> bool:
        return self._full_path(key).exists()

    def list_objects(self, prefix: str = "") -> list:
        prefix_path = self.root_dir / prefix
        if not prefix_path.exists():
            return []
        results = []
        for path in prefix_path.rglob("*"):
            if path.is_file():
                key = str(path.relative_to(self.root_dir))
                results.append(key)
        return results

    def get_metadata(self, key: str) -> Optional[StorageObject]:
        path = self._full_path(key)
        if not path.exists():
            return None
        stat = path.stat()
        return StorageObject(
            key=key,
            size=stat.st_size,
            last_modified=stat.st_mtime,
        )


class S3StorageBackend(StorageBackend):
    """S3/MinIO 存储后端"""

    def __init__(self, endpoint: str, bucket: str,
                 access_key: str, secret_key: str,
                 region: str = "us-east-1"):
        self.bucket = bucket
        try:
            import boto3
            from botocore.config import Config
            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
                config=Config(signature_version="s3v4"),
            )
            # 确保 bucket 存在
            try:
                self.client.head_bucket(Bucket=bucket)
            except Exception:
                self.client.create_bucket(Bucket=bucket)
        except ImportError:
            raise ImportError("请安装 boto3: pip install boto3")

    def put(self, key: str, data: bytes) -> bool:
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
            return True
        except Exception as e:
            print(f"S3 put error: {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except Exception:
            return None

    def delete(self, key: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list_objects(self, prefix: str = "") -> list:
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix
            )
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception:
            return []

    def get_metadata(self, key: str) -> Optional[StorageObject]:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            return StorageObject(
                key=key,
                size=response["ContentLength"],
                last_modified=response["LastModified"].timestamp(),
                content_type=response.get("ContentType", ""),
            )
        except Exception:
            return None


def create_backend(config: dict) -> StorageBackend:
    """根据配置创建存储后端"""
    backend_type = config.get("type", "local")

    if backend_type == "local":
        return LocalStorageBackend(config["root_dir"])
    elif backend_type == "s3":
        return S3StorageBackend(
            endpoint=config["endpoint"],
            bucket=config["bucket"],
            access_key=config["access_key"],
            secret_key=config["secret_key"],
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
