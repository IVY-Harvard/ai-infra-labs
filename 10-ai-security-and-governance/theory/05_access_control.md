# AI 系统访问控制

## 1. 访问控制概述

### 1.1 AI 系统的访问控制特殊性

```
传统 Web 应用：       用户 → API → 数据库
AI 应用：             用户 → API → 模型 → 工具 → 数据库
                                    ↑        ↑
                              需要控制   需要控制
                              模型访问   工具权限

AI 系统的三维权限：
┌──────────────────────────────────────┐
│                                      │
│     用户维度 × 模型维度 × 数据维度    │
│                                      │
│  用户：谁在访问？（角色、部门、级别）  │
│  模型：访问哪个模型？（GPT-4/Llama）  │
│  数据：能看到哪些数据？（RAG 知识库）  │
│                                      │
└──────────────────────────────────────┘
```

### 1.2 威胁模型

| 威胁 | 描述 | 影响 |
|------|------|------|
| 未授权访问 | 无权限用户使用 AI 服务 | 资源浪费、数据泄露 |
| 权限提升 | 低权限用户获取高权限操作 | 敏感数据访问 |
| API 滥用 | 大量请求消耗配额 | 服务不可用、成本失控 |
| 横向移动 | 通过 AI 工具访问其他系统 | 供应链攻击 |
| 审计缺失 | 无法追踪操作来源 | 合规风险、事后追责困难 |

## 2. RBAC 设计

### 2.1 角色定义

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Set

class Permission(Enum):
    """AI 系统权限枚举"""
    # 模型访问
    MODEL_QUERY = "model:query"           # 查询模型
    MODEL_FINETUNE = "model:finetune"     # 微调模型
    MODEL_DEPLOY = "model:deploy"         # 部署模型
    MODEL_DELETE = "model:delete"         # 删除模型
    
    # 数据访问
    DATA_READ_PUBLIC = "data:read:public"       # 读取公开数据
    DATA_READ_INTERNAL = "data:read:internal"   # 读取内部数据
    DATA_READ_CONFIDENTIAL = "data:read:confidential"  # 读取机密数据
    DATA_WRITE = "data:write"                   # 写入数据
    DATA_DELETE = "data:delete"                 # 删除数据
    
    # 系统管理
    ADMIN_USER = "admin:user"             # 用户管理
    ADMIN_ROLE = "admin:role"             # 角色管理
    ADMIN_AUDIT = "admin:audit"           # 查看审计日志
    ADMIN_POLICY = "admin:policy"         # 策略管理
    
    # API 使用
    API_BASIC = "api:basic"               # 基础 API（低配额）
    API_ADVANCED = "api:advanced"         # 高级 API（高配额）
    API_UNLIMITED = "api:unlimited"       # 无限制 API

@dataclass
class Role:
    """角色定义"""
    name: str
    description: str
    permissions: Set[Permission]
    
    # 配额限制
    max_tokens_per_day: int = 100000
    max_requests_per_minute: int = 10
    allowed_models: Set[str] = field(default_factory=set)

# 预定义角色
ROLES = {
    "viewer": Role(
        name="viewer",
        description="只读用户",
        permissions={Permission.MODEL_QUERY, Permission.DATA_READ_PUBLIC, Permission.API_BASIC},
        max_tokens_per_day=10000,
        max_requests_per_minute=5,
        allowed_models={"llama-3-8b"}
    ),
    "developer": Role(
        name="developer",
        description="开发者",
        permissions={
            Permission.MODEL_QUERY, Permission.MODEL_FINETUNE,
            Permission.DATA_READ_PUBLIC, Permission.DATA_READ_INTERNAL,
            Permission.DATA_WRITE, Permission.API_ADVANCED
        },
        max_tokens_per_day=500000,
        max_requests_per_minute=30,
        allowed_models={"llama-3-8b", "llama-3-70b", "qwen-72b"}
    ),
    "data_scientist": Role(
        name="data_scientist",
        description="数据科学家",
        permissions={
            Permission.MODEL_QUERY, Permission.MODEL_FINETUNE, Permission.MODEL_DEPLOY,
            Permission.DATA_READ_PUBLIC, Permission.DATA_READ_INTERNAL,
            Permission.DATA_READ_CONFIDENTIAL,
            Permission.DATA_WRITE, Permission.API_ADVANCED
        },
        max_tokens_per_day=2000000,
        max_requests_per_minute=60,
        allowed_models={"llama-3-8b", "llama-3-70b", "qwen-72b", "deepseek-v3"}
    ),
    "admin": Role(
        name="admin",
        description="系统管理员",
        permissions=set(Permission),  # 所有权限
        max_tokens_per_day=-1,  # 无限制
        max_requests_per_minute=-1,
        allowed_models=set()  # 空集表示所有模型
    ),
}
```

### 2.2 用户-角色-权限模型

```python
from sqlalchemy import Column, String, Integer, Table, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

# 多对多关联表
user_roles = Table(
    "user_roles", Base.metadata,
    Column("user_id", String, ForeignKey("users.id")),
    Column("role_id", String, ForeignKey("roles.id")),
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    department = Column(String)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Integer, default=1)
    
    roles = relationship("RoleModel", secondary=user_roles, back_populates="users")
    api_keys = relationship("APIKeyModel", back_populates="user")

class RoleModel(Base):
    __tablename__ = "roles"
    
    id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    permissions_json = Column(String)  # JSON 存储权限列表
    max_tokens_per_day = Column(Integer, default=100000)
    max_requests_per_minute = Column(Integer, default=10)
    allowed_models_json = Column(String)  # JSON 存储允许的模型
    
    users = relationship("User", secondary=user_roles, back_populates="roles")
```

## 3. API Key 管理

### 3.1 API Key 生命周期

```
创建 → 分发 → 使用 → 轮换 → 吊销

每个阶段的安全要求：
┌──────┬────────────────────────────────────┐
│ 创建  │ 随机生成、足够长度、绑定用户/角色   │
├──────┼────────────────────────────────────┤
│ 分发  │ 安全通道传输、不在日志中明文记录     │
├──────┼────────────────────────────────────┤
│ 使用  │ HTTPS 传输、Header 传递、不在 URL 中│
├──────┼────────────────────────────────────┤
│ 轮换  │ 定期轮换、支持多 Key 并存过渡       │
├──────┼────────────────────────────────────┤
│ 吊销  │ 即时生效、记录吊销原因和时间        │
└──────┴────────────────────────────────────┘
```

### 3.2 API Key 实现

```python
import secrets
import hashlib
import datetime
from dataclasses import dataclass

@dataclass
class APIKey:
    key_id: str           # 公开标识（前缀）
    key_hash: str         # 存储哈希
    user_id: str          # 关联用户
    name: str             # Key 描述
    permissions: list     # 权限范围
    created_at: datetime.datetime
    expires_at: datetime.datetime
    last_used_at: datetime.datetime = None
    is_active: bool = True

class APIKeyManager:
    """API Key 管理器"""
    
    PREFIX = "aig"  # AI Gateway 前缀
    KEY_LENGTH = 48
    
    def __init__(self, db_session):
        self.db = db_session
    
    def create_key(self, user_id: str, name: str, 
                   permissions: list = None,
                   ttl_days: int = 90) -> tuple:
        """
        创建新的 API Key
        返回: (key_id, full_key) —— full_key 只在创建时返回一次
        """
        # 生成随机 Key
        random_part = secrets.token_urlsafe(self.KEY_LENGTH)
        key_id = f"{self.PREFIX}_{secrets.token_hex(4)}"
        full_key = f"{key_id}_{random_part}"
        
        # 只存储哈希
        key_hash = self._hash_key(full_key)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions or [],
            created_at=datetime.datetime.utcnow(),
            expires_at=datetime.datetime.utcnow() + datetime.timedelta(days=ttl_days),
        )
        
        self.db.save(api_key)
        return key_id, full_key
    
    def validate_key(self, full_key: str) -> APIKey:
        """验证 API Key，返回关联信息"""
        key_hash = self._hash_key(full_key)
        
        # 从数据库查找
        api_key = self.db.find_by_hash(key_hash)
        if not api_key:
            return None
        
        # 检查是否有效
        if not api_key.is_active:
            return None
        if api_key.expires_at < datetime.datetime.utcnow():
            return None
        
        # 更新最后使用时间
        api_key.last_used_at = datetime.datetime.utcnow()
        self.db.update(api_key)
        
        return api_key
    
    def revoke_key(self, key_id: str, reason: str = ""):
        """吊销 API Key"""
        api_key = self.db.find_by_id(key_id)
        if api_key:
            api_key.is_active = False
            self.db.update(api_key)
            # 记录吊销日志
            self._log_revocation(key_id, reason)
    
    def rotate_key(self, old_key_id: str) -> tuple:
        """轮换 Key：创建新 Key，旧 Key 在过渡期后自动失效"""
        old_key = self.db.find_by_id(old_key_id)
        if not old_key:
            raise ValueError(f"Key not found: {old_key_id}")
        
        # 创建新 Key
        new_id, new_full_key = self.create_key(
            user_id=old_key.user_id,
            name=f"{old_key.name} (rotated)",
            permissions=old_key.permissions
        )
        
        # 旧 Key 设置 24 小时过渡期
        old_key.expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        self.db.update(old_key)
        
        return new_id, new_full_key
    
    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _log_revocation(self, key_id: str, reason: str):
        print(f"[AUDIT] Key revoked: {key_id}, reason: {reason}")
```

## 4. Token 级别的配额与限流

### 4.1 限流策略

```python
import time
import redis
from dataclasses import dataclass

@dataclass
class QuotaConfig:
    """配额配置"""
    max_requests_per_minute: int = 60
    max_tokens_per_day: int = 1000000
    max_tokens_per_request: int = 4096
    max_concurrent_requests: int = 5

class TokenRateLimiter:
    """基于 Redis 的 Token 限流器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def check_and_consume(self, user_id: str, token_count: int,
                          config: QuotaConfig) -> dict:
        """
        检查配额并消费
        返回: {"allowed": bool, "reason": str, "remaining": int}
        """
        now = time.time()
        pipe = self.redis.pipeline()
        
        # 1. 检查请求频率（滑动窗口）
        rpm_key = f"rate:{user_id}:rpm"
        pipe.zremrangebyscore(rpm_key, 0, now - 60)
        pipe.zcard(rpm_key)
        pipe.zadd(rpm_key, {str(now): now})
        pipe.expire(rpm_key, 120)
        
        # 2. 检查日 Token 配额
        day_key = f"quota:{user_id}:tokens:{self._today()}"
        pipe.get(day_key)
        
        # 3. 检查并发数
        concurrent_key = f"concurrent:{user_id}"
        pipe.get(concurrent_key)
        
        results = pipe.execute()
        
        # 解析结果
        current_rpm = results[1]
        daily_tokens = int(results[4] or 0)
        concurrent = int(results[5] or 0)
        
        # 检查请求频率
        if config.max_requests_per_minute > 0 and current_rpm >= config.max_requests_per_minute:
            return {
                "allowed": False,
                "reason": f"Rate limit exceeded: {current_rpm}/{config.max_requests_per_minute} RPM",
                "remaining_rpm": 0,
                "retry_after": 60
            }
        
        # 检查单次请求 Token 限制
        if token_count > config.max_tokens_per_request:
            return {
                "allowed": False,
                "reason": f"Token limit per request: {token_count}/{config.max_tokens_per_request}",
                "remaining_tokens": config.max_tokens_per_request
            }
        
        # 检查日 Token 配额
        if config.max_tokens_per_day > 0 and daily_tokens + token_count > config.max_tokens_per_day:
            return {
                "allowed": False,
                "reason": f"Daily token quota exceeded",
                "remaining_tokens": max(0, config.max_tokens_per_day - daily_tokens)
            }
        
        # 检查并发数
        if config.max_concurrent_requests > 0 and concurrent >= config.max_concurrent_requests:
            return {
                "allowed": False,
                "reason": f"Too many concurrent requests",
                "concurrent": concurrent
            }
        
        # 消费配额
        self.redis.incrby(day_key, token_count)
        self.redis.expire(day_key, 86400)
        self.redis.incr(concurrent_key)
        self.redis.expire(concurrent_key, 300)
        
        return {
            "allowed": True,
            "remaining_tokens": config.max_tokens_per_day - daily_tokens - token_count,
            "remaining_rpm": config.max_requests_per_minute - current_rpm - 1
        }
    
    def release_concurrent(self, user_id: str):
        """释放并发计数"""
        key = f"concurrent:{user_id}"
        self.redis.decr(key)
    
    def _today(self) -> str:
        return time.strftime("%Y%m%d")
```

## 5. 审计日志

### 5.1 审计日志设计

```python
import json
import uuid
import datetime
from enum import Enum

class AuditEventType(Enum):
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    API_KEY_CREATED = "apikey.created"
    API_KEY_REVOKED = "apikey.revoked"
    MODEL_QUERY = "model.query"
    MODEL_QUERY_BLOCKED = "model.query.blocked"
    DATA_ACCESS = "data.access"
    PERMISSION_CHANGE = "permission.change"
    GUARDRAIL_TRIGGERED = "guardrail.triggered"
    POLICY_VIOLATION = "policy.violation"

class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
    
    def log(self, event_type: AuditEventType, user_id: str,
            details: dict = None, request_id: str = None):
        """记录审计事件"""
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type.value,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "user_id": user_id,
            "request_id": request_id or str(uuid.uuid4()),
            "details": details or {},
            "source_ip": self._get_source_ip(),
        }
        
        # 确保不记录敏感信息
        event = self._sanitize(event)
        self.storage.append(event)
    
    def _sanitize(self, event: dict) -> dict:
        """清理敏感信息"""
        sensitive_keys = {"password", "api_key", "token", "secret"}
        if "details" in event:
            for key in list(event["details"].keys()):
                if key.lower() in sensitive_keys:
                    event["details"][key] = "***REDACTED***"
        return event
    
    def _get_source_ip(self) -> str:
        return "0.0.0.0"  # 实际中从请求上下文获取
```

## 6. SOC2 / ISO27001 对 AI 系统的要求

### 6.1 SOC2 Trust Service Criteria

```
SOC2 对 AI 系统的适用要求：

1. Security（安全）
   - 访问控制机制
   - 加密传输和存储
   - 入侵检测
   → AI: 模型 API 的认证授权、模型权重加密

2. Availability（可用性）
   - SLA 承诺
   - 灾备方案
   - 容量规划
   → AI: GPU 集群高可用、模型服务降级方案

3. Processing Integrity（处理完整性）
   - 输入验证
   - 输出准确性
   - 错误处理
   → AI: 护栏系统、幻觉检测、输出验证

4. Confidentiality（机密性）
   - 数据分类
   - 访问限制
   - 数据加密
   → AI: PII 保护、模型知识隔离

5. Privacy（隐私）
   - 数据收集通知
   - 使用目的限制
   - 数据保留策略
   → AI: 训练数据合规、用户对话隐私
```

### 6.2 合规映射

```
ISO 27001 控制项 → AI 系统实施：

A.9 访问控制
  → RBAC 模型、API Key 管理、MFA

A.10 密码学
  → 模型权重加密、传输加密（TLS 1.3）

A.12 运营安全
  → 审计日志、变更管理、容量监控

A.14 系统开发安全
  → 安全的 ML Pipeline、代码审查

A.16 信息安全事件管理
  → 安全事件检测和响应流程

A.18 合规
  → 定期审计、合规报告
```

## 7. 实施建议

### 7.1 渐进式实施路线

```
Phase 1（1-2 周）：基础访问控制
  - API Key 认证
  - 基础 RBAC
  - 请求限流

Phase 2（2-4 周）：精细化控制
  - Token 级配额
  - 模型级权限
  - 数据访问控制

Phase 3（4-8 周）：审计与合规
  - 完整审计日志
  - 合规报告
  - 安全监控

Phase 4（持续）：运营优化
  - 自动异常检测
  - 策略自动化
  - 定期安全审计
```

### 7.2 常见陷阱

1. **过度授权**：默认给所有用户 admin 权限
2. **硬编码密钥**：API Key 写在代码中
3. **缺少轮换**：API Key 从不轮换
4. **日志不全**：只记录成功请求
5. **忽略内部威胁**：只防外部不防内部
