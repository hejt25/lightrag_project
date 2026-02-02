"""
Prompt Manager
==============
Prompt管理和版本控制

功能：
1. 版本管理 - 记录每次变更
2. A/B测试支持 - 多版本并行
3. 变更检测 - 自动发现变更
4. 导出导入 - 支持配置文件
5. 性能追踪 - 记录使用效果
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """变更类型"""
    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    UNCHANGED = "unchanged"


@dataclass
class PromptVersion:
    """Prompt版本记录"""
    version: str
    template_name: str
    template_dict: Dict[str, Any]
    created_at: str
    created_by: str = "system"
    changelog: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "template_name": self.template_name,
            "template_dict": self.template_dict,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "changelog": self.changelog,
            "metrics": self.metrics
        }


@dataclass
class PromptChange:
    """变更记录"""
    component_type: str
    change_type: ChangeType
    old_hash: Optional[str]
    new_hash: str
    details: str


@dataclass
class PromptMetrics:
    """Prompt使用指标"""
    template_name: str
    template_version: str
    timestamp: str
    query: str
    response_time_ms: int
    token_count: int
    user_feedback: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class PromptManager:
    """
    Prompt管理中枢

    使用示例：
    manager = PromptManager()
    manager.register_template(my_template)

    # 版本控制
    manager.save_version("v1.0.0", "Initial version")

    # 变更检测
    changes = manager.detect_changes(my_template)

    # 性能追踪
    manager.record_metrics(PromptMetrics(...))
    """

    def __init__(self, storage_dir: str = "./prompt_versions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.templates: Dict[str, Dict[str, Any]] = {}
        self.versions: Dict[str, List[PromptVersion]] = {}
        self.metrics: List[PromptMetrics] = []
        self.current_versions: Dict[str, str] = {}  # template_name -> version

        self._load_from_disk()

    def _load_from_disk(self):
        """从磁盘加载"""
        versions_file = self.storage_dir / "versions.json"
        if versions_file.exists():
            with open(versions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.versions = {
                    k: [PromptVersion(**v) for v in vs]
                    for k, vs in data.get("versions", {}).items()
                }

        current_file = self.storage_dir / "current_versions.json"
        if current_file.exists():
            with open(current_file, 'r', encoding='utf-8') as f:
                self.current_versions = json.load(f)

    def _save_to_disk(self):
        """保存到磁盘"""
        versions_file = self.storage_dir / "versions.json"
        with open(versions_file, 'w', encoding='utf-8') as f:
            json.dump({
                "versions": {
                    k: [v.to_dict() if hasattr(v, 'to_dict') else v for v in vs]
                    for k, vs in self.versions.items()
                }
            }, f, ensure_ascii=False, indent=2)

        current_file = self.storage_dir / "current_versions.json"
        with open(current_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_versions, f, ensure_ascii=False)

    def register_template(self, template_dict: Dict[str, Any]):
        """注册模板"""
        name = template_dict["name"]
        self.templates[name] = template_dict
        logger.info(f"Registered template: {name}")

    def save_version(
        self,
        template_name: str,
        version: str,
        template_dict: Dict[str, Any],
        changelog: str = "",
        created_by: str = "system"
    ) -> PromptVersion:
        """
        保存版本快照

        Args:
            template_name: 模板名称
            version: 版本号
            template_dict: 模板内容
            changelog: 变更日志
            created_by: 创建者

        Returns:
            PromptVersion: 版本记录
        """
        if template_name not in self.versions:
            self.versions[template_name] = []

        prompt_version = PromptVersion(
            version=version,
            template_name=template_name,
            template_dict=template_dict,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            changelog=changelog
        )

        self.versions[template_name].append(prompt_version)
        self.current_versions[template_name] = version
        self._save_to_disk()

        logger.info(f"Saved version {version} for template {template_name}")
        return prompt_version

    def get_version(self, template_name: str, version: str) -> Optional[PromptVersion]:
        """获取指定版本"""
        if template_name not in self.versions:
            return None
        for v in self.versions[template_name]:
            if v.version == version:
                return v
        return None

    def get_current_version(self, template_name: str) -> Optional[PromptVersion]:
        """获取当前版本"""
        if template_name not in self.current_versions:
            return None
        return self.get_version(template_name, self.current_versions[template_name])

    def list_versions(self, template_name: str) -> List[str]:
        """列出所有版本号"""
        if template_name not in self.versions:
            return []
        return [v.version for v in self.versions[template_name]]

    def rollback(self, template_name: str, version: str) -> bool:
        """回滚到指定版本"""
        old_version = self.get_version(template_name, version)
        if old_version is None:
            return False

        self.current_versions[template_name] = version
        self._save_to_disk()
        logger.info(f"Rolled back {template_name} to {version}")
        return True

    def detect_changes(
        self,
        current_template: Dict[str, Any]
    ) -> List[PromptChange]:
        """
        检测模板变更

        Args:
            current_template: 当前模板

        Returns:
            List[PromptChange]: 变更列表
        """
        name = current_template["name"]
        changes = []

        if name not in self.templates:
            return [PromptChange(
                component_type="template",
                change_type=ChangeType.ADDED,
                old_hash=None,
                new_hash=self._hash_dict(current_template),
                details="New template added"
            )]

        old_template = self.templates[name]

        # 比较组件
        old_components = old_template.get("components", {})
        new_components = current_template.get("components", {})

        all_keys = set(old_components.keys()) | set(new_components.keys())

        for key in all_keys:
            old_hash = old_components.get(key, {}).get("content", "")
            new_hash = new_components.get(key, {}).get("content", "")

            if old_hash == new_hash:
                continue

            if old_hash and not new_hash:
                change_type = ChangeType.REMOVED
            elif not old_hash and new_hash:
                change_type = ChangeType.ADDED
            else:
                change_type = ChangeType.MODIFIED

            changes.append(PromptChange(
                component_type=key,
                change_type=change_type,
                old_hash=old_hash[:8] if old_hash else None,
                new_hash=new_hash[:8],
                details=f"{key} was {change_type.value}"
            ))

        return changes

    def _hash_dict(self, d: Dict) -> str:
        """计算字典哈希"""
        import hashlib
        content = json.dumps(d, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def record_metrics(self, metrics: PromptMetrics):
        """记录使用指标"""
        self.metrics.append(metrics)

        # 定期清理（保留最近1000条）
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]

    def get_metrics_summary(
        self,
        template_name: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """获取指标汇总"""
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=days)

        filtered = [
            m for m in self.metrics
            if datetime.fromisoformat(m.timestamp) >= cutoff
            and (template_name is None or m.template_name == template_name)
        ]

        if not filtered:
            return {"count": 0, "avg_response_time_ms": 0, "success_rate": 0}

        success_count = sum(1 for m in filtered if m.success)
        response_times = [m.response_time_ms for m in filtered if m.success]

        return {
            "count": len(filtered),
            "success_count": success_count,
            "success_rate": success_count / len(filtered) * 100 if filtered else 0,
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "total_tokens": sum(m.token_count for m in filtered)
        }

    def export_template(self, template_name: str, path: str):
        """导出模板到文件"""
        version = self.get_current_version(template_name)
        if version is None:
            raise ValueError(f"Template not found: {template_name}")

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(version.template_dict, f, ensure_ascii=False, indent=2)

    def import_template(self, path: str, set_current: bool = True):
        """从文件导入模板"""
        with open(path, 'r', encoding='utf-8') as f:
            template_dict = json.load(f)

        name = template_dict["name"]
        version = template_dict.get("version", "1.0.0")

        self.register_template(template_dict)
        if set_current:
            self.current_versions[name] = version

        logger.info(f"Imported template: {name} v{version}")
        return template_dict

    def generate_changelog(
        self,
        old_version: str,
        new_version: str,
        template_name: str
    ) -> str:
        """生成变更日志"""
        changes = self.detect_changes(self.templates.get(template_name, {}))

        lines = [f"# Changelog: {template_name}", f"", f"From {old_version} to {new_version}", f""]

        if not changes:
            lines.append("- No changes")
        else:
            for change in changes:
                lines.append(f"- [{change.change_type.value}] {change.component_type}")

        return "\n".join(lines)
