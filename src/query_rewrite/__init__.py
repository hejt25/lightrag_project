"""
Query Rewrite Module
====================
针对保险问答场景的Query改写模块

集成：
- 用户画像 (UserProfile)
- 会话历史 (ConversationHistory)
- 用户行为 (UserBehavior)
- Query改写器 (QueryRewriter)
"""

from .query_rewriter import (
    InsuranceQueryRewriter,
    RewriteResult,
    UserProfile,
    ConversationHistoryManager,
    UserBehaviorTracker
)

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    income_level: Optional[str] = None  # low, medium, high
    occupation: Optional[str] = None
    marital_status: Optional[str] = None
    has_children: bool = False
    children_ages: List[int] = field(default_factory=list)
    insurance_history: List[str] = field(default_factory=list)  # 已购险种
    insurance_needs: List[str] = field(default_factory=list)  # 潜在需求
    risk_preference: Optional[str] = None  # conservative, moderate, aggressive
    budget_range: Optional[str] = None  # budget, standard, premium
    source_channel: Optional[str] = None  # 渠道来源

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "age": self.age,
            "gender": self.gender,
            "income_level": self.income_level,
            "occupation": self.occupation,
            "marital_status": self.marital_status,
            "has_children": self.has_children,
            "children_ages": self.children_ages,
            "insurance_history": self.insurance_history,
            "insurance_needs": self.insurance_needs,
            "risk_preference": self.risk_preference,
            "budget_range": self.budget_range,
            "source_channel": self.source_channel
        }


@dataclass
class ConversationMessage:
    """会话消息"""
    role: str  # user, assistant
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None  # 意图识别结果
    entities: List[Dict[str, Any]] = field(default_factory=list)  # 实体提取结果

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
            "entities": self.entities
        }


@dataclass
class ConversationSession:
    """会话session"""
    session_id: str
    user_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    context_summary: str = ""  # 对话上下文摘要

    def add_message(self, message: ConversationMessage):
        self.messages.append(message)
        self.updated_at = datetime.now()
        self._update_summary()

    def _update_summary(self):
        """更新上下文摘要"""
        if len(self.messages) >= 2:
            last_user_msgs = [m.content for m in self.messages if m.role == "user"][-3:]
            self.context_summary = " | ".join(last_user_msgs)

    def get_history(self, n: int = 5) -> List[ConversationMessage]:
        """获取最近n条消息"""
        return self.messages[-n:] if n > 0 else self.messages

    def get_user_messages(self) -> List[str]:
        """获取所有用户消息"""
        return [m.content for m in self.messages if m.role == "user"]


class ConversationHistoryManager:
    """会话历史管理器"""

    def __init__(self, max_history: int = 10):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_history = max_history

    def get_session(self, session_id: str, user_id: str) -> ConversationSession:
        """获取或创建session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                user_id=user_id
            )
        return self.sessions[session_id]

    def add_message(self, session_id: str, user_id: str, role: str, content: str,
                    intent: Optional[str] = None, entities: Optional[List] = None):
        """添加消息"""
        session = self.get_session(session_id, user_id)
        message = ConversationMessage(
            role=role,
            content=content,
            intent=intent,
            entities=entities or []
        )
        session.add_message(message)

        # 限制历史长度
        if len(session.messages) > self.max_history * 2:
            session.messages = session.messages[-self.max_history * 2:]

    def get_context(self, session_id: str, n: int = 5) -> str:
        """获取对话上下文"""
        if session_id not in self.sessions:
            return ""
        session = self.sessions[session_id]
        return session.context_summary

    def get_last_query(self, session_id: str) -> Optional[str]:
        """获取最后一条用户消息"""
        if session_id not in self.sessions:
            return None
        messages = self.sessions[session_id].get_user_messages()
        return messages[-1] if messages else None


class UserBehaviorTracker:
    """用户行为追踪器"""

    def __init__(self):
        self.behaviors: Dict[str, List[Dict[str, Any]]] = {}

    def track(self, user_id: str, behavior_type: str, data: Dict[str, Any]):
        """记录行为"""
        if user_id not in self.behaviors:
            self.behaviors[user_id] = []

        self.behaviors[user_id].append({
            "type": behavior_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })

        # 保留最近100条行为记录
        if len(self.behaviors[user_id]) > 100:
            self.behaviors[user_id] = self.behaviors[user_id][-100:]

    def get_behaviors(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取用户行为记录"""
        return self.behaviors.get(user_id, [])[-limit:]

    def get_insurance_viewed(self, user_id: str) -> List[str]:
        """获取用户查看过的保险产品"""
        viewed = []
        for b in self.get_behaviors(user_id):
            if b["type"] == "view_product":
                viewed.append(b["data"].get("product_name", ""))
        return viewed

    def get_clicked_products(self, user_id: str) -> List[str]:
        """获取用户点击过的产品"""
        clicked = []
        for b in self.get_behaviors(user_id):
            if b["type"] == "click_product":
                clicked.append(b["data"].get("product_name", ""))
        return clicked


# 保险领域常用实体和术语
INSURANCE_ENTITIES = {
    "product_types": [
        "寿险", "健康险", "医疗险", "意外险", "重疾险",
        "年金险", "万能险", "分红险", "车险", "家财险", "旅游险"
    ],
    "coverage": [
        "保额", "保费", "保障期限", "缴费期限", "等待期",
        "免赔额", "赔付比例", "续保"
    ],
    "claim": [
        "理赔", "报案", "索赔", "理赔材料", "理赔流程"
    ],
    "comparison_keywords": [
        "哪个好", "推荐", "比较", "区别", "多少钱",
        "值得买吗", "性价比", "可靠吗"
    ]
}
