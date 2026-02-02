"""
Query Rewriter for Insurance Domain
====================================
保险领域Query改写器

核心功能：
1. 融合用户画像进行个性化改写
2. 结合会话历史进行指代消解和上下文补全
3. 利用用户行为进行意图扩展
4. 保险领域术语优化
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


# ===== Data Classes =====

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    income_level: Optional[str] = None
    occupation: Optional[str] = None
    marital_status: Optional[str] = None
    has_children: bool = False
    children_ages: List[int] = field(default_factory=list)
    insurance_history: List[str] = field(default_factory=list)
    insurance_needs: List[str] = field(default_factory=list)
    risk_preference: Optional[str] = None
    budget_range: Optional[str] = None
    source_channel: Optional[str] = None

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
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationSession:
    """会话session"""
    session_id: str
    user_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    context_summary: str = ""

    def add_message(self, message: ConversationMessage):
        self.messages.append(message)
        self.updated_at = datetime.now()
        self._update_summary()

    def _update_summary(self):
        if len(self.messages) >= 2:
            last_user_msgs = [m.content for m in self.messages if m.role == "user"][-3:]
            self.context_summary = " | ".join(last_user_msgs)

    def get_user_messages(self) -> List[str]:
        return [m.content for m in self.messages if m.role == "user"]


class ConversationHistoryManager:
    """会话历史管理器"""

    def __init__(self, max_history: int = 10):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_history = max_history

    def get_session(self, session_id: str, user_id: str) -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                user_id=user_id
            )
        return self.sessions[session_id]

    def add_message(self, session_id: str, user_id: str, role: str, content: str,
                    intent: Optional[str] = None, entities: Optional[List] = None):
        session = self.get_session(session_id, user_id)
        message = ConversationMessage(
            role=role,
            content=content,
            intent=intent,
            entities=entities or []
        )
        session.add_message(message)
        if len(session.messages) > self.max_history * 2:
            session.messages = session.messages[-self.max_history * 2:]

    def get_context(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return ""
        return self.sessions[session_id].context_summary

    def get_last_query(self, session_id: str) -> Optional[str]:
        if session_id not in self.sessions:
            return None
        messages = self.sessions[session_id].get_user_messages()
        return messages[-1] if messages else None


class UserBehaviorTracker:
    """用户行为追踪器"""

    def __init__(self):
        self.behaviors: Dict[str, List[Dict[str, Any]]] = {}

    def track(self, user_id: str, behavior_type: str, data: Dict[str, Any]):
        if user_id not in self.behaviors:
            self.behaviors[user_id] = []
        self.behaviors[user_id].append({
            "type": behavior_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        if len(self.behaviors[user_id]) > 100:
            self.behaviors[user_id] = self.behaviors[user_id][-100:]

    def get_behaviors(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        return self.behaviors.get(user_id, [])[-limit:]


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


@dataclass
class RewriteResult:
    """改写结果"""
    original_query: str
    rewritten_query: str
    rewrite_type: str  # expansion, specification, context, personalization
    confidence: float
    context_used: Dict[str, Any]  # 使用的上下文信息
    suggestions: List[str] = field(default_factory=list)  # 后续建议


class InsuranceTermExpander:
    """保险术语扩展器"""

    def __init__(self):
        # 常见缩写和全称映射
        self.abbreviations = {
            "重疾": "重大疾病保险",
            "医疗险": "医疗保险",
            "意外险": "意外伤害保险",
            "寿险": "人寿保险",
            "年金": "年金保险",
            "车险": "机动车保险",
            "保单": "保险单",
            "理赔": "保险理赔",
        }

        # 同义词扩展
        self.synonyms = {
            "哪个好": ["推荐", "比较", "选择"],
            "多少钱": ["价格", "保费", "费用"],
            "值得买吗": ["性价比", "是否划算", "推荐购买"],
            "可靠吗": ["安全性", "信誉", "理赔能力"],
            "区别": ["不同", "差异", "对比"],
        }

    def expand(self, query: str) -> str:
        """扩展术语"""
        expanded = query
        for abbr, full in self.abbreviations.items():
            if abbr in expanded and full not in expanded:
                expanded = expanded.replace(abbr, f"{abbr}({full})")
        return expanded

    def expand_synonyms(self, query: str) -> str:
        """扩展同义词"""
        expanded = query
        for word, synonyms in self.synonyms.items():
            if word in expanded:
                for syn in synonyms:
                    if syn not in expanded:
                        expanded += f" 或 {syn}"
        return expanded


class ContextResolver:
    """上下文解析器 - 处理指代消解"""

    def __init__(self):
        self.pronouns = {
            "它": ["该保险", "这个产品", "这份保险"],
            "这个": ["这个保险产品"],
            "那个": ["那个保险"],
            "这种": ["这类保险"],
            "便宜的": ["保费较低的", "价格实惠的"],
            "保障好的": ["保障全面的", "保障责任好的"],
        }

    def resolve(self, query: str, context: str) -> str:
        """解析指代词"""
        resolved = query
        for pronoun, references in self.pronouns.items():
            if pronoun in query and context:
                # 选择最相关的引用
                best_ref = references[0]
                for ref in references:
                    if ref in context:
                        best_ref = ref
                        break
                resolved = resolved.replace(pronoun, best_ref)
        return resolved


class IntentClassifier:
    """意图分类器 - 保险领域"""

    INTENTS = {
        "product_inquiry": ["什么保险", "有哪些", "哪个产品", "推荐"],
        "price_inquiry": ["多少钱", "价格", "保费", "费用"],
        "comparison": ["哪个好", "区别", "对比", "比较"],
        "claim_guidance": ["理赔", "怎么赔", "报案", "索赔"],
        "coverage_inquiry": ["保额", "保障", "责任", "赔什么"],
        "policy_detail": ["条款", "细则", "注意", "除外"],
        "purchase_guide": ["怎么买", "投保", "购买", "缴费"],
        "renewal_inquiry": ["续保", "续费", "继续保"],
    }

    def classify(self, query: str) -> Tuple[str, float]:
        """分类意图"""
        best_intent = "general_inquiry"
        best_score = 0.0

        query_lower = query.lower()
        for intent, keywords in self.INTENTS.items():
            for keyword in keywords:
                if keyword in query:
                    score = len(keyword) / len(query)
                    if score > best_score:
                        best_intent = intent
                        best_score = score

        return best_intent, best_score if best_score > 0 else 0.5


class UserProfileEnricher:
    """用户画像 enrichment 器"""

    # 年龄段对应的常见保险需求
    AGE_BASED_NEEDS = {
        (0, 18): ["医疗险", "意外险", "教育金"],
        (18, 30): ["重疾险", "医疗险", "意外险"],
        (30, 50): ["重疾险", "寿险", "医疗险", "教育金"],
        (50, 65): ["医疗险", "意外险", "养老险"],
        (65, 100): ["医疗险", "意外险", "防癌险"],
    }

    def enrich_query(self, query: str, profile: UserProfile) -> str:
        """根据用户画像丰富查询"""
        enriched = query

        if profile.age:
            age_needs = self._get_age_based_needs(profile.age)
            # 如果用户提到"适合"或"推荐"，添加年龄相关需求
            if "适合" in query or "推荐" in query or "买" in query:
                if age_needs:
                    # 检查是否已经包含相关需求
                    for need in age_needs:
                        if need not in query:
                            enriched += f" {need}"

        # 根据预算范围调整查询
        if profile.budget_range:
            if profile.budget_range == "budget":
                enriched = self._add_budget_constraint(enriched, "低保费", "经济型")
            elif profile.budget_range == "premium":
                enriched = self._add_budget_constraint(enriched, "高保额", "全面保障")

        # 根据已购保险排除
        if profile.insurance_history:
            # 添加"除外已购"或"新增"意图
            if "再买" in query or "还买" in query or "再加" in query:
                existing = "、".join(profile.insurance_history[:2])
                enriched = f"{enriched}(已有{existing})"

        return enriched

    def _get_age_based_needs(self, age: int) -> List[str]:
        """根据年龄获取推荐需求"""
        for age_range, needs in self.AGE_BASED_NEEDS.items():
            if age_range[0] <= age < age_range[1]:
                return needs
        return []

    def _add_budget_constraint(self, query: str, *keywords) -> str:
        """添加预算约束"""
        return f"{query}，{' '.join(keywords)}"


class BehaviorIntentExpander:
    """基于用户行为扩展意图"""

    def __init__(self):
        self.behavior_intent_map = {
            "view_product": ["详细了解", "保障范围", "理赔条件"],
            "click_product": ["对比", "评价", "口碑"],
            "search": ["推荐", "哪个好", "最新"],
            "compare": ["区别", "性价比", "对比"],
            "ask_price": ["多少钱", "保费", "费用"],
        }

    def expand(self, query: str, behaviors: List[Dict[str, Any]]) -> str:
        """根据行为扩展查询意图"""
        expanded = query

        # 统计行为类型
        behavior_types = {}
        for b in behaviors:
            btype = b.get("type", "")
            behavior_types[btype] = behavior_types.get(btype, 0) + 1

        # 根据主要行为类型扩展
        if behavior_types:
            main_behavior = max(behavior_types, key=behavior_types.get)
            expansions = self.behavior_intent_map.get(main_behavior, [])

            if expansions and ("想了解" in query or "怎么" in query or "？" in query):
                for exp in expansions[:2]:
                    if exp not in expanded:
                        expanded += f" {exp}"

        return expanded


class InsuranceQueryRewriter:
    """保险领域Query改写器 - 主类"""

    def __init__(self):
        self.term_expander = InsuranceTermExpander()
        self.context_resolver = ContextResolver()
        self.intent_classifier = IntentClassifier()
        self.profile_enricher = UserProfileEnricher()
        self.behavior_expander = BehaviorIntentExpander()

        # 会话历史和行为追踪（可在外部注入）
        self.history_manager = ConversationHistoryManager()
        self.behavior_tracker = UserBehaviorTracker()

    def rewrite(
        self,
        query: str,
        user_id: str,
        session_id: str,
        user_profile: Optional[UserProfile] = None
    ) -> RewriteResult:
        """
        改写Query

        Args:
            query: 原始查询
            user_id: 用户ID
            session_id: 会话ID
            user_profile: 用户画像（可选）

        Returns:
            RewriteResult: 改写结果
        """
        original = query
        context_used = {}

        # 1. 术语扩展
        query = self.term_expander.expand(query)
        query = self.term_expander.expand_synonyms(query)

        # 2. 上下文解析（指代消解）
        context = self.history_manager.get_context(session_id)
        if context:
            query = self.context_resolver.resolve(query, context)
            context_used["conversation_history"] = True

        # 3. 意图分类
        intent, confidence = self.intent_classifier.classify(query)
        context_used["intent"] = intent

        # 4. 用户画像 enrichment
        if user_profile:
            query = self.profile_enricher.enrich_query(query, user_profile)
            context_used["user_profile"] = user_profile.to_dict()

        # 5. 行为意图扩展
        behaviors = self.behavior_tracker.get_behaviors(user_id)
        if behaviors:
            query = self.behavior_expander.expand(query, behaviors)
            context_used["behaviors_count"] = len(behaviors)

        # 6. 生成改写建议
        suggestions = self._generate_suggestions(query, intent, user_profile)

        # 确定改写类型
        rewrite_type = self._determine_rewrite_type(original, query)

        result = RewriteResult(
            original_query=original,
            rewritten_query=query,
            rewrite_type=rewrite_type,
            confidence=confidence,
            context_used=context_used,
            suggestions=suggestions
        )

        logger.info(f"Query rewritten: '{original}' -> '{query}' (type: {rewrite_type})")

        return result

    def _generate_suggestions(
        self,
        query: str,
        intent: str,
        profile: Optional[UserProfile]
    ) -> List[str]:
        """生成后续建议"""
        suggestions = []

        if intent == "product_inquiry":
            suggestions.append("可补充：年龄、预算、保障需求")
        elif intent == "price_inquiry":
            suggestions.append("建议提供：年缴还是月缴、缴费期限偏好")
        elif intent == "claim_guidance":
            suggestions.append("可告知：出险时间、具体事故情况")
        elif intent == "comparison":
            suggestions.append("可说明：对比维度（价格/保障/公司）")

        if profile and profile.risk_preference:
            if profile.risk_preference == "conservative":
                suggestions.append("推荐：大公司、稳健型产品")
            elif profile.risk_preference == "aggressive":
                suggestions.append("推荐：高性价比、创新型产品")

        return suggestions

    def _determine_rewrite_type(self, original: str, rewritten: str) -> str:
        """确定改写类型"""
        if len(rewritten) > len(original) * 1.5:
            return "expansion"
        elif len(rewritten) > len(original):
            return "specification"
        elif original != rewritten:
            return "context"
        else:
            return "personalization"

    # ===== 外部接口方法 =====

    def set_history_manager(self, manager: ConversationHistoryManager):
        """设置会话历史管理器"""
        self.history_manager = manager

    def set_behavior_tracker(self, tracker: UserBehaviorTracker):
        """设置行为追踪器"""
        self.behavior_tracker = tracker

    def add_user_behavior(self, user_id: str, behavior_type: str, data: Dict[str, Any]):
        """添加用户行为"""
        self.behavior_tracker.track(user_id, behavior_type, data)

    def add_conversation_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        entities: Optional[List] = None
    ):
        """添加会话消息"""
        self.history_manager.add_message(
            session_id, user_id, role, content, intent, entities
        )

    def get_user_context(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """获取用户完整上下文"""
        return {
            "user_id": user_id,
            "session_id": session_id,
            "history_summary": self.history_manager.get_context(session_id),
            "last_query": self.history_manager.get_last_query(session_id),
            "recent_behaviors": self.behavior_tracker.get_behaviors(user_id)
        }
