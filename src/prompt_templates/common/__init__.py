"""
Common Prompt Templates
========================
通用Prompt模板
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from ..core.base import (
    BasePromptTemplate,
    PromptComponent,
    ComponentType,
    OutputFormat,
    FewShotExample
)


def create_general_persona() -> PromptComponent:
    """创建通用角色"""
    return PromptComponent(
        name="general_persona",
        content="""你是一位知识渊博、乐于助人的助手。
- 回答问题清晰准确
- 善于用通俗易懂的语言解释复杂概念
- 保持友好、专业的态度""",
        component_type=ComponentType.PERSONA,
        description="通用助手角色"
    )


def create_general_system_prompt() -> PromptComponent:
    """创建通用系统提示"""
    return PromptComponent(
        name="general_system",
        content="请根据提供的参考资料回答用户的问题。",
        component_type=ComponentType.SYSTEM_PROMPT,
        description="通用系统提示"
    )


def create_general_constraints() -> PromptComponent:
    """创建通用约束"""
    return PromptComponent(
        name="general_constraints",
        content="""要求：
1. 基于提供的参考资料回答
2. 如信息不足，坦诚告知
3. 回答简洁、重点突出
4. 不编造或猜测信息""",
        component_type=ComponentType.CONSTRAINT,
        description="通用回答约束"
    )


def create_rag_context_template() -> PromptComponent:
    """创建RAG上下文模板"""
    return PromptComponent(
        name="rag_context",
        content="""参考信息：
{context}""",
        component_type=ComponentType.CONTEXT_TEMPLATE,
        variables=["context"],
        description="RAG上下文模板"
    )


def create_rag_question_template() -> PromptComponent:
    """创建RAG问题模板"""
    return PromptComponent(
        name="rag_question",
        content="""用户问题：{question}

请根据以上参考信息回答：""",
        component_type=ComponentType.QUESTION_TEMPLATE,
        variables=["question"],
        description="RAG问题模板"
    )


class GeneralRAGPrompt(BasePromptTemplate):
    """通用RAG问答模板"""

    name = "GeneralRAGPrompt"
    version = "1.0.0"
    description = "通用检索增强问答"

    def _register_default_components(self):
        self.add_component(create_general_persona())
        self.add_component(create_general_system_prompt())
        self.add_component(create_general_constraints())
        self.add_component(create_rag_context_template())
        self.add_component(create_rag_question_template())


# ===== 对话模板 =====

def create_conversation_context_template() -> PromptComponent:
    """创建对话上下文模板"""
    return PromptComponent(
        name="conversation_context",
        content="""背景信息（供参考）：
{context}""",
        component_type=ComponentType.CONTEXT_TEMPLATE,
        variables=["context"],
        description="对话上下文模板"
    )


def create_conversation_history_template() -> PromptComponent:
    """创建对话历史模板"""
    return PromptComponent(
        name="conversation_history",
        content="""对话历史：
{history}""",
        component_type=ComponentType.DOMAIN_KNOWLEDGE,
        variables=["history"],
        description="对话历史模板"
    )


def create_conversation_question_template() -> PromptComponent:
    """创建对话问题模板"""
    return PromptComponent(
        name="conversation_question",
        content="""当前问题：{question}

请结合对话历史和背景信息回答：""",
        component_type=ComponentType.QUESTION_TEMPLATE,
        variables=["question", "history"],
        description="对话问题模板"
    )


class ConversationalPrompt(BasePromptTemplate):
    """对话式问答模板"""

    name = "ConversationalPrompt"
    version = "1.0.0"
    description = "支持对话历史的问答"

    def _register_default_components(self):
        self.add_component(create_general_persona())
        self.add_component(create_general_system_prompt())
        self.add_component(create_general_constraints())
        self.add_component(create_conversation_context_template())
        self.add_component(create_conversation_history_template())
        self.add_component(create_conversation_question_template())
