"""
Core Base Classes
=================
Prompt系统的核心抽象和基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import json
import hashlib


class ComponentType(Enum):
    """组件类型枚举"""
    PERSONA = "persona"
    SYSTEM_PROMPT = "system_prompt"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CONSTRAINT = "constraint"
    FEW_SHOT = "few_shot"
    OUTPUT_FORMAT = "output_format"
    CONTEXT_TEMPLATE = "context_template"
    QUESTION_TEMPLATE = "question_template"


@dataclass
class PromptComponent:
    """
    Prompt组件 - 最小可复用单元

    特点：
    - 单一职责 - 每个组件只负责一个方面
    - 可组合 - 组件可以自由组合成完整Prompt
    - 可版本化 - 支持版本管理和回滚
    """

    name: str
    content: str
    component_type: ComponentType
    version: str = "1.0.0"
    description: str = ""
    variables: List[str] = field(default_factory=list)  # 模板变量
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def __str__(self) -> str:
        return self.content

    def render(self, **kwargs) -> str:
        """渲染组件，替换模板变量"""
        rendered = self.content
        for key, value in kwargs.items():
            if key in self.variables:
                rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered

    def get_hash(self) -> str:
        """获取内容哈希，用于变更检测"""
        content = f"{self.name}:{self.version}:{self.content}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class OutputFormat:
    """
    输出格式定义

    支持：
    - JSON结构化输出
    - Markdown格式化
    - 自由文本
    """

    name: str
    format_type: str  # "json", "markdown", "text", "xml"
    schema: Optional[Dict[str, Any]] = None  # JSON Schema
    template: str = ""
    example: str = ""

    def to_prompt(self) -> str:
        """转换为Prompt文本"""
        if self.format_type == "json":
            return f"""请按照以下JSON格式输出：
```json
{json.dumps(self.schema, ensure_ascii=False, indent=2)}
```

示例：
```json
{self.example}
```"""
        elif self.format_type == "markdown":
            return f"""请按照以下格式组织回答：
{self.template}"""
        else:
            return ""


@dataclass
class FewShotExample:
    """少样本示例"""
    question: str
    answer: str
    context: Optional[str] = None
    explanation: Optional[str] = None

    def to_prompt(self, include_context: bool = True) -> str:
        parts = [f"问题：{self.question}"]
        if include_context and self.context:
            parts.insert(0, f"参考信息：{self.context}")
        parts.append(f"答案：{self.answer}")
        return "\n".join(parts)


class BasePromptTemplate(ABC):
    """
    Prompt模板抽象基类

    模板由多个组件组合而成：
    - Persona - 角色设定
    - System Prompt - 系统指令
    - Domain Knowledge - 领域知识
    - Constraints - 回答约束
    - Few-Shot Examples - 任务示例
    - Output Format - 输出格式
    - Context Template - 上下文呈现
    - Question Template - 问题呈现
    """

    name: str = "BaseTemplate"
    version: str = "1.0.0"
    description: str = ""

    def __init__(self):
        self.components: Dict[ComponentType, PromptComponent] = {}
        self._register_default_components()

    @abstractmethod
    def _register_default_components(self):
        """注册默认组件 - 子类实现"""
        pass

    def add_component(self, component: PromptComponent, override: bool = True):
        """添加组件"""
        if component.component_type in self.components and not override:
            return
        self.components[component.component_type] = component

    def remove_component(self, component_type: ComponentType):
        """移除组件"""
        if component_type in self.components:
            del self.components[component_type]

    def get_component(self, component_type: ComponentType) -> Optional[PromptComponent]:
        """获取组件"""
        return self.components.get(component_type)

    def get_all_components(self) -> List[PromptComponent]:
        """获取所有组件"""
        return list(self.components.values())

    def render(self, context: str, question: str, **kwargs) -> str:
        """渲染完整Prompt"""
        rendered_components = {}
        for ctype, component in self.components.items():
            rendered_components[ctype] = component.render(
                context=context,
                question=question,
                **kwargs
            )

        prompt_parts = []

        # 1. Persona
        if ComponentType.PERSONA in rendered_components:
            prompt_parts.append(rendered_components[ComponentType.PERSONA])

        # 2. System Prompt
        if ComponentType.SYSTEM_PROMPT in rendered_components:
            prompt_parts.append(rendered_components[ComponentType.SYSTEM_PROMPT])

        # 3. Domain Knowledge
        if ComponentType.DOMAIN_KNOWLEDGE in rendered_components:
            prompt_parts.append(rendered_components[ComponentType.DOMAIN_KNOWLEDGE])

        # 4. Context
        if ComponentType.CONTEXT_TEMPLATE in rendered_components and context:
            prompt_parts.append(
                rendered_components[ComponentType.CONTEXT_TEMPLATE].format(context=context)
            )

        # 5. Constraints
        if ComponentType.CONSTRAINT in rendered_components:
            prompt_parts.append(rendered_components[ComponentType.CONSTRAINT])

        # 6. Question
        if ComponentType.QUESTION_TEMPLATE in rendered_components:
            prompt_parts.append(
                rendered_components[ComponentType.QUESTION_TEMPLATE].format(question=question)
            )
        else:
            prompt_parts.append(f"用户问题：{question}")

        # 7. Output Format
        if ComponentType.OUTPUT_FORMAT in rendered_components:
            prompt_parts.append(rendered_components[ComponentType.OUTPUT_FORMAT])

        # 8. Few-Shot Examples
        if ComponentType.FEW_SHOT in rendered_components:
            prompt_parts.append(rendered_components[ComponentType.FEW_SHOT])

        return "\n\n".join(prompt_parts)

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "components": {
                ct.value: {
                    "name": c.name,
                    "content": c.content,
                    "version": c.version
                }
                for ct, c in self.components.items()
            }
        }

    def get_version_info(self) -> Dict[str, str]:
        """获取版本信息"""
        return {
            "template_name": self.name,
            "template_version": self.version,
            "components": {
                ct.value: c.get_hash()
                for ct, c in self.components.items()
            }
        }
