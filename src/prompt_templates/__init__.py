"""
Prompt Templates Module
=======================
专业、模块化的Prompt模板系统

设计原则：
1. 责任清晰模块化 - 分离系统提示、约束条件、示例、输出格式
2. 易于维护和迭代 - 支持版本管理、配置化、动态组合
3. 场景适配 - 支持通用和保险领域专用模板

模块结构：
- core/           - 核心抽象和基类
- components/     - 可复用的Prompt组件
- insurance/      - 保险领域专用模板
- common/         - 通用模板
- manager.py      - Prompt管理和版本控制
- builder.py      - Prompt构建器
"""

from .core.base import (
    PromptComponent,
    BasePromptTemplate,
    OutputFormat,
    FewShotExample,
    ComponentType
)
from .core.builder import PromptBuilder
from .manager import PromptManager
from .insurance import (
    InsuranceProductPrompt,
    InsuranceClaimPrompt,
    InsuranceComparisonPrompt,
    InsuranceRecommendationPrompt
)
from .common import (
    GeneralRAGPrompt,
    ConversationalPrompt
)

__version__ = "1.0.0"

__all__ = [
    # Core
    "PromptComponent",
    "BasePromptTemplate",
    "OutputFormat",
    "FewShotExample",
    "ComponentType",
    # Builder
    "PromptBuilder",
    "RenderedPrompt",
    # Manager
    "PromptManager",
    # Insurance
    "InsuranceProductPrompt",
    "InsuranceClaimPrompt",
    "InsuranceComparisonPrompt",
    "InsuranceRecommendationPrompt",
    # Common
    "GeneralRAGPrompt",
    "ConversationalPrompt"
]
