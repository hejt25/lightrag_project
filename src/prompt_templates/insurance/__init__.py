"""
Insurance Prompt Templates
==========================
保险领域专用Prompt模板

模板类型：
1. InsuranceProductPrompt - 产品咨询
2. InsuranceClaimPrompt - 理赔指导
3. InsuranceComparisonPrompt - 产品对比
4. InsuranceRecommendationPrompt - 产品推荐
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


# ===== 组件工厂函数 =====

def create_persona(role: str = "保险顾问") -> PromptComponent:
    """创建角色设定组件"""
    return PromptComponent(
        name="insurance_persona",
        content=f"""你是一位专业、贴心的{role}。
- 具备丰富的保险专业知识
- 善于用通俗易懂的语言解释复杂的保险概念
- 始终以客户利益为中心，提供客观建议
- 回答问题时条理清晰，重点突出""",
        component_type=ComponentType.PERSONA,
        description="保险顾问角色设定"
    )


def create_system_prompt() -> PromptComponent:
    """创建系统提示组件"""
    return PromptComponent(
        name="insurance_system",
        content="请根据提供的保险产品信息和专业知识，回答用户的问题。",
        component_type=ComponentType.SYSTEM_PROMPT,
        description="系统提示"
    )


def create_insurance_constraints() -> PromptComponent:
    """创建保险领域约束组件"""
    return PromptComponent(
        name="insurance_constraints",
        content="""回答时请注意：
1. 严格基于提供的参考资料回答，不要编造信息
2. 如信息不足，明确告知用户需要进一步确认
3. 涉及具体数字（保费、保额等）需注明信息来源和时效
4. 提醒用户仔细阅读保险条款
5. 不做夸大宣传，客观呈现产品特点""",
        component_type=ComponentType.CONSTRAINT,
        variables=["context", "question"],
        description="保险领域回答约束"
    )


def create_product_context_template() -> PromptComponent:
    """创建产品信息上下文模板"""
    return PromptComponent(
        name="product_context",
        content="""参考产品信息：
{context}""",
        component_type=ComponentType.CONTEXT_TEMPLATE,
        variables=["context"],
        description="产品信息上下文模板"
    )


def create_question_template() -> PromptComponent:
    """创建问题模板"""
    return PromptComponent(
        name="insurance_question",
        content="""用户问题：{question}

请根据以上信息回答用户问题：""",
        component_type=ComponentType.QUESTION_TEMPLATE,
        variables=["question"],
        description="问题呈现模板"
    )


# ===== 保险产品咨询模板 =====

class InsuranceProductPrompt(BasePromptTemplate):
    """保险产品咨询模板"""

    name = "InsuranceProductPrompt"
    version = "1.0.0"
    description = "保险产品信息咨询"

    def _register_default_components(self):
        self.add_component(create_persona("保险产品顾问"))
        self.add_component(create_system_prompt())
        self.add_component(create_insurance_constraints())
        self.add_component(create_product_context_template())
        self.add_component(create_question_template())


# ===== 理赔指导模板 =====

def create_claim_context_template() -> PromptComponent:
    """创建理赔信息上下文模板"""
    return PromptComponent(
        name="claim_context",
        content="""参考理赔信息：
{context}""",
        component_type=ComponentType.CONTEXT_TEMPLATE,
        variables=["context"],
        description="理赔信息上下文模板"
    )


def create_claim_constraints() -> PromptComponent:
    """创建理赔约束组件"""
    return PromptComponent(
        name="claim_constraints",
        content="""理赔指导要求：
1. 清晰说明理赔流程和所需材料
2. 提醒用户注意理赔时效
3. 说明可能的拒赔情况
4. 建议用户妥善保存相关单据
5. 如情况复杂，建议咨询专业理赔人员""",
        component_type=ComponentType.CONSTRAINT,
        description="理赔指导约束"
    )


class InsuranceClaimPrompt(BasePromptTemplate):
    """理赔指导模板"""

    name = "InsuranceClaimPrompt"
    version = "1.0.0"
    description = "保险理赔指导"

    def _register_default_components(self):
        self.add_component(create_persona("理赔顾问"))
        self.add_component(create_system_prompt())
        self.add_component(create_claim_constraints())
        self.add_component(create_claim_context_template())
        self.add_component(create_question_template())


# ===== 产品对比模板 =====

def create_comparison_context_template() -> PromptComponent:
    """创建对比上下文模板"""
    return PromptComponent(
        name="comparison_context",
        content="""待对比产品信息：
{context}""",
        component_type=ComponentType.CONTEXT_TEMPLATE,
        variables=["context"],
        description="产品对比上下文模板"
    )


def create_comparison_constraints() -> PromptComponent:
    """创建对比约束组件"""
    return PromptComponent(
        name="comparison_constraints",
        content="""产品对比要求：
1. 从保障范围、保费、保额、等待期、除外责任等维度对比
2. 突出各自的优势和不足
3. 客观中立，不偏袒任何产品
4. 根据用户需求给出针对性建议
5. 如对比维度不全，予以说明""",
        component_type=ComponentType.CONSTRAINT,
        description="产品对比约束"
    )


def create_comparison_output_format() -> PromptComponent:
    """创建对比输出格式组件"""
    return PromptComponent(
        name="comparison_output_format",
        content="""建议使用表格形式呈现对比结果，格式如下：

| 维度 | 产品A | 产品B |
|------|-------|-------|
| 保障范围 | xxx | xxx |
| 保额 | xxx | xxx |
| 保费 | xxx | xxx |
| ... | ... | ... |

最后给出综合评价和选购建议。""",
        component_type=ComponentType.OUTPUT_FORMAT,
        description="对比输出格式"
    )


class InsuranceComparisonPrompt(BasePromptTemplate):
    """产品对比模板"""

    name = "InsuranceComparisonPrompt"
    version = "1.0.0"
    description = "保险产品对比"

    def _register_default_components(self):
        self.add_component(create_persona("保险产品分析师"))
        self.add_component(create_system_prompt())
        self.add_component(create_comparison_constraints())
        self.add_component(create_comparison_context_template())
        self.add_component(create_question_template())
        self.add_component(create_comparison_output_format())


# ===== 产品推荐模板 =====

def create_recommendation_context_template() -> PromptComponent:
    """创建推荐上下文模板"""
    return PromptComponent(
        name="recommendation_context",
        content="""符合用户需求的保险产品：
{context}""",
        component_type=ComponentType.CONTEXT_TEMPLATE,
        variables=["context"],
        description="推荐产品上下文模板"
    )


def create_recommendation_constraints() -> PromptComponent:
    """创建推荐约束组件"""
    return PromptComponent(
        name="recommendation_constraints",
        content="""产品推荐要求：
1. 充分考虑用户的年龄、预算、健康状况、风险偏好
2. 优先推荐匹配用户需求的产品
3. 说明推荐理由
4. 提醒注意健康告知和除外责任
5. 建议用户根据实际情况选择""",
        component_type=ComponentType.CONSTRAINT,
        description="产品推荐约束"
    )


def create_recommendation_output_format() -> PromptComponent:
    """创建推荐输出格式组件"""
    return PromptComponent(
        name="recommendation_output_format",
        content="""请按以下格式推荐：

## 推荐产品

**产品名称**：xxx
**推荐理由**：xxx
**保障亮点**：xxx
**预计保费**：xxx/年
**适合人群**：xxx

## 不推荐或暂缓考虑的原因

xxx

## 投保建议

1. xxx
2. xxx
3. xxx""",
        component_type=ComponentType.OUTPUT_FORMAT,
        description="推荐输出格式"
    )


class InsuranceRecommendationPrompt(BasePromptTemplate):
    """产品推荐模板"""

    name = "InsuranceRecommendationPrompt"
    version = "1.0.0"
    description = "保险产品推荐"

    def _register_default_components(self):
        self.add_component(create_persona("资深保险顾问"))
        self.add_component(create_system_prompt())
        self.add_component(create_recommendation_constraints())
        self.add_component(create_recommendation_context_template())
        self.add_component(create_question_template())
        self.add_component(create_recommendation_output_format())
