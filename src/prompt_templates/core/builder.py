"""
Prompt Builder
==============
Prompt构建器 - 动态组合和构建Prompt
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from ..core.base import (
    BasePromptTemplate,
    PromptComponent,
    ComponentType,
    FewShotExample
)


@dataclass
class RenderedPrompt:
    """渲染后的Prompt"""
    full_prompt: str
    system_prompt: str
    user_prompt: str
    template_used: str
    template_version: str
    context_length: int
    question_length: int


class PromptBuilder:
    """
    Prompt构建器

    功能：
    - 动态选择和组合模板
    - 支持条件插入组件
    - 支持动态Few-Shot示例选择
    - 支持Prompt参数调优
    """

    def __init__(self):
        self.templates: Dict[str, BasePromptTemplate] = {}
        self.default_template: Optional[str] = None

    def register_template(self, template: BasePromptTemplate, set_default: bool = False):
        """注册模板"""
        self.templates[template.name] = template
        if set_default or self.default_template is None:
            self.default_template = template.name

    def get_template(self, name: str) -> Optional[BasePromptTemplate]:
        """获取模板"""
        return self.templates.get(name)

    def build(
        self,
        context: str,
        question: str,
        template_name: Optional[str] = None,
        template: Optional[BasePromptTemplate] = None,
        few_shot_examples: Optional[List[FewShotExample]] = None,
        enable_few_shot: bool = True,
        max_context_length: int = 8000,
        custom_components: Optional[Dict[ComponentType, str]] = None,
        **kwargs
    ) -> RenderedPrompt:
        """
        构建Prompt

        Args:
            context: 检索到的上下文
            question: 用户问题
            template_name: 模板名称
            template: 模板实例（优先级更高）
            few_shot_examples: Few-Shot示例列表
            enable_few_shot: 是否启用Few-Shot
            max_context_length: 最大上下文长度
            custom_components: 自定义组件覆盖
            **kwargs: 其他渲染参数

        Returns:
            RenderedPrompt: 渲染后的Prompt
        """
        # 选择模板
        if template is not None:
            selected_template = template
        elif template_name is not None:
            selected_template = self.templates.get(template_name)
            if selected_template is None:
                raise ValueError(f"Template not found: {template_name}")
        elif self.default_template is not None:
            selected_template = self.templates[self.default_template]
        else:
            raise ValueError("No template specified")

        # 截断过长上下文
        if len(context) > max_context_length:
            context = context[:max_context_length] + "...[内容已截断]"

        # 添加自定义组件覆盖
        if custom_components:
            for ctype, content in custom_components.items():
                selected_template.add_component(
                    PromptComponent(
                        name=f"custom_{ctype.value}",
                        content=content,
                        component_type=ctype
                    )
                )

        # 添加Few-Shot示例
        if enable_few_shot and few_shot_examples:
            examples_text = "\n\n".join(
                ex.to_prompt(include_context=False) for ex in few_shot_examples
            )
            few_shot_component = PromptComponent(
                name="dynamic_few_shot",
                content=f"参考示例：\n\n{examples_text}",
                component_type=ComponentType.FEW_SHOT
            )
            selected_template.add_component(few_shot_component)

        # 渲染Prompt
        full_prompt = selected_template.render(context, question, **kwargs)

        # 分离system和user部分（简化处理）
        # 实际应根据消息格式分离
        system_parts = []
        user_parts = []

        for part in full_prompt.split("\n\n"):
            if "用户问题" in part or "问题：" in part:
                user_parts.append(part)
            else:
                system_parts.append(part)

        system_prompt = "\n\n".join(system_parts)
        user_prompt = "\n\n".join(user_parts)

        return RenderedPrompt(
            full_prompt=full_prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            template_used=selected_template.name,
            template_version=selected_template.version,
            context_length=len(context),
            question_length=len(question)
        )

    def build_with_variants(
        self,
        context: str,
        question: str,
        variant_params: List[Dict[str, Any]]
    ) -> Dict[str, RenderedPrompt]:
        """
        构建多个Prompt变体用于A/B测试

        Args:
            context: 上下文
            question: 问题
            variant_params: 变体参数列表

        Returns:
            Dict[变体名称, RenderedPrompt]
        """
        results = {}

        for params in variant_params:
            variant_name = params.pop("name", f"variant_{len(results)}")
            try:
                result = self.build(context, question, **params)
                results[variant_name] = result
            except Exception as e:
                results[variant_name] = f"Error: {str(e)}"

        return results

    def compare_prompts(
        self,
        context: str,
        question: str,
        template_names: List[str]
    ) -> Dict[str, str]:
        """对比多个模板生成的Prompt"""
        results = {}
        for name in template_names:
            prompt = self.build(context, question, template_name=name)
            results[name] = prompt.full_prompt[:500] + "..." if len(prompt.full_prompt) > 500 else prompt.full_prompt
        return results

    def estimate_tokens(self, text: str) -> int:
        """估算token数量（简单估算：中文约2字符/token，英文约4字符/token）"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return chinese_chars // 2 + other_chars // 4
