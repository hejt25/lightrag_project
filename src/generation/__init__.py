"""
Generation Module
=================
生成模块 - 负责基于检索结果生成回答

支持两种Prompt模式：
1. LegacyPromptBuilder - 原有简单模式
2. NewPromptPipeline - 新版专业Prompt模板系统
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """生成结果"""
    answer: str
    context_used: List[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_name: str


class BaseLLM(ABC):
    """LLM基类"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        """生成回答"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", api_base: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> GenerationResult:
        """生成回答"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return GenerationResult(
            answer=response.choices[0].message.content,
            context_used=[],
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            model_name=self.model
        )


class LocalLLM(BaseLLM):
    """本地LLM（OpenAI兼容API）"""

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", api_base: str = "http://localhost:8000/v1"):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.client = openai.OpenAI(base_url=api_base, api_key="not-needed")
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> GenerationResult:
        """生成回答"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return GenerationResult(
            answer=response.choices[0].message.content,
            context_used=[],
            prompt_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
            completion_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else 0,
            total_tokens=response.usage.total_tokens if hasattr(response, 'usage') else 0,
            model_name=self.model
        )


class LegacyPromptBuilder:
    """提示词构建器 - 原有简单模式（向后兼容）"""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个智能助手。请根据提供的上下文信息回答用户的问题。

要求：
1. 仅基于提供的上下文信息进行回答，不要编造信息
2. 如果上下文中没有相关信息，请明确说明
3. 回答要简洁、准确
4. 在回答中标注信息来源

上下文信息：
{context}

用户问题：{question}

请根据以上上下文信息回答问题："""

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """构建完整提示词"""
        context_text = "\n\n".join([f"[来源{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        prompt = self.system_prompt.format(context=context_text, question=question)
        return prompt

    def build_rag_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """构建带元数据的RAG提示词"""
        context_text = "\n\n".join([
            f"[来源{i+1}] (相关度:{ctx.get('score', 0):.3f})\n{ctx.get('content', '')}"
            for i, ctx in enumerate(contexts)
        ])
        prompt = self.system_prompt.format(context=context_text, question=question)
        return prompt


class NewPromptPipeline:
    """
    新版专业Prompt管道

    使用新的Prompt模板系统，支持：
    - 多种预定义模板（通用、保险领域）
    - 模板版本管理
    - A/B测试
    - 动态组件组合
    """

    def __init__(
        self,
        llm: BaseLLM,
        template_name: str = "GeneralRAGPrompt",
        max_tokens: int = 2048,
        temperature: float = 0.7
    ):
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature

        from src.prompt_templates import (
            PromptBuilder,
            GeneralRAGPrompt,
            InsuranceProductPrompt,
            InsuranceClaimPrompt,
            InsuranceComparisonPrompt,
            InsuranceRecommendationPrompt,
            ConversationalPrompt
        )

        self.builder = PromptBuilder()
        self.templates = {
            "GeneralRAGPrompt": GeneralRAGPrompt,
            "InsuranceProductPrompt": InsuranceProductPrompt,
            "InsuranceClaimPrompt": InsuranceClaimPrompt,
            "InsuranceComparisonPrompt": InsuranceComparisonPrompt,
            "InsuranceRecommendationPrompt": InsuranceRecommendationPrompt,
            "ConversationalPrompt": ConversationalPrompt,
        }

        if template_name in self.templates:
            self.builder.register_template(self.templates[template_name](), set_default=True)
            self.current_template_name = template_name
        else:
            self.builder.register_template(GeneralRAGPrompt(), set_default=True)
            self.current_template_name = "GeneralRAGPrompt"

    def generate(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        template_name: Optional[str] = None
    ) -> GenerationResult:
        """基于上下文生成回答"""
        context_text = "\n\n".join([
            f"[来源{i+1}] (相关度:{ctx.get('score', 0):.3f})\n{ctx.get('content', '')}"
            for i, ctx in enumerate(contexts)
        ])

        if template_name and template_name in self.templates:
            self.builder.register_template(self.templates[template_name]())

        rendered = self.builder.build(context_text, question)
        prompt = rendered.full_prompt

        logger.info(f"Generated prompt with {len(prompt)} characters (template: {rendered.template_used})")

        result = self.llm.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        result.context_used = [ctx.get("id", "") for ctx in contexts]
        return result

    def generate_with_history(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        template_name: str = "ConversationalPrompt"
    ) -> GenerationResult:
        """基于历史对话生成回答"""
        if template_name in self.templates:
            self.builder.register_template(self.templates[template_name]())

        context_text = "\n\n".join([
            f"[来源{i+1}] {ctx.get('content', '')}"
            for i, ctx in enumerate(contexts)
        ])

        rendered = self.builder.build(context_text, question, history=history)

        messages = [{"role": "user", "content": rendered.full_prompt}]

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            result = GenerationResult(
                answer=response.choices[0].message.content,
                context_used=[ctx.get("id", "") for ctx in contexts],
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model_name=self.llm.model
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result = GenerationResult(
                answer=f"生成失败: {str(e)}",
                context_used=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model_name=self.llm.model
            )

        return result

    def get_template_info(self) -> Dict[str, Any]:
        """获取当前模板信息"""
        return {
            "current_template_name": self.current_template_name,
            "available_templates": list(self.templates.keys())
        }


class GenerationPipeline(NewPromptPipeline):
    """生成管道 - 兼容旧接口"""

    def __init__(
        self,
        llm: BaseLLM,
        prompt_builder: Optional[LegacyPromptBuilder] = None,
        use_new_pipeline: bool = False,
        template_name: str = "GeneralRAGPrompt",
        max_tokens: int = 2048,
        temperature: float = 0.7
    ):
        if use_new_pipeline:
            super().__init__(llm, template_name, max_tokens, temperature)
            self.legacy_builder = None
        else:
            self.llm = llm
            self.legacy_builder = prompt_builder or LegacyPromptBuilder()
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.use_new_pipeline = False

    def generate(self, question: str, contexts: List[Dict[str, Any]]) -> GenerationResult:
        """基于上下文生成回答"""
        if self.use_new_pipeline:
            return super().generate(question, contexts)

        prompt = self.legacy_builder.build_rag_prompt(question, contexts)
        logger.info(f"Generated prompt with {len(prompt)} characters")

        result = self.llm.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        result.context_used = [ctx.get("id", "") for ctx in contexts]
        return result

    def generate_with_history(self, question: str, contexts: List[Dict[str, Any]], history: List[Dict[str, str]]) -> GenerationResult:
        """基于历史对话生成回答"""
        context_text = "\n\n".join([f"[来源{i+1}] {ctx.get('content', '')}" for i, ctx in enumerate(contexts)])

        messages = []
        messages.append({"role": "system", "content": self.legacy_builder.system_prompt.format(context=context_text, question="")})
        messages.extend(history)
        messages.append({"role": "user", "content": question})

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            result = GenerationResult(
                answer=response.choices[0].message.content,
                context_used=[ctx.get("id", "") for ctx in contexts],
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model_name=self.llm.model
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result = GenerationResult(
                answer=f"生成失败: {str(e)}",
                context_used=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model_name=self.llm.model
            )

        return result
