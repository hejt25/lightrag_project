"""
Generation Module
=================
生成模块 - 负责基于检索结果生成回答
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


class PromptBuilder:
    """提示词构建器"""

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


class GenerationPipeline:
    """生成管道"""

    def __init__(
        self,
        llm: BaseLLM,
        prompt_builder: Optional[PromptBuilder] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, question: str, contexts: List[Dict[str, Any]]) -> GenerationResult:
        """基于上下文生成回答"""
        # 构建提示词
        prompt = self.prompt_builder.build_rag_prompt(question, contexts)

        logger.info(f"Generated prompt with {len(prompt)} characters")

        # 调用LLM
        result = self.llm.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        result.context_used = [ctx.get("id", "") for ctx in contexts]

        return result

    def generate_with_history(self, question: str, contexts: List[Dict[str, Any]], history: List[Dict[str, str]]) -> GenerationResult:
        """基于历史对话生成回答"""
        # 构建带历史的提示词
        context_text = "\n\n".join([f"[来源{i+1}] {ctx.get('content', '')}" for i, ctx in enumerate(contexts)])

        messages = []

        # 添加系统提示词
        messages.append({"role": "system", "content": self.prompt_builder.system_prompt.format(context=context_text, question="")})

        # 添加历史对话
        messages.extend(history)

        # 添加当前问题
        messages.append({"role": "user", "content": question})

        # 调用LLM
        try:
            import openai
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
