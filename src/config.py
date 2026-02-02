"""
LightRAG Configuration
=======================
配置文件管理
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class EmbeddingConfig:
    """Embedding配置"""
    model_name: str = "BAAI/bge-small-zh"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    storage_type: str = "faiss"  # faiss, milvus, qdrant
    index_path: str = "./data/vector_store"
    dimension: int = 512
    metric: str = "cosine"


@dataclass
class LLMConfig:
    """大语言模型配置"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    api_base: str = "http://localhost:8000/v1"
    max_tokens: int = 2048
    temperature: float = 0.7


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_tracing: bool = True
    enable_metrics: bool = True
    log_level: str = "INFO"
    trace_export_endpoint: Optional[str] = None


@dataclass
class LightRAGConfig:
    """主配置类"""
    # 基础配置
    project_name: str = "LightRAG"
    version: str = "1.0.0"
    data_path: str = "./data/raw"
    output_path: str = "./data/output"

    # 子配置
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # RAG特有配置
    retrieval_top_k: int = 5
    context_window: int = 8192
    chunk_size: int = 512
    chunk_overlap: int = 50

    @classmethod
    def from_json(cls, config_path: str) -> "LightRAGConfig":
        """从JSON文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict:
        """导出为字典"""
        return {
            "project_name": self.project_name,
            "version": self.version,
            "data_path": self.data_path,
            "output_path": self.output_path,
            "retrieval_top_k": self.retrieval_top_k,
            "context_window": self.context_window,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding": {
                "model_name": self.embedding.model_name,
                "device": self.embedding.device,
                "batch_size": self.embedding.batch_size,
                "max_length": self.embedding.max_length
            },
            "vector_store": {
                "storage_type": self.vector_store.storage_type,
                "index_path": self.vector_store.index_path,
                "dimension": self.vector_store.dimension,
                "metric": self.vector_store.metric
            },
            "llm": {
                "model_name": self.llm.model_name,
                "api_base": self.llm.api_base,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature
            },
            "monitoring": {
                "enable_tracing": self.monitoring.enable_tracing,
                "enable_metrics": self.monitoring.enable_metrics,
                "log_level": self.monitoring.log_level
            }
        }

    def save(self, config_path: str):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# 默认配置实例
default_config = LightRAGConfig()
