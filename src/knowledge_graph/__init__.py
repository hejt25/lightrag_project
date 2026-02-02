"""
Knowledge Graph Module
======================
知识图谱模块 - 支持实体识别、关系抽取和图谱检索
"""

from .knowledge_graph_core import (
    Entity,
    Relation,
    Triplet,
    KnowledgeGraph,
    EntityExtractor,
    RelationExtractor,
    GraphBuilder
)
from .graph_retriever import (
    GraphRetrievalResult,
    GraphRetriever,
    HybridGraphRetriever,
    GraphEnhancedRAG
)

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "Entity",
    "Relation",
    "Triplet",
    "KnowledgeGraph",
    # Extractors
    "EntityExtractor",
    "RelationExtractor",
    "GraphBuilder",
    # Retriever
    "GraphRetrievalResult",
    "GraphRetriever",
    "HybridGraphRetriever",
    "GraphEnhancedRAG"
]
