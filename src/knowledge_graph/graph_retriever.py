"""
Graph Retriever
================
基于知识图谱的检索器

功能：
1. 实体检索 - 根据查询中的实体名查找图谱
2. 关系检索 - 查找与实体相关的邻居和路径
3. 语义检索 - 结合向量检索和图谱检索
4. 多跳推理 - 支持多跳关系查询
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

from .knowledge_graph_core import (
    KnowledgeGraph,
    Entity,
    Relation,
    EntityExtractor,
    RelationExtractor,
    GraphBuilder
)

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrievalResult:
    """图谱检索结果"""
    entity_id: str
    entity_name: str
    entity_type: str
    description: str
    score: float
    source: str = ""
    neighbors: List[Dict[str, Any]] = field(default_factory=list)
    path: List[str] = field(default_factory=list)


class GraphRetriever:
    """图谱检索器"""

    def __init__(self, graph: Optional[KnowledgeGraph] = None):
        self.graph = graph
        self.entity_extractor = EntityExtractor()

    def set_graph(self, graph: KnowledgeGraph) -> None:
        """设置知识图谱"""
        self.graph = graph

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_neighbors: bool = True,
        max_hops: int = 2
    ) -> List[GraphRetrievalResult]:
        """
        基于图谱检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            include_neighbors: 是否包含邻居节点
            max_hops: 最大跳数

        Returns:
            List[GraphRetrievalResult]: 检索结果
        """
        if self.graph is None:
            logger.warning("Graph not set, returning empty results")
            return []

        # 1. 从查询中抽取实体
        query_entities = self.entity_extractor.extract(query)

        if not query_entities:
            # 无实体时，进行模糊搜索
            return self._search_by_keyword(query, top_k, include_neighbors)

        results = []
        seen_entities = set()

        # 2. 对每个抽取的实体进行检索
        for query_entity in query_entities:
            # 精确匹配
            entity = self.graph.find_entity(query_entity.name)

            # 模糊匹配
            if not entity:
                candidates = self.graph.search_by_name_contains(query_entity.name)
                if candidates:
                    entity = candidates[0]

            if entity and entity.id not in seen_entities:
                seen_entities.add(entity.id)

                # 获取邻居
                neighbors = []
                if include_neighbors:
                    neighbors = self._get_entity_neighbors(entity.id, max_hops)

                result = GraphRetrievalResult(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    score=entity.frequency / 10.0,  # 基于频率的分数
                    source="|".join(entity.sources) if entity.sources else "",
                    neighbors=neighbors
                )
                results.append(result)

        # 3. 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def _search_by_keyword(
        self,
        keyword: str,
        top_k: int,
        include_neighbors: bool
    ) -> List[GraphRetrievalResult]:
        """关键词搜索"""
        entities = self.graph.search_by_name_contains(keyword)

        results = []
        for entity in entities[:top_k]:
            neighbors = []
            if include_neighbors:
                neighbors = self._get_entity_neighbors(entity.id, 1)

            results.append(GraphRetrievalResult(
                entity_id=entity.id,
                entity_name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                score=entity.frequency / 10.0,
                neighbors=neighbors
            ))

        return results

    def _get_entity_neighbors(
        self,
        entity_id: str,
        max_hops: int
    ) -> List[Dict[str, Any]]:
        """获取实体邻居"""
        neighbors = []
        connected = self.graph.get_connected_entities(entity_id, max_hops)

        for eid, (entity, depth, _) in connected.items():
            if eid == entity_id:
                continue

            neighbors.append({
                "entity_id": eid,
                "name": entity.name,
                "type": entity.entity_type,
                "distance": depth
            })

        return neighbors

    def retrieve_with_reasoning(
        self,
        query: str,
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """
        带推理的检索

        Returns:
            Dict包含：
            - entities: 检索到的实体
            - relations: 检索到的关系
            - reasoning_path: 推理路径
            - contextual_info: 上下文信息
        """
        results = self.retrieve(query, top_k=10, include_neighbors=True, max_hops=max_hops)

        # 构建推理信息
        reasoning_path = []
        contextual_info = []

        for result in results:
            reasoning_path.append({
                "entity": result.entity_name,
                "type": result.entity_type,
                "neighbors": len(result.neighbors)
            })

            for neighbor in result.neighbors[:3]:  # 只取前3个邻居
                contextual_info.append({
                    "entity": result.entity_name,
                    "related": neighbor["name"],
                    "relation": f"distance_{neighbor['distance']}"
                })

        return {
            "entities": [
                {
                    "id": r.entity_id,
                    "name": r.entity_name,
                    "type": r.entity_type,
                    "score": r.score
                }
                for r in results
            ],
            "reasoning_path": reasoning_path,
            "contextual_info": contextual_info,
            "total_entities": len(results)
        }


class HybridGraphRetriever:
    """混合图谱检索器 - 结合向量检索和图谱检索"""

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        vector_store=None,
        embedder=None
    ):
        self.graph_retriever = GraphRetriever(graph)
        self.vector_store = vector_store
        self.embedder = embedder
        self.graph_weight: float = 0.3  # 图谱结果权重
        self.vector_weight: float = 0.7  # 向量结果权重

    def set_graph(self, graph: KnowledgeGraph) -> None:
        """设置知识图谱"""
        self.graph_retriever.set_graph(graph)

    def set_vector_store(self, vector_store, embedder) -> None:
        """设置向量存储"""
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid"  # "graph", "vector", "hybrid"
    ) -> Dict[str, Any]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            mode: 检索模式
                - "graph": 仅图谱检索
                - "vector": 仅向量检索
                - "hybrid": 混合检索

        Returns:
            Dict包含：
            - graph_results: 图谱检索结果
            - vector_results: 向量检索结果
            - fused_results: 融合后的结果
            - reasoning_info: 推理信息
        """
        result = {
            "mode": mode,
            "query": query,
            "graph_results": [],
            "vector_results": [],
            "fused_results": [],
            "reasoning_info": {}
        }

        # 1. 图谱检索
        if mode in ["graph", "hybrid"]:
            graph_results = self.graph_retriever.retrieve(
                query,
                top_k=top_k,
                include_neighbors=True
            )
            result["graph_results"] = [
                {
                    "id": r.entity_id,
                    "name": r.entity_name,
                    "type": r.entity_type,
                    "description": r.description,
                    "score": r.score,
                    "neighbors": r.neighbors
                }
                for r in graph_results
            ]
            result["reasoning_info"] = self.graph_retriever.retrieve_with_reasoning(query)

        # 2. 向量检索
        if mode in ["vector", "hybrid"] and self.vector_store and self.embedder:
            query_vector = self.embedder.embed_query(query)
            vector_results = self.vector_store.search(query_vector, top_k)

            result["vector_results"] = [
                {
                    "id": r.get("id", ""),
                    "content": r.get("content", "")[:200],
                    "score": r.get("score", 0)
                }
                for r in vector_results
            ]

        # 3. 结果融合
        if mode == "hybrid" and result["graph_results"] and result["vector_results"]:
            fused = self._fuse_results(
                result["graph_results"],
                result["vector_results"],
                top_k
            )
            result["fused_results"] = fused
        elif mode == "graph":
            result["fused_results"] = result["graph_results"]
        elif mode == "vector":
            result["fused_results"] = result["vector_results"]

        return result

    def _fuse_results(
        self,
        graph_results: List[Dict],
        vector_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """融合结果"""
        # 归一化分数
        max_g_score = max(r["score"] for r in graph_results) if graph_results else 1
        max_v_score = max(r["score"] for r in vector_results) if vector_results else 1

        # 合并
        fused_map = {}

        for r in graph_results:
            key = r["id"] if r["id"] else r["name"]
            fused_map[key] = {
                **r,
                "final_score": (r["score"] / max_g_score) * self.graph_weight,
                "sources": ["graph"]
            }

        for r in vector_results:
            key = r["id"]
            if key in fused_map:
                fused_map[key]["final_score"] += (r["score"] / max_v_score) * self.vector_weight
                fused_map[key]["sources"].append("vector")
            else:
                fused_map[key] = {
                    **r,
                    "final_score": (r["score"] / max_v_score) * self.vector_weight,
                    "sources": ["vector"]
                }

        # 按融合分数排序
        fused = list(fused_map.values())
        fused.sort(key=lambda x: x["final_score"], reverse=True)

        return fused[:top_k]


class GraphEnhancedRAG:
    """图谱增强的RAG"""

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        vector_store=None,
        embedder=None
    ):
        self.hybrid_retriever = HybridGraphRetriever(graph, vector_store, embedder)

        # 初始化图谱构建器
        self.graph_builder = GraphBuilder()

        # 如果没有图谱，创建一个
        if graph is None:
            self.graph = KnowledgeGraph("default")
            self.hybrid_retriever.set_graph(self.graph)
        else:
            self.graph = graph

    def build_graph(self, documents: List[Dict[str, str]]) -> KnowledgeGraph:
        """从文档构建知识图谱"""
        self.graph = self.graph_builder.build_from_documents(documents)
        self.hybrid_retriever.set_graph(self.graph)
        return self.graph

    def add_document_to_graph(self, doc_id: str, content: str) -> None:
        """添加文档到图谱"""
        self.graph_builder.build_from_document(doc_id, content)
        self.hybrid_retriever.set_graph(self.graph_builder.get_graph())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """检索"""
        return self.hybrid_retriever.retrieve(query, top_k, mode)

    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计"""
        return self.graph.get_stats()

    def export_graph(self, path: str) -> None:
        """导出图谱"""
        import json
        graph_dict = self.graph.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, ensure_ascii=False, indent=2)

    def load_graph(self, path: str) -> None:
        """加载图谱"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            graph_dict = json.load(f)
        self.graph = KnowledgeGraph.from_dict(graph_dict)
        self.hybrid_retriever.set_graph(self.graph)
