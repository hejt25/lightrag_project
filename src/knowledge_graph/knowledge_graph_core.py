"""
Knowledge Graph Core
====================
知识图谱核心模块 - 实体、关系、图谱定义
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体"""
    id: str
    name: str
    entity_type: str  # PERSON, ORG, PRODUCT, etc.
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    frequency: int = 1
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type,
            "description": self.description,
            "attributes": self.attributes,
            "frequency": self.frequency,
            "sources": self.sources
        }


@dataclass
class Relation:
    """关系"""
    source_id: str
    target_id: str
    relation_type: str  # HAS_ATTRIBUTE, RELATED_TO, IS_A, etc.
    weight: float = 1.0
    description: str = ""
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relation_type,
            "weight": self.weight,
            "description": self.description,
            "sources": self.sources
        }


@dataclass
class Triplet:
    """三元组 (头实体, 关系, 尾实体)"""
    head: str
    relation: str
    tail: str
    confidence: float = 1.0
    source: str = ""

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


class KnowledgeGraph:
    """知识图谱"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # name -> entity_ids
        self.type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        self.adjacency: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

    def add_entity(self, entity: Entity) -> str:
        """添加实体"""
        if entity.id not in self.entities:
            self.entities[entity.id] = entity
            self.entity_index[entity.name.lower()].add(entity.id)
            self.type_index[entity.entity_type].add(entity.id)
        else:
            self.entities[entity.id].frequency += 1
        return entity.id

    def add_relation(self, relation: Relation) -> None:
        """添加关系"""
        self.relations.append(relation)
        self.adjacency[relation.source_id].append(
            (relation.target_id, relation.relation_type, relation.weight)
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(entity_id)

    def find_entity(self, name: str) -> Optional[Entity]:
        """按名称查找实体"""
        name_lower = name.lower()
        if name_lower in self.entity_index:
            entity_id = list(self.entity_index[name_lower])[0]
            return self.entities.get(entity_id)
        return None

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None
    ) -> List[Tuple[Entity, str, float]]:
        """获取邻居节点"""
        neighbors = []
        for neighbor_id, rel_type, weight in self.adjacency.get(entity_id, []):
            if relation_type is None or rel_type == relation_type:
                entity = self.get_entity(neighbor_id)
                if entity:
                    neighbors.append((entity, rel_type, weight))
        return neighbors

    def get_connected_entities(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Tuple[Entity, int, float]]:
        """获取相连实体（带深度）"""
        result = {}
        self._bfs_collect(entity_id, max_depth, 0, set(), result)
        return result

    def _bfs_collect(
        self,
        current_id: str,
        max_depth: int,
        current_depth: int,
        visited: Set[str],
        result: Dict[str, Tuple[Entity, int, float]]
    ):
        """BFS收集相连实体"""
        if current_depth > max_depth or current_id in visited:
            return
        visited.add(current_id)
        entity = self.get_entity(current_id)
        if entity:
            min_dist = result.get(current_id, (entity, max_depth + 1, 0))[1]
            if current_depth < min_dist:
                result[current_id] = (entity, current_depth, 0)
        for neighbor_id, rel_type, weight in self.adjacency.get(current_id, []):
            if neighbor_id not in visited:
                self._bfs_collect(neighbor_id, max_depth, current_depth + 1, visited, result)

    def search_by_type(self, entity_type: str) -> List[Entity]:
        """按类型搜索"""
        entity_ids = self.type_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def search_by_name_contains(self, keyword: str) -> List[Entity]:
        """按名称模糊搜索"""
        keyword_lower = keyword.lower()
        results = []
        for name_lower, entity_ids in self.entity_index.items():
            if keyword_lower in name_lower:
                for eid in entity_ids:
                    if eid in self.entities:
                        results.append(self.entities[eid])
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        type_counts = {}
        for etype, eids in self.type_index.items():
            type_counts[etype] = len(eids)
        return {
            "name": self.name,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "type_distribution": type_counts
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": [r.to_dict() for r in self.relations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        graph = cls(name=data.get("name", "default"))
        for entity_data in data.get("entities", {}).values():
            graph.add_entity(Entity(**entity_data))
        for rel_data in data.get("relations", []):
            graph.add_relation(Relation(**rel_data))
        return graph


class EntityExtractor:
    """实体抽取器"""

    ENTITY_TYPES = {
        "PRODUCT": ["重疾险", "医疗险", "意外险", "寿险", "年金险", "万能险", "车险", "家财险", "旅游险", "防癌险"],
        "COMPANY": ["平安", "中国人寿", "太平洋", "新华保险", "泰康", "人保", "大地保险", "阳光保险"],
        "CONCEPT": ["保额", "保费", "保障期限", "缴费期限", "等待期", "免赔额", "赔付比例", "现金价值"],
        "ACTION": ["理赔", "投保", "核保", "续保", "退保", "索赔"],
    }

    def __init__(self):
        self.entity_patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[str, List[re.Pattern]]:
        patterns = {}
        for etype, keywords in self.ENTITY_TYPES.items():
            pattern = r"(" + "|".join(re.escape(k) for k in keywords) + ")"
            patterns[etype] = [re.compile(pattern)]
        return patterns

    def extract(self, text: str, source: str = "") -> List[Entity]:
        """从文本中抽取实体"""
        entities = []
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = Entity(
                        id=self._generate_id(match.group()),
                        name=match.group(),
                        entity_type=entity_type,
                        description=f"Extracted from {source}" if source else "",
                        sources=[source] if source else []
                    )
                    entities.append(entity)
        return entities

    def _generate_id(self, name: str) -> str:
        import hashlib
        content = f"{name}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class RelationExtractor:
    """关系抽取器"""

    RELATION_TYPES = {
        "PRODUCED_BY": r"(平安|中国人寿|太平洋|新华|泰康).*(重疾险|医疗险|意外险|寿险)",
        "HAS_ATTRIBUTE": r"(具有|含有|包含|保障).*(重疾|轻症|身故|全残)",
        "IS_A": r"(是|属于).*(重疾险|医疗险|意外险|保险产品)",
        "RELATED_TO": r"(和|与|及).*(相关|关联)",
        "COVERS": r"(保障|覆盖|包括).*(疾病|意外|医疗)",
    }

    def __init__(self):
        self.relation_patterns = {k: re.compile(v) for k, v in self.RELATION_TYPES.items()}

    def extract(self, text: str, entities: List[Entity], source: str = "") -> List[Relation]:
        relations = []
        for rel_type, pattern in self.relation_patterns.items():
            if pattern.search(text):
                if entities:
                    relation = Relation(
                        source_id=entities[0].id,
                        target_id="",
                        relation_type=rel_type,
                        weight=0.8,
                        description=f"Extracted from text",
                        sources=[source] if source else []
                    )
                    relations.append(relation)
        return relations


class GraphBuilder:
    """图谱构建器"""

    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.graph = KnowledgeGraph()

    def build_from_document(self, doc_id: str, content: str, extract_relations: bool = True) -> KnowledgeGraph:
        entities = self.entity_extractor.extract(content, source=doc_id)
        unique_entities = {}
        for entity in entities:
            key = entity.name.lower()
            if key not in unique_entities:
                unique_entities[key] = entity
        for entity in unique_entities.values():
            self.graph.add_entity(entity)
        if extract_relations:
            relations = self.relation_extractor.extract(content, list(unique_entities.values()), source=doc_id)
            for relation in relations:
                self.graph.add_relation(relation)
        logger.info(f"Built graph from {doc_id}: {len(unique_entities)} entities, {len(relations) if extract_relations else 0} relations")
        return self.graph

    def build_from_documents(self, documents: List[Dict[str, str]]) -> KnowledgeGraph:
        for doc in documents:
            self.build_from_document(doc.get("id", f"doc_{len(self.graph.entities)}"), doc.get("content", ""))
        return self.graph

    def get_graph(self) -> KnowledgeGraph:
        return self.graph

    def merge_graph(self, other_graph: KnowledgeGraph) -> None:
        for entity in other_graph.entities.values():
            if entity.id not in self.graph.entities:
                self.graph.add_entity(entity)
        for relation in other_graph.relations:
            self.graph.add_relation(relation)
