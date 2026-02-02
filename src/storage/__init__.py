"""
Vector Storage Module
=====================
向量存储模块 - 负责向量数据的存储和索引
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """向量存储基类"""

    @abstractmethod
    def add(self, documents: List[Dict[str, Any]]) -> bool:
        """添加文档向量"""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """相似度搜索"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """删除文档"""
        pass

    @abstractmethod
    def save(self, path: str) -> bool:
        """保存索引"""
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        """加载索引"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS向量存储"""

    def __init__(self, dimension: int = 512, metric: str = "cosine", index_path: str = "./data/vector_store"):
        try:
            import faiss
        except ImportError:
            logger.error("faiss not installed")
            raise ImportError("Please install faiss: pip install faiss-cpu")

        self.dimension = dimension
        self.metric = metric
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # 初始化FAISS索引
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)
            self.normalize = True
        elif metric == "euclidean":
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = False
        else:
            self.index = faiss.IndexFlatIP(dimension)
            self.normalize = True

        # 文档存储
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}

    def add(self, documents: List[Dict[str, Any]]) -> bool:
        """添加文档向量"""
        if not documents:
            return True

        vectors = []
        ids = []

        for doc in documents:
            embedding = np.array(doc["embedding"], dtype=np.float32)

            if self.normalize:
                faiss.normalize_L2(embedding.reshape(1, -1))
                embedding = embedding.reshape(1, -1)

            vectors.append(embedding.flatten())
            ids.append(doc["id"])

        vectors = np.vstack(vectors)

        # 添加到FAISS索引
        self.index.add(vectors)

        # 更新映射
        start_idx = self.index.ntotal - len(vectors)
        for i, doc_id in enumerate(ids):
            self.id_to_index[doc_id] = start_idx + i
            self.documents[doc_id] = doc

        logger.info(f"Added {len(documents)} documents, total: {self.index.ntotal}")
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """相似度搜索"""
        if self.index.ntotal == 0:
            return []

        query = query_vector.astype(np.float32).reshape(1, -1)

        if self.normalize:
            faiss.normalize_L2(query)

        # 搜索
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            # 找到对应的文档ID
            doc_id = None
            for did, didx in self.id_to_index.items():
                if didx == idx:
                    doc_id = did
                    break

            if doc_id and doc_id in self.documents:
                doc = self.documents[doc_id].copy()
                doc["score"] = float(score)
                results.append(doc)

        return results

    def delete(self, ids: List[str]) -> bool:
        """删除文档（FAISS不支持真正删除，标记删除）"""
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                if doc_id in self.id_to_index:
                    del self.id_to_index[doc_id]
        return True

    def save(self, path: str) -> bool:
        """保存索引和文档"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(self.index, str(path / "index.faiss"))

        # 保存文档元数据
        with open(path / "documents.json", 'w', encoding='utf-8') as f:
            # 移除embedding以减小文件大小
            docs_to_save = {
                k: {kk: vvk for kk, vvk in v.items() if kk != "embedding"}
                for k, v in self.documents.items()
            }
            json.dump(docs_to_save, f, ensure_ascii=False, indent=2)

        # 保存ID映射
        with open(path / "id_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(self.id_to_index, f, ensure_ascii=False)

        logger.info(f"Saved index to {path}")
        return True

    def load(self, path: str) -> bool:
        """加载索引和文档"""
        path = Path(path)

        if not path.exists():
            logger.warning(f"Index path {path} not found")
            return False

        # 加载FAISS索引
        if (path / "index.faiss").exists():
            self.index = faiss.read_index(str(path / "index.faiss"))

        # 加载文档
        if (path / "documents.json").exists():
            with open(path / "documents.json", 'r', encoding='utf-8') as f:
                self.documents = json.load(f)

        # 加载ID映射
        if (path / "id_mapping.json").exists():
            with open(path / "id_mapping.json", 'r', encoding='utf-8') as f:
                self.id_to_index = json.load(f)

        logger.info(f"Loaded index from {path}, total documents: {len(self.documents)}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "dimension": self.dimension,
            "metric": self.metric,
            "documents": list(self.documents.keys())[:10]  # 只返回前10个ID
        }


class InMemoryVectorStore(BaseVectorStore):
    """内存向量存储（简单实现）"""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.documents: Dict[str, Dict[str, Any]] = {}

    def add(self, documents: List[Dict[str, Any]]) -> bool:
        """添加文档向量"""
        for doc in documents:
            self.vectors.append(np.array(doc["embedding"], dtype=np.float32))
            self.documents[doc["id"]] = doc
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """简单的余弦相似度搜索"""
        if not self.vectors:
            return []

        # 计算相似度
        similarities = []
        for i, vec in enumerate(self.vectors):
            score = np.dot(query_vector, vec) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-8
            )
            similarities.append((score, i))

        # 排序并返回top_k
        similarities.sort(reverse=True, key=lambda x: x[0])

        results = []
        for score, idx in similarities[:top_k]:
            doc = list(self.documents.values())[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

    def delete(self, ids: List[str]) -> bool:
        """删除文档"""
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
        return True

    def save(self, path: str) -> bool:
        """保存到文件"""
        import pickle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "vector_store.pkl", 'wb') as f:
            pickle.dump({
                "vectors": self.vectors,
                "documents": self.documents,
                "dimension": self.dimension
            }, f)
        return True

    def load(self, path: str) -> bool:
        """从文件加载"""
        import pickle
        path = Path(path)

        if not path.exists():
            return False

        with open(path / "vector_store.pkl", 'rb') as f:
            data = pickle.load(f)
            self.vectors = data["vectors"]
            self.documents = data["documents"]
            self.dimension = data["dimension"]
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": len(self.documents),
            "vector_count": len(self.vectors),
            "dimension": self.dimension
        }


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self, store: BaseVectorStore):
        self.store = store

    def create_index(self, documents: List[Dict[str, Any]], dimension: int) -> bool:
        """创建索引"""
        self.store.add(documents)
        return True

    def search(self, query: str, embed_func, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索"""
        query_vector = embed_func(query)
        return self.store.search(query_vector, top_k)

    def save_index(self, path: str) -> bool:
        """保存索引"""
        return self.store.save(path)

    def load_index(self, path: str) -> bool:
        """加载索引"""
        return self.store.load(path)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return self.store.get_stats()
