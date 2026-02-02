"""
Retrieval Module
=================
检索模块 - 负责从向量存储中检索相关文档
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None


class BaseRetriever(ABC):
    """检索器基类"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """检索相关文档"""
        pass


class VectorRetriever(BaseRetriever):
    """向量检索器"""

    def __init__(self, vector_store, embedder, top_k: int = 5):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """基于向量的相似度检索"""
        k = top_k or self.top_k

        # 向量化查询
        query_vector = self.embedder.embed_query(query)

        # 搜索
        results = self.vector_store.search(query_vector, k)

        # 转换为RetrievalResult
        retrieval_results = []
        for doc in results:
            retrieval_results.append(RetrievalResult(
                document_id=doc.get("id", ""),
                content=doc.get("content", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {}),
                chunk_id=doc.get("chunk_id")
            ))

        logger.info(f"Retrieved {len(retrieval_results)} documents for query: {query[:50]}...")
        return retrieval_results


class BM25Retriever(BaseRetriever):
    """BM25关键词检索器"""

    def __init__(self, documents: List[Dict[str, Any]]):
        try:
            import rank_bm25
        except ImportError:
            logger.error("rank_bm25 not installed")
            raise ImportError("Please install rank-bm25: pip install rank-bm25")

        self.documents = documents
        self.tokenized_docs = [self._tokenize(doc.get("content", "")) for doc in documents]

        # 初始化BM25
        self.bm25 = rank_bm25.BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        return text.lower().split()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """BM25检索"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                results.append(RetrievalResult(
                    document_id=doc.get("id", ""),
                    content=doc.get("content", ""),
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {})
                ))

        return results


class HybridRetriever(BaseRetriever):
    """混合检索器 - 结合向量和关键词检索"""

    def __init__(
        self,
        vector_store,
        embedder,
        documents: List[Dict[str, Any]],
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        top_k: int = 5
    ):
        self.vector_retriever = VectorRetriever(vector_store, embedder, top_k * 2)
        self.bm25_retriever = BM25Retriever(documents)
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """混合检索"""
        k = top_k or self.top_k

        # 并行检索
        vector_results = self.vector_retriever.retrieve(query, k * 2)
        keyword_results = self.bm25_retriever.retrieve(query, k * 2)

        # 合并结果
        combined = {}

        for result in vector_results:
            doc_key = result.document_id
            combined[doc_key] = {
                "document_id": doc_key,
                "content": result.content,
                "metadata": result.metadata,
                "vector_score": result.score,
                "keyword_score": 0.0,
                "final_score": result.score * self.vector_weight
            }

        for result in keyword_results:
            doc_key = result.document_id
            if doc_key in combined:
                combined[doc_key]["keyword_score"] = result.score
                combined[doc_key]["final_score"] = (
                    combined[doc_key]["vector_score"] * self.vector_weight +
                    result.score * self.keyword_weight
                )
            else:
                combined[doc_key] = {
                    "document_id": doc_key,
                    "content": result.content,
                    "metadata": result.metadata,
                    "vector_score": 0.0,
                    "keyword_score": result.score,
                    "final_score": result.score * self.keyword_weight
                }

        # 排序并返回top_k
        sorted_results = sorted(combined.values(), key=lambda x: x["final_score"], reverse=True)

        results = []
        for doc in sorted_results[:k]:
            results.append(RetrievalResult(
                document_id=doc["document_id"],
                content=doc["content"],
                score=doc["final_score"],
                metadata=doc["metadata"]
            ))

        return results


class Reranker:
    """重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise ImportError("Please install sentence-transformers")

        self.model = SentenceTransformer(model_name)

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 3) -> List[RetrievalResult]:
        """重排序"""
        if not candidates:
            return []

        # 准备输入
        pairs = [(query, candidate.content) for candidate in candidates]

        # 计算分数
        scores = self.model.predict(pairs, normalize_probs=True)

        # 按分数排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k
        results = []
        for idx, score in indexed_scores[:top_k]:
            candidate = candidates[idx]
            results.append(RetrievalResult(
                document_id=candidate.document_id,
                content=candidate.content,
                score=float(score),
                metadata=candidate.metadata,
                chunk_id=candidate.chunk_id
            ))

        return results


class RetrievalPipeline:
    """检索管道"""

    def __init__(
        self,
        vector_store,
        embedder,
        documents: List[Dict[str, Any]],
        use_hybrid: bool = False,
        use_reranker: bool = False,
        top_k: int = 5
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

        if use_reranker:
            self.reranker = Reranker()
        else:
            self.reranker = None

        if use_hybrid:
            self.retriever = HybridRetriever(
                vector_store, embedder, documents,
                vector_weight=0.6, keyword_weight=0.4,
                top_k=top_k
            )
        else:
            self.retriever = VectorRetriever(vector_store, embedder, top_k * 2)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """执行检索"""
        results = self.retriever.retrieve(query, top_k or self.top_k)

        # 重排序
        if self.reranker and len(results) > 3:
            results = self.reranker.rerank(query, results, top_k or self.top_k)

        return results
