"""
Main Pipeline
=============
LightRAG主流程管道
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import time

from .config import LightRAGConfig
from .ingestion import DataIngestionPipeline, Document
from .embedding import EmbeddingPipeline, SentenceTransformerEmbedder
from .storage import VectorStoreManager, FAISSVectorStore
from .retrieval import RetrievalPipeline
from .generation import GenerationPipeline, LocalLLM, PromptBuilder
from .monitoring import Monitor, PerformanceMonitor

logger = logging.getLogger(__name__)


class LightRAGPipeline:
    """LightRAG主流程管道"""

    def __init__(self, config: Optional[LightRAGConfig] = None):
        self.config = config or LightRAGConfig()
        self.monitor = Monitor(
            enable_tracing=self.config.monitoring.enable_tracing,
            enable_metrics=self.config.monitoring.enable_metrics,
            log_dir="./logs"
        )
        self.perf_monitor = PerformanceMonitor(self.monitor)

        # 初始化组件
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        # 1. 数据摄入
        self.ingestion_pipeline = DataIngestionPipeline(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # 2. 向量化
        embedder = SentenceTransformerEmbedder(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device
        )
        self.embedding_pipeline = EmbeddingPipeline(
            embedder=embedder,
            batch_size=self.config.embedding.batch_size
        )

        # 3. 向量存储
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_pipeline.get_dimension(),
            metric=self.config.vector_store.metric,
            index_path=self.config.vector_store.index_path
        )
        self.vector_store_manager = VectorStoreManager(self.vector_store)

        # 4. 检索
        self.retrieval_pipeline = RetrievalPipeline(
            vector_store=self.vector_store,
            embedder=embedder,
            documents=[],  # 将在索引时更新
            use_hybrid=False,
            top_k=self.config.retrieval_top_k
        )

        # 5. 生成
        llm = LocalLLM(
            model=self.config.llm.model_name,
            api_base=self.config.llm.api_base
        )
        self.generation_pipeline = GenerationPipeline(
            llm=llm,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature
        )

    def index(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """构建索引"""
        with self.perf_monitor.measure("indexing"):
            with self.monitor.span("ingestion"):
                data_path = data_dir or self.config.data_path
                documents = self.ingestion_pipeline.process(data_path)

                self.monitor.record_data_ingestion(
                    files_processed=len(set(d.metadata.get("source", "") for d in documents)),
                    chunks_created=len(documents)
                )

            with self.monitor.span("embedding"):
                start_time = time.time()
                documents = self.embedding_pipeline.embed_documents([
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ])
                embedding_time = time.time() - start_time

                self.monitor.record_embedding(
                    documents_count=len(documents),
                    latency=embedding_time
                )

            with self.monitor.span("storage"):
                self.vector_store.add(documents)
                self.vector_store_manager.save_index(self.config.vector_store.index_path)

            stats = self.vector_store_manager.get_stats()
            logger.info(f"Indexing completed: {stats}")

        return stats

    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """直接索引文档"""
        with self.perf_monitor.measure("indexing_documents"):
            with self.monitor.span("document_ingestion"):
                # 分块
                raw_docs = [Document(
                    id=doc.get("id", f"doc_{i}"),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                ) for i, doc in enumerate(documents)]
                chunked_docs = self.ingestion_pipeline.process_documents(raw_docs)

            with self.monitor.span("embedding"):
                start_time = time.time()
                embedded_docs = self.embedding_pipeline.embed_documents([
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                    for doc in chunked_docs
                ])
                embedding_time = time.time() - start_time

                self.monitor.record_embedding(len(embedded_docs), embedding_time)

            with self.monitor.span("storage"):
                self.vector_store.add(embedded_docs)

            return self.vector_store_manager.get_stats()

    def load_index(self, index_path: Optional[str] = None):
        """加载已有索引"""
        path = index_path or self.config.vector_store.index_path
        return self.vector_store_manager.load_index(path)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_contexts: bool = False
    ) -> Dict[str, Any]:
        """查询"""
        k = top_k or self.config.retrieval_top_k

        with self.perf_monitor.measure("query"):
            # 1. 检索
            with self.monitor.span("retrieval"):
                start_time = time.time()
                results = self.retrieval_pipeline.retrieve(question, k)
                retrieval_time = time.time() - start_time

                self.monitor.record_retrieval(
                    query=question,
                    results_count=len(results),
                    latency=retrieval_time
                )

            # 2. 生成
            with self.monitor.span("generation"):
                start_time = time.time()
                contexts = [
                    {
                        "id": r.document_id,
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata
                    }
                    for r in results
                ]

                generation_result = self.generation_pipeline.generate(question, contexts)
                generation_time = time.time() - start_time

                self.monitor.record_generation(generation_time, generation_result.total_tokens)
                self.monitor.record_tokens(
                    generation_result.prompt_tokens,
                    generation_result.completion_tokens
                )

            response = {
                "answer": generation_result.answer,
                "retrieved_documents": len(results),
                "contexts": contexts if return_contexts else None,
                "metrics": {
                    "retrieval_latency_ms": retrieval_time * 1000,
                    "generation_latency_ms": generation_time * 1000,
                    "total_latency_ms": (retrieval_time + generation_time) * 1000,
                    "prompt_tokens": generation_result.prompt_tokens,
                    "completion_tokens": generation_result.completion_tokens,
                    "total_tokens": generation_result.total_tokens
                }
            }

        return response

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            "vector_store": self.vector_store_manager.get_stats(),
            "monitoring": self.monitor.get_stats()
        }

    def export_monitoring_data(self, output_dir: str = "./monitoring"):
        """导出监控数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.monitor.export_trace(str(output_path / "trace.json"))
        self.monitor.export_metrics(str(output_path / "metrics.json"))

        stats = self.get_stats()
        with open(output_path / "stats.json", 'w', encoding='utf-8') as f:
            import json
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return output_dir


class EasyLightRAG:
    """简化版LightRAG - 快速使用"""

    def __init__(
        self,
        data_dir: str = "./data",
        model_name: str = "BAAI/bge-small-zh",
        llm_api_base: str = "http://localhost:8000/v1"
    ):
        config = LightRAGConfig(
            data_path=data_dir,
            embedding={"model_name": model_name},
            llm={"api_base": llm_api_base}
        )
        self.pipeline = LightRAGPipeline(config)

    def build_index(self) -> Dict[str, Any]:
        """构建索引"""
        return self.pipeline.index()

    def query(self, question: str) -> str:
        """查询"""
        result = self.pipeline.query(question)
        return result["answer"]


# 使用示例
if __name__ == "__main__":
    # 初始化
    rag = LightRAGPipeline()

    # 构建索引
    print("Building index...")
    rag.index()

    # 查询
    print("\nQuerying...")
    result = rag.query("什么是人工智能？")
    print(f"Answer: {result['answer']}")
    print(f"Metrics: {result['metrics']}")

    # 导出监控数据
    rag.export_monitoring_data()
