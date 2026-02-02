"""
Unit Tests for LightRAG
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import LightRAGConfig, Document
from src.ingestion import Chunker, TextLoader
from src.monitoring import Monitor, TraceContext


class TestConfig:
    """配置测试"""

    def test_default_config(self):
        config = LightRAGConfig()
        assert config.project_name == "LightRAG"
        assert config.version == "1.0.0"

    def test_config_to_dict(self):
        config = LightRAGConfig()
        data = config.to_dict()
        assert "project_name" in data
        assert "embedding" in data


class TestIngestion:
    """摄入测试"""

    def test_chunker_basic(self):
        """测试分块器基本功能"""
        chunker = Chunker(chunk_size=50, chunk_overlap=10)

        # Use content with proper sentence-ending punctuation
        content = "这是一个测试句子。这应该是第二个句子。这是第三句。"

        doc = Document(
            id="test",
            content=content * 10,  # Repeat to create longer content
            metadata={"source": "test.txt"}
        )

        chunks = chunker.chunk([doc])
        # Verify we get valid chunks
        assert len(chunks) >= 1
        # Verify chunks have proper structure
        for chunk in chunks:
            assert chunk.id.startswith("test_chunk_")
            assert chunk.chunk_id is not None
            assert len(chunk.content) > 0  # Content should not be empty

    def test_text_loader(self, tmp_path):
        """测试文本加载器"""
        loader = TextLoader()
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World\nTest content")

        docs = loader.load(str(test_file))
        assert len(docs) == 1
        assert "Hello World" in docs[0].content


class TestMonitoring:
    """监控测试"""

    def test_monitor_init(self):
        monitor = Monitor(enable_tracing=True, enable_metrics=True)
        assert monitor.enable_tracing is True

    def test_trace_context(self):
        context = TraceContext()
        assert context.spans == []

    def test_span_tracking(self):
        monitor = Monitor(enable_tracing=True)
        with monitor.span("test_span", {"key": "value"}) as span:
            pass

        trace = monitor.get_trace()
        assert len(trace["spans"]) == 1
        assert trace["spans"][0]["name"] == "test_span"

    def test_metrics_recording(self):
        monitor = Monitor(enable_metrics=True)
        monitor.record_latency("test_op", 0.5)
        metrics = monitor.get_metrics()
        assert len(metrics) == 1


class TestPipeline:
    """管道测试"""

    def test_pipeline_init(self, tmp_path):
        """测试管道初始化"""
        pytest.importorskip("sentence_transformers")

        from src import LightRAGPipeline

        config = LightRAGConfig(data_path=str(tmp_path))
        pipeline = LightRAGPipeline(config)
        assert pipeline is not None

    def test_easy_rag(self):
        """测试简化版接口"""
        pytest.importorskip("sentence_transformers")

        from src import EasyLightRAG

        rag = EasyLightRAG(data_dir="./nonexistent")
        assert rag is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
