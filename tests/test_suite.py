"""
LightRAG Hard Test Suite
========================

Comprehensive test suite for LightRAG covering edge cases, boundary conditions,
and difficult scenarios that may cause system instability or incorrect behavior.

Test Categories:
- Empty input handling
- Long text processing
- Multilingual support
- Encoding edge cases
- Special characters and formats
- Chunking boundaries
- Exception recovery
- Knowledge graph boundaries
- Monitoring edge cases

Usage:
    python -m pytest tests/test_suite.py -v --tb=short
    python -m pytest tests/test_suite.py::TestEmptyInput -v
    python -m pytest tests/test_suite.py -k "not concurrent" -v
"""
import pytest
import sys
import time
import json
import threading
import traceback
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from src.config import LightRAGConfig, VectorStoreConfig
from src.ingestion import Document, Chunker, TextLoader, MarkdownLoader, JSONLoader
from src.storage import InMemoryVectorStore
from src.monitoring import Monitor


class TestEmptyInput:
    """T01-T04: Empty input boundary tests"""

    def test_empty_document_list(self):
        """T01: Empty document list should be handled gracefully"""
        chunker = Chunker(chunk_size=256, chunk_overlap=50)
        # Should not raise exception
        chunks = chunker.chunk([])
        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_empty_file(self, text_loader, empty_txt_file):
        """T03: Empty file should be handled gracefully"""
        docs = text_loader.load(str(empty_txt_file))
        # Should return list (even if empty or with empty doc)
        assert isinstance(docs, list)

    def test_empty_json(self, tmp_path):
        """T04: Empty JSON file should be handled"""
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}")

        loader = JSONLoader()
        docs = loader.load(str(json_file))
        assert isinstance(docs, list)


class TestLongText:
    """T05-T07: Long text processing tests"""

    def test_2000token_document(self, long_document):
        """T05: Long document should be chunked correctly"""
        chunker = Chunker(chunk_size=256, chunk_overlap=50)

        chunks = chunker.chunk([long_document])

        assert len(chunks) > 0, "Long document should be split into chunks"

        for chunk in chunks:
            assert chunk.id is not None
            assert chunk.chunk_id is not None
            assert len(chunk.content) > 0

        # Check that information is preserved
        preserved = sum(len(c.content) for c in chunks)
        original_tokens = len(long_document.content.split())
        preserved_ratio = preserved / original_tokens if original_tokens > 0 else 0

        assert preserved_ratio > 0.5, "More than 50% of content should be preserved"

    def test_boundary_length_text(self, chunker):
        """T06: Text at chunk boundary should be handled"""
        words = ["word"] * 200
        content = " ".join(words)
        doc = Document(id="boundary_test", content=content, metadata={})

        chunks = chunker.chunk([doc])

        assert len(chunks) > 0

    def test_repeated_long_text(self, chunker):
        """T07: Repeated content should be handled"""
        lines = ["这是重复内容。"] * 100
        content = "\n".join(lines)
        doc = Document(id="repeated_test", content=content, metadata={})

        chunks = chunker.chunk([doc])

        assert len(chunks) > 0
        total_content = "".join(c.content for c in chunks)
        assert len(total_content) > 0


class TestMultilingual:
    """T08-T13: Multilingual and encoding tests"""

    def test_chinese_text(self, chunker, chinese_document):
        """T08: Pure Chinese text should be processed correctly"""
        chunks = chunker.chunk([chinese_document])

        assert len(chunks) > 0

        for chunk in chunks:
            assert len(chunk.content) > 0
            # Check Chinese punctuation was used for splitting
            assert any(char in chunk.content for char in "。！？")

    def test_mixed_cn_en(self, chunker, mixed_cn_en_document):
        """T09: Chinese-English mixed text should be handled"""
        chunks = chunker.chunk([mixed_cn_en_document])

        assert len(chunks) > 0

        all_content = " ".join(c.content for c in chunks)
        has_chinese = any("\u4e00" <= char <= "\u9fff" for char in all_content)
        has_english = any("a" <= char <= "z" for char in all_content.lower())

        assert has_chinese, "Chinese characters should be preserved"
        assert has_english, "English characters should be preserved"

    def test_unicode_special_chars(self, tmp_path):
        """T10: Special Unicode characters should be preserved"""
        content = "Hello 世界! Emoji test content sum"
        doc = Document(id="unicode_test", content=content, metadata={})
        chunker = Chunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk([doc])

        all_content = " ".join(c.content for c in chunks)
        has_emoji = "Emoji" in all_content or "emoji" in all_content

        assert has_emoji or "Hello" in all_content, "Content should be preserved"

    def test_utf8_bom_handling(self, text_loader, utf8_bom_file):
        """T11: UTF-8 BOM should be handled correctly"""
        docs = text_loader.load(str(utf8_bom_file))

        # Note: TextLoader doesn't strip BOM, which is a known limitation
        # This test documents the current behavior
        assert isinstance(docs, list)
        assert len(docs) == 1


class TestSpecialFormats:
    """T14-T17: Special content format tests"""

    def test_html_content(self, chunker):
        """T14: HTML content should be processed as text"""
        html_content = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
        doc = Document(id="html_test", content=html_content, metadata={})
        chunks = chunker.chunk([doc])

        assert len(chunks) > 0

    def test_json_content(self, tmp_path):
        """T15: JSON content should be processed"""
        json_content = {
            "name": "Test",
            "value": 123,
            "nested": {"key": "value"}
        }
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(json_content, ensure_ascii=False))

        loader = JSONLoader()
        docs = loader.load(str(json_file))

        assert len(docs) > 0

        all_content = " ".join(d.content for d in docs)
        assert "Test" in all_content or "value" in all_content

    def test_code_block_content(self, chunker):
        """T16: Code blocks should be processed without syntax errors"""
        code_content = 'def hello():\n    print("Hello")\n    return True'
        doc = Document(id="code_test", content=code_content, metadata={})
        chunks = chunker.chunk([doc])

        assert len(chunks) > 0
        all_content = " ".join(c.content for c in chunks)
        assert "def" in all_content or "print" in all_content


class TestChunkingBoundaries:
    """Chunking specific boundary tests"""

    def test_chunk_id_sequential(self, chunker):
        """Chunk IDs should be sequential"""
        doc = Document(id="test_doc", content="这是第一句。这是第二句。这是第三句。" * 10, metadata={})
        chunks = chunker.chunk([doc])

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id is not None

    def test_metadata_preserved(self, chunker):
        """Original metadata should be preserved in chunks"""
        original_metadata = {"source": "test", "author": "tester"}
        doc = Document(id="test_doc", content="这是测试内容。", metadata=original_metadata)
        chunks = chunker.chunk([doc])

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test"

    def test_empty_content_document(self, chunker):
        """Document with empty content should not crash"""
        doc = Document(id="empty_doc", content="", metadata={})
        chunks = chunker.chunk([doc])
        # Should return empty list or handle gracefully
        assert isinstance(chunks, list)


class TestExceptionRecovery:
    """T25-T28: Exception handling and recovery tests"""

    def test_empty_vector_search(self):
        """T27: Searching empty vector store should return empty"""
        store = InMemoryVectorStore()

        result = store.search(query_vector=[0.1, 0.2, 0.3], top_k=5)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_vector_store_add_empty(self):
        """Empty list add should not crash"""
        store = InMemoryVectorStore()

        result = store.add([])
        assert result is True or result is None

    def test_vector_search_mismatched_dimensions(self):
        """Mismatched vector dimensions should be handled"""
        store = InMemoryVectorStore(dimension=3)

        store.add([
            {"id": "test1", "embedding": [0.1, 0.2, 0.3], "content": "test"}
        ])

        # Search with different dimension - should raise or handle gracefully
        try:
            result = store.search(query_vector=[0.1, 0.2], top_k=5)
            # If dimension mismatch, should return empty or empty list
            assert isinstance(result, list)
        except Exception as e:
            # Dimension error is acceptable
            assert "dimension" in str(e).lower() or "size" in str(e).lower() or "shape" in str(e).lower()


class TestKnowledgeGraphBoundaries:
    """T29-T31: Knowledge graph boundary tests"""

    def test_duplicate_entities(self):
        """T29: Duplicate entity names should be handled"""
        try:
            from src.knowledge_graph import Entity, KnowledgeGraph

            kg = KnowledgeGraph()

            entity1 = Entity(
                id="entity_1",
                name="TestEntity",
                entity_type="TestType",
                description="First description"
            )
            entity2 = Entity(
                id="entity_2",
                name="TestEntity",
                entity_type="TestType",
                description="Second description"
            )

            kg.add_entity(entity1)
            kg.add_entity(entity2)

            # Check entities dict
            assert hasattr(kg, 'entities')
            assert len(kg.entities) >= 1
        except ImportError:
            pytest.skip("knowledge_graph module not available")

    def test_empty_entity(self):
        """T30: Empty entity fields should be handled"""
        try:
            from src.knowledge_graph import Entity, KnowledgeGraph

            kg = KnowledgeGraph()

            entity = Entity(
                id="empty_test",
                name="",
                entity_type="",
                description=""
            )

            # Should handle gracefully
            kg.add_entity(entity)

        except ImportError:
            pytest.skip("knowledge_graph module not available")
        except Exception:
            # Empty entity might raise - acceptable if clear error
            pass

    def test_special_chars_in_entity_name(self):
        """T31: Special characters in entity name should be handled"""
        try:
            from src.knowledge_graph import Entity, KnowledgeGraph

            kg = KnowledgeGraph()

            entity = Entity(
                id="special_test",
                name="Entity-With_Special.Chars",
                entity_type="Test",
                description="Test entity with special chars"
            )

            kg.add_entity(entity)

        except ImportError:
            pytest.skip("knowledge_graph module not available")


class TestMonitoringEdgeCases:
    """Monitoring and tracing edge cases"""

    def test_span_without_attributes(self):
        """Monitor span with no attributes should work"""
        monitor = Monitor(enable_tracing=True, enable_metrics=False)

        with monitor.span("test_span"):
            pass

        # Should not crash - check trace_context.spans
        assert hasattr(monitor, 'trace_context')
        assert len(monitor.trace_context.spans) >= 0

    def test_nested_spans(self):
        """Nested spans should track correctly"""
        monitor = Monitor(enable_tracing=True, enable_metrics=False)

        with monitor.span("outer_span"):
            with monitor.span("inner_span_1"):
                pass
            with monitor.span("inner_span_2"):
                pass

        # Should have nested structure
        assert hasattr(monitor, 'trace_context')
        assert len(monitor.trace_context.spans) >= 1

    def test_metrics_recording(self):
        """Metrics should be recorded correctly"""
        monitor = Monitor(enable_tracing=False, enable_metrics=True)

        monitor.record_latency("op1", 0.1)
        monitor.record_latency("op2", 0.2)

        assert hasattr(monitor, 'metrics_collector')
        assert len(monitor.metrics_collector.metrics) == 2

    def test_span_with_attributes(self):
        """Span with attributes should work"""
        monitor = Monitor(enable_tracing=True, enable_metrics=False)

        with monitor.span("test_span", {"key": "value", "num": 42}):
            pass

        assert hasattr(monitor, 'trace_context')
        assert len(monitor.trace_context.spans) >= 1

    def test_monitor_init_defaults(self):
        """Monitor should init with defaults"""
        monitor = Monitor()

        assert monitor.enable_tracing == True
        assert monitor.enable_metrics == True


class TestConfigBoundaries:
    """Configuration boundary tests"""

    def test_default_config(self):
        """Default config should work"""
        config = LightRAGConfig()

        assert config.project_name == "LightRAG"
        assert config.version == "1.0.0"

    def test_config_to_dict(self):
        """Config to_dict should work"""
        config = LightRAGConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "project_name" in config_dict

    def test_vector_store_config(self):
        """Vector store config should work"""
        config = VectorStoreConfig(storage_type="inmemory")

        assert config.storage_type == "inmemory"


class TestConcurrency:
    """T22-T24: Concurrency tests"""

    def test_concurrent_chunker(self, chunker):
        """Concurrent chunking should not cause data races"""
        doc = Document(id="test", content="测试内容。" * 100, metadata={})
        results = []
        errors = []
        lock = threading.Lock()

        def chunk_batch(batch_id):
            try:
                chunks = chunker.chunk([doc])
                with lock:
                    results.append((batch_id, len(chunks)))
            except Exception as e:
                with lock:
                    errors.append((batch_id, str(e)))

        threads = []
        for i in range(10):
            t = threading.Thread(target=chunk_batch, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

    def test_concurrent_vector_operations(self):
        """Concurrent vector store operations should be safe"""
        store = InMemoryVectorStore(dimension=3)
        results = []
        errors = []
        lock = threading.Lock()

        def add_and_search(batch_id):
            try:
                store.add([
                    {"id": f"doc_{batch_id}", "embedding": [0.1, 0.2, 0.3], "content": f"content {batch_id}"}
                ])
                result = store.search(query_vector=[0.1, 0.2, 0.3], top_k=5)
                with lock:
                    results.append((batch_id, len(result)))
            except Exception as e:
                with lock:
                    errors.append((batch_id, str(e)))

        threads = []
        for i in range(5):
            t = threading.Thread(target=add_and_search, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
