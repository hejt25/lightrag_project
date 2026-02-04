"""
Pytest fixtures for LightRAG tests.
"""
import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from src.pipeline import LightRAGPipeline, EasyLightRAG
from src.config import LightRAGConfig, VectorStoreConfig
from src.ingestion import Document, Chunker, TextLoader, MarkdownLoader, JSONLoader
from src.retrieval import RetrievalPipeline, VectorRetriever
from src.monitoring import Monitor


@pytest.fixture
def rag_pipeline(tmp_path):
    """Create a test RAG pipeline instance with in-memory storage."""
    config = LightRAGConfig(
        project_name="TestLightRAG",
        vector_store=VectorStoreConfig(storage_type="inmemory")
    )
    pipeline = LightRAGPipeline(config=config)
    yield pipeline


@pytest.fixture
def easy_rag(tmp_path):
    """Create a test EasyLightRAG instance."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    rag = EasyLightRAG(data_dir=str(data_dir))
    yield rag


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary test data directory."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="test_doc_001",
        content="LightRAG 是一个强大的检索增强生成框架。"
                "它支持多种数据格式，包括文本、Markdown 和 JSON。"
                "主要功能包括文档分块、向量化、存储和检索。"
                "用户可以通过简单的 API 进行查询。",
        metadata={"source": "test", "type": "text"}
    )


@pytest.fixture
def sample_documents():
    """Create multiple sample documents for testing."""
    return [
        Document(
            id=f"test_doc_{i:03d}",
            content=f"这是测试文档 {i} 的内容，包含关于 LightRAG 的信息。文档 {i} 讨论了不同的主题。",
            metadata={"source": "test", "type": "text"}
        )
        for i in range(1, 11)
    ]


@pytest.fixture
def chinese_document():
    """Create a Chinese document for testing."""
    return Document(
        id="chinese_doc_001",
        content="""
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
        它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式
        做出反应的智能机器。研究范围包括机器学习、自然语言处理、
        计算机视觉等多个领域。深度学习是当前最热门的研究方向之一。
        """,
        metadata={"source": "test", "type": "text"}
    )


@pytest.fixture
def mixed_cn_en_document():
    """Create a Chinese-English mixed document."""
    return Document(
        id="mixed_doc_001",
        content="""
        LightRAG 是一个 RAG (Retrieval-Augmented Generation) 框架。
        它使用 Large Language Model (LLM) 来增强 Retrieval 的效果。
        主要特性包括：Installation、Configuration、Usage。
        支持 Vector Store、Embedding、Retrieval 等功能。
        """,
        metadata={"source": "test", "type": "text"}
    )


@pytest.fixture
def long_document():
    """Create a long document (>2000 tokens equivalent)."""
    content_parts = [
        "这是长文档的第一部分，介绍 LightRAG 的背景。"
    ]
    # Repeat to create length
    for i in range(50):
        content_parts.append(
            f"第 {i + 1} 章：详细讨论某个技术主题。"
            f"这部分包含大量的技术细节、分析和解释。"
            f"LightRAG 在这个领域有独特的优势。"
            f"我们继续深入探讨更多的实现细节。"
            f"包括架构设计、性能优化、可扩展性等方面。"
        )
    content_parts.append("这是长文档的结论部分，总结主要观点。")
    return Document(
        id="long_doc_001",
        content="\n".join(content_parts),
        metadata={"source": "test", "type": "text"}
    )


@pytest.fixture
def chunker():
    """Create a Chunker instance for testing."""
    return Chunker(chunk_size=256, chunk_overlap=50)


@pytest.fixture
def text_loader():
    """Create a TextLoader instance for testing."""
    return TextLoader()


@pytest.fixture
def indexed_pipeline(rag_pipeline, sample_documents):
    """Create a pipeline with indexed documents for retrieval testing."""
    rag_pipeline.index_documents(sample_documents)
    return rag_pipeline


@pytest.fixture
def indexed_pipeline_with_chunks(rag_pipeline, chunker, sample_document):
    """Create a pipeline with properly chunked documents."""
    chunks = chunker.chunk([sample_document])
    rag_pipeline.index_documents(chunks)
    return rag_pipeline


# Utility fixtures for creating test files
@pytest.fixture
def empty_txt_file(test_data_dir):
    """Create an empty .txt file."""
    file_path = test_data_dir / "empty.txt"
    file_path.write_text("")
    return file_path


@pytest.fixture
def empty_json_file(test_data_dir):
    """Create an empty JSON file."""
    file_path = test_data_dir / "empty.json"
    file_path.write_text("{}")
    return file_path


@pytest.fixture
def utf8_bom_file(test_data_dir):
    """Create a UTF-8 BOM file."""
    file_path = test_data_dir / "utf8_bom.txt"
    # Write BOM followed by content
    file_path.write_bytes(b'\xef\xbb\xbf' + "Hello World with UTF-8 BOM".encode('utf-8'))
    return file_path


@pytest.fixture
def special_chars_file(test_data_dir):
    """Create a file with special characters."""
    file_path = test_data_dir / "special_chars.txt"
    content = "Hello 世界! Emoji测试: 数学符号: sum"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def html_fragment_file(test_data_dir):
    """Create an HTML fragment file."""
    file_path = test_data_dir / "html_fragment.txt"
    content = """<html>
<body>
    <h1>Title</h1>
    <p>This is a <strong>test</strong> paragraph.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
</body>
</html>"""
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def code_block_file(test_data_dir):
    """Create a code block file."""
    file_path = test_data_dir / "code_block.py"
    content = '''def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True

class Calculator:
    def __init__(self, initial=0):
        self.value = initial

    def add(self, x):
        self.value += x
        return self
'''
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def chinese_txt_file(test_data_dir):
    """Create a Chinese text file."""
    file_path = test_data_dir / "chinese.txt"
    content = """人工智能（AI）是计算机科学的重要分支。

深度学习是当前最热门的技术方向。机器学习作为AI的核心，在各个领域都有广泛应用。

自然语言处理（NLP）让计算机能够理解和生成人类语言。"""
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def mixed_cn_en_file(test_data_dir):
    """Create a Chinese-English mixed file."""
    file_path = test_data_dir / "mixed_cn_en.txt"
    content = """LightRAG 是一个 RAG 框架。

It supports multiple data formats: Text, Markdown, JSON.

主要功能包括：Embedding、Retrieval、Generation。"""
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def long_txt_file(test_data_dir):
    """Create a long text file (>2000 tokens)."""
    file_path = test_data_dir / "long_text.txt"
    content_parts = ["# LightRAG 详细技术文档\n"]
    for i in range(100):
        content_parts.append(f"""
## 第 {i + 1} 章：技术深度分析

LightRAG 框架在处理{i}相关任务时展现出卓越的性能。

### 架构设计

系统采用模块化设计，主要包含以下组件：
1. 数据摄入模块（Ingestion）
2. 向量化模块（Embedding）
3. 存储模块（Storage）
4. 检索模块（Retrieval）
5. 生成模块（Generation）

### 性能优化

在{i * 10}次测试中，平均响应时间为{i * 0.5}毫秒。

### 扩展性

系统支持水平扩展，可处理百万级文档。
""")
    content_parts.append("\n## 结论\n本文档详细介绍了 LightRAG 的各个方面。")
    file_path.write_text("\n".join(content_parts), encoding='utf-8')
    return file_path
