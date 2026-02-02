"""
Data Ingestion Module
=====================
数据摄入模块 - 负责数据的读取、清洗、分块
"""

import os
import re
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """文档数据类"""
    id: str
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None


@dataclass
class ChunkedDocument:
    """分块后的文档"""
    id: str
    chunks: List[Document]
    total_chunks: int


class BaseLoader(ABC):
    """文档加载器基类"""

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """加载文档"""
        pass


class TextLoader(BaseLoader):
    """纯文本加载器"""

    def load(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(
            id=Path(file_path).stem,
            content=content,
            metadata={"source": file_path, "type": "text"}
        )]


class MarkdownLoader(BaseLoader):
    """Markdown加载器"""

    def load(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取标题和内容
        lines = content.split('\n')
        metadata = {
            "source": file_path,
            "type": "markdown",
            "title": ""
        }

        # 提取YAML front matter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) > 1:
                yaml_content = parts[1]
                for line in yaml_content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip().strip('"')

        return [Document(
            id=Path(file_path).stem,
            content=content,
            metadata=metadata
        )]


class JSONLoader(BaseLoader):
    """JSON加载器"""

    def load(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            documents = []
            for i, item in enumerate(data):
                documents.append(Document(
                    id=f"{Path(file_path).stem}_{i}",
                    content=json.dumps(item, ensure_ascii=False),
                    metadata={"source": file_path, "type": "json", "index": i}
                ))
            return documents
        else:
            return [Document(
                id=Path(file_path).stem,
                content=json.dumps(data, ensure_ascii=False),
                metadata={"source": file_path, "type": "json"}
            )]


class PDFLoader(BaseLoader):
    """PDF加载器（需要安装PyPDF2或pdfplumber）"""

    def load(self, file_path: str) -> List[Document]:
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed, using placeholder")
            return [Document(
                id=Path(file_path).stem,
                content=f"PDF content from {file_path}",
                metadata={"source": file_path, "type": "pdf"}
            )]

        with pdfplumber.open(file_path) as pdf:
            content = "\n".join(page.extract_text() or "" for page in pdf.pages)

        return [Document(
            id=Path(file_path).stem,
            content=content,
            metadata={"source": file_path, "type": "pdf", "pages": len(pdf.pages)}
        )]


class Chunker:
    """文本分块器"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: List[Document]) -> List[Document]:
        """对文档列表进行分块"""
        chunked_docs = []
        for doc in documents:
            chunks = self._split_document(doc)
            chunked_docs.extend(chunks)
        return chunked_docs

    def _split_document(self, doc: Document) -> List[Document]:
        """拆分单个文档"""
        content = doc.content
        chunks = []

        # 按句子分割
        sentences = self._split_into_sentences(content)

        current_chunk = ""
        current_size = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.chunk_size and current_chunk:
                # 保存当前块
                chunk = Document(
                    id=f"{doc.id}_chunk_{chunk_id}",
                    content=current_chunk.strip(),
                    metadata={**doc.metadata, "chunk_index": chunk_id},
                    chunk_id=f"{doc.id}_chunk_{chunk_id}"
                )
                chunks.append(chunk)
                chunk_id += 1

                # 保留重叠部分
                overlap_size = self._get_overlap_size(current_chunk, self.chunk_overlap)
                current_chunk = current_chunk[-overlap_size:] + " " + sentence
                current_size = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " "
                current_chunk += sentence
                current_size += sentence_size + 1

        # 保存最后一个块
        if current_chunk.strip():
            chunk = Document(
                id=f"{doc.id}_chunk_{chunk_id}",
                content=current_chunk.strip(),
                metadata={**doc.metadata, "chunk_index": chunk_id},
                chunk_id=f"{doc.id}_chunk_{chunk_id}"
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        # 简单的句子分割
        sentences = re.split(r'(?<=[。！？?!])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_size(self, text: str, target_overlap: int) -> int:
        """获取重叠部分大小"""
        words = text.split()
        if len(words) <= target_overlap:
            return len(text)
        return len(' '.join(words[-target_overlap:]))


class DataIngestionPipeline:
    """数据摄入管道"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.loaders = {
            '.txt': TextLoader(),
            '.md': MarkdownLoader(),
            '.json': JSONLoader(),
            '.pdf': PDFLoader()
        }
        self.chunker = Chunker(chunk_size, chunk_overlap)

    def load_directory(self, data_dir: str) -> List[Document]:
        """加载目录中的所有文档"""
        all_docs = []
        data_path = Path(data_dir)

        for file_path in data_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.loaders:
                try:
                    loader = self.loaders[file_path.suffix]
                    docs = loader.load(str(file_path))
                    all_docs.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        return all_docs

    def process(self, data_dir: str) -> List[Document]:
        """完整的数据处理流程"""
        # 1. 加载文档
        documents = self.load_directory(data_dir)
        logger.info(f"Loaded {len(documents)} documents")

        # 2. 分块
        chunked_docs = self.chunker.chunk(documents)
        logger.info(f"Created {len(chunked_docs)} chunks")

        return chunked_docs

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """直接处理文档列表"""
        chunked_docs = self.chunker.chunk(documents)
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
