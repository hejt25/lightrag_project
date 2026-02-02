"""
Embedding Module
================
向量化模块 - 负责文本的向量化处理
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """向量化基类"""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本列表向量化"""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """向量化查询文本"""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """基于Sentence-Transformers的向量化"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量向量化"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """向量化查询"""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return np.array(embedding).squeeze()

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension


class HuggingFaceEmbedder(BaseEmbedder):
    """基于HuggingFace Transformers的向量化（本地模型）"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh", device: str = "cpu"):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            logger.error("transformers or torch not installed")
            raise ImportError("Please install transformers and torch")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.model_name = model_name

        # 获取模型维度
        with torch.no_grad():
            dummy_input = self.tokenizer("test", return_tensors="pt").to(device)
            self.dimension = self.model(**dummy_input).last_hidden_state.shape[2]

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量向量化"""
        import torch

        all_embeddings = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**encoded)
                # 使用[CLS] token的输出或平均池化
                embeddings = outputs.last_hidden_state.mean(dim=1)
                # L2归一化
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """向量化查询"""
        import torch

        with torch.no_grad():
            encoded = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**encoded)
            embedding = outputs.last_hidden_state.mean(dim=1)
            embedding = embedding / embedding.norm(dim=1, keepdim=True)

        return embedding.squeeze().cpu().numpy()

    def get_dimension(self) -> int:
        return self.dimension


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API向量化"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        try:
            import openai
        except ImportError:
            logger.error("openai not installed")
            raise ImportError("Please install openai: pip install openai")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量向量化"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """向量化查询"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        return np.array(response.data[0].embedding)

    def get_dimension(self) -> int:
        # OpenAI text-embedding-3-small 维度为1536
        return 1536


class EmbeddingPipeline:
    """向量化管道"""

    def __init__(self, embedder: BaseEmbedder, batch_size: int = 32):
        self.embedder = embedder
        self.batch_size = batch_size

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对文档列表进行向量化"""
        texts = [doc.get("content", "") for doc in documents]

        logger.info(f"Embedding {len(texts)} documents...")

        embeddings = self.embedder.embed(texts)

        # 为每个文档添加向量
        for i, doc in enumerate(documents):
            doc["embedding"] = embeddings[i].tolist()
            doc["embedding_dimension"] = len(embeddings[i])

        logger.info(f"Completed embedding, dimension: {embeddings.shape[1]}")

        return documents

    def embed_query(self, query: str) -> np.ndarray:
        """向量化查询"""
        return self.embedder.embed_query(query)

    def get_dimension(self) -> int:
        return self.embedder.get_dimension()
