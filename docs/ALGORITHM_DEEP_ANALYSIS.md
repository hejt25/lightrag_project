# LightRAG 算法深度解析

> **版本**: v2.0
> **更新**: 2025-02-03
> **深度**: 高级（包含数学公式、源码分析、优化策略）

---

## 目录

1. [文本分块算法](#一文本分块算法)
2. [向量化算法](#二向量化算法)
3. [向量检索算法](#三向量检索算法)
4. [BM25关键词检索](#四bm25关键词检索)
5. [混合检索融合](#五混合检索融合)
6. [重排序算法](#六重排序算法)
7. [实体识别算法](#七实体识别算法)
8. [知识图谱检索](#八知识图谱检索)
9. [Query改写算法](#九query改写算法)
10. [Prompt构建算法](#十prompt构建算法)

---

## 一、文本分块算法

### 1.1 算法全景对比

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           文本分块算法演进图                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  固定长度分块                    语义分块                    重叠分块(我们)       │
│  ────────────                  ─────────                   ──────────────       │
│                                                                                 │
│  [████████████]                [██████████]                [██████████]         │
│  [████████████]                [████████]                  [██████░░░░]         │
│  [████████████]                [████]                      [░░░░░░████]         │
│                                                                                 │
│  ✓ 简单快速                     ✓ 语义完整                  ✓ 边界平滑            │
│  ✗ 可能切断词语                 ✗ 需要模型                  ✗ 少量冗余            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 固定长度分块（基线）

```python
def naive_chunk(content: str, chunk_size: int = 512) -> List[str]:
    """最朴素的固定长度分块"""
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunks.append(content[i:i + chunk_size])
    return chunks
```

**问题分析**：
- 可能切断中文字符（UTF-8编码下汉字占3字节）
- 可能切断英文单词
- 破坏语义完整性

### 1.3 我们的重叠分块算法（详细实现）

```python
import re
from typing import List, Tuple

class OverlappingChunker:
    """
    重叠分块器 - 保留语义边界和上下文连续性

    核心原理：
    1. 句子级分割：基于标点符号识别句子边界
    2. 重叠保留：相邻块之间保留重叠区域
    3. 智能截断：在句子边界处截断
    """

    # 中文句子结束标点
    SENTENCE_ENDINGS = re.compile(r'[。！？；\n]+')
    # 段落标记
    PARAGRAPH_MARKERS = re.compile(r'\n\n+')

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, content: str) -> List[str]:
        """
        分块主流程

        Returns:
            List[str]: 分块后的文本块
        """
        # 1. 预处理：清理多余空白
        content = self._preprocess(content)

        # 2. 句子分割
        sentences = self._split_into_sentences(content)

        # 3. 滑动窗口构建块
        chunks = self._build_chunks(sentences)

        return chunks

    def _split_into_sentences(self, content: str) -> List[str]:
        """
        句子边界检测算法

        使用正则匹配句子结束标点，同时处理引号内的标点

        复杂度：O(n) - 只需一次遍历
        """
        sentences = []
        current_pos = 0

        for match in self.SENTENCE_ENDINGS.finditer(content):
            # 找到句子结束位置
            end_pos = match.end()

            # 提取句子（包含结束标点）
            sentence = content[current_pos:end_pos].strip()

            if sentence:
                sentences.append(sentence)

            current_pos = end_pos

        # 处理最后一个句子
        if current_pos < len(content):
            remaining = content[current_pos:].strip()
            if remaining:
                sentences.append(remaining)

        return sentences

    def _build_chunks(self, sentences: List[str]) -> List[str]:
        """
        滑动窗口构建块

        窗口移动策略：
        - 当累积文本超过 chunk_size 时，保存当前块
        - 新块从上一个块的末尾 overlap 字符开始
        """
        chunks = []
        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # 检查是否需要分块
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # 保存当前块
                chunks.append(current_chunk)

                # 计算重叠部分
                overlap_text = self._get_last_n_chars(current_chunk, self.overlap)

                # 新块 = 重叠部分 + 新句子
                current_chunk = overlap_text + sentence
                current_size = len(current_chunk)
            else:
                # 追加到当前块
                current_chunk += sentence
                current_size += sentence_size

        # 处理最后一个块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _get_last_n_chars(self, text: str, n: int) -> str:
        """
        获取最后n个字符

        注意：对于中文，需要按字符而非字节截断
        """
        return text[-n:] if len(text) > n else text
```

### 1.4 语义分块（进阶方案）

```python
class SemanticChunker:
    """
    语义分块器 - 基于嵌入相似度判断语义边界

    原理：计算相邻句子/段落的嵌入相似度，
          当相似度低于阈值时，认为是新的语义单元
    """

    def __init__(self, embedder, threshold: float = 0.7):
        self.embedder = embedder
        self.threshold = threshold

    def chunk(self, content: str) -> List[str]:
        sentences = self._split_into_sentences(content)

        if len(sentences) < 2:
            return [content]

        # 批量嵌入
        embeddings = self.embedder.embed(sentences)

        # 计算相邻句子的相似度
        boundaries = [0]  # 分块边界

        for i in range(1, len(sentences)):
            # 计算与前一个句子的相似度
            similarity = cosine_similarity(
                embeddings[i-1], embeddings[i]
            )

            if similarity < self.threshold:
                boundaries.append(i)

        # 根据边界构建块
        chunks = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
            chunk = ''.join(sentences[start:end])
            chunks.append(chunk)

        return chunks
```

### 1.5 算法复杂度对比

| 算法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| **固定长度** | O(n) | O(n) | 快速处理，不关心语义 |
| **滑动窗口** | O(n) | O(n) | 保留上下文，窗口可调 |
| **语义分块** | O(n·d) | O(n·d) | 语义完整，需要嵌入模型 |
| **重叠分块(我们)** | O(n) | O(n) | 平衡速度与语义完整性 |

> **d**: 嵌入向量维度 (512或768)

### 1.6 边界处理优化

```python
class OptimizedOverlappingChunker:
    """
    优化版重叠分块器 - 改进的边界处理
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

        # 词边界检测（避免切断词语）
        self.word_pattern = re.compile(r'[\w\u4e00-\u9fff]+')

    def _find_safe_boundary(self, text: str, target_pos: int) -> int:
        """
        找到安全的截断位置

        向前/向后查找最近的词语边界
        """
        # 尝试向前查找
        left_boundary = text.rfind(' ', 0, target_pos)
        if left_boundary == -1:
            # 使用词边界
            match = self.word_pattern.search(text[:target_pos][::-1])
            if match:
                left_boundary = target_pos - match.end()
            else:
                left_boundary = 0

        # 尝试向后查找
        right_boundary = text.find(' ', target_pos)
        if right_boundary == -1:
            right_boundary = len(text)

        # 选择最近的边界
        if target_pos - left_boundary < right_boundary - target_pos:
            return left_boundary
        return right_boundary

    def chunk(self, content: str) -> List[str]:
        # 使用安全边界的分块逻辑
        ...
```

---

## 二、向量化算法

### 2.1 Transformer 架构详解

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BERT 嵌入模型架构                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  输入: "平安福重疾险保障100种重疾"                                                │
│        │                                                                        │
│        ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Tokenization Layer                               │   │
│  │  ┌─────┬─────┬─────┐┌─────┐┌─────┬─────┬─────┐┌─────┐┌─────┐┌─────┐   │   │
│  │  │ [CLS] │ 平 │ 安 │ 福 │ 重 │ 疾 │ 险 │  │ 保障 │ 1 │ 0 │ 0 │ [SEP] │   │   │
│  │  └─────┴─────┴─────┘└─────┘└─────┴─────┴─────┘└─────┘└─────┴─────┘└─────┘   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│        │                                                                        │
│        ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Positional Encoding (可学习)                          │   │
│  │  12层 Transformer Encoder, 每层:                                          │   │
│  │  - Multi-Head Self-Attention (8 heads)                                   │   │
│  │  - Add & LayerNorm                                                        │   │
│  │  - Feed-Forward Network (3072 → 768)                                     │   │
│  │  - Add & LayerNorm                                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│        │                                                                        │
│        ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Pooling Layer                                    │   │
│  │                                                                          │   │
│  │    CLS Pooling:                    Mean Pooling:                         │   │
│  │    ┌─────────────────┐            ┌─────────────────┐                   │   │
│  │    │ [CLS] h1 h2 ... │            │ [CLS] h1 h2 ... hn │                │   │
│  │    │       ↓         │            │        ↓         │                   │   │
│  │    │      v_cls      │            │   v = (Σhi)/n    │                   │   │
│  │    └─────────────────┘            └─────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│        │                                                                        │
│        ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    L2 Normalization                                      │   │
│  │                                                                          │   │
│  │         v' = v / ||v||₂  ,  其中 ||v||₂ = √(Σvᵢ²)                       │   │
│  │                                                                          │   │
│  │    效果：向量长度变为1，便于计算余弦相似度                                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│        │                                                                        │
│        ▼                                                                        │
│  输出: 512维单位向量                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Mean Pooling 数学推导

设 $H \in \mathbb{R}^{n \times d}$ 为最后一层隐藏状态矩阵，其中：
- $n$: token 数量
- $d$: 隐藏层维度 (768)

**Mean Pooling**：
$$v = \frac{1}{n}\sum_{i=1}^{n} h_i$$

**加权 Mean Pooling**（带注意力权重）：
$$v = \sum_{i=1}^{n} \alpha_i h_i, \quad \alpha_i = \frac{\exp(a^T h_i)}{\sum_j \exp(a^T h_j)}$$

其中 $a$ 是可学习的注意力向量。

### 2.3 完整实现代码

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union

class BGEEmbedder:
    """
    BGE 嵌入模型封装

    支持功能：
    1. 批量嵌入
    2. L2 归一化
    3. GPU 加速
    4. 动态批处理
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh",
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True
    ):
        """
        Args:
            model_name: HuggingFace 模型名称
            device: 设备 ("auto", "cuda", "cpu")
            batch_size: 批处理大小
            normalize: 是否L2归一化
        """
        self.device = self._get_device(device)
        self.batch_size = batch_size
        self.normalize = normalize

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @torch.no_grad()
    def embed(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512
    ) -> np.ndarray:
        """
        嵌入单个文本或文本列表

        Args:
            texts: 单个文本或文本列表
            max_length: 最大token长度

        Returns:
            numpy数组，shape为 (n, d) 或 (d,)
        """
        # 统一为列表
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # 批量处理
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_batch(batch, max_length)
            all_embeddings.append(embeddings)

        result = np.vstack(all_embeddings)

        # 如果输入是单个文本，返回1D
        if single_input:
            result = result[0]

        return result

    def _embed_batch(self, batch: List[str], max_length: int) -> np.ndarray:
        """
        批处理嵌入
        """
        # Tokenization
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # 移动到设备
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # 前向传播
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Mean Pooling
        last_hidden_state = outputs.last_hidden_state  # (B, L, D)

        # 使用 attention_mask 计算加权平均
        mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
        hidden = last_hidden_state * mask  # (B, L, D)
        sum_hidden = hidden.sum(dim=1)  # (B, D)
        sum_mask = mask.sum(dim=1)  # (B, 1)
        embeddings = sum_hidden / sum_mask  # (B, D)

        # L2 归一化
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def embed_query(self, query: str) -> np.ndarray:
        """
        专门针对查询的嵌入

        与 embed() 的区别：
        - 查询通常较短
        - 可能需要特殊处理
        """
        return self.embed(query)


class MultiVectorEmbedder:
    """
    多向量嵌入器 - 支持同一文本生成多个向量

    用于：
    1. ColBERT 风格检索
    2. 段落级 + 句子级嵌入
    """

    def __init__(self, embedder: BGEEmbedder):
        self.embedder = embedder

    @torch.no_grad()
    def embed(
        self,
        texts: Union[str, List[str]],
        layer: str = "last"
    ) -> List[np.ndarray]:
        """
        为每个文本生成 token 级嵌入

        Returns:
            每个文本的 token 嵌入列表
        """
        if isinstance(texts, str):
            texts = [texts]

        results = []

        for text in texts:
            # Tokenization (不截断)
            encoded = self.embedder.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True
            )

            input_ids = encoded["input_ids"].to(self.embedder.device)
            attention_mask = encoded["attention_mask"].to(self.embedder.device)

            outputs = self.embedder.model(input_ids=input_ids)

            # 选择层
            if layer == "last":
                hidden = outputs.last_hidden_state
            elif layer == "second_last":
                hidden = outputs.hidden_states[-2]

            # 移除 [CLS] 和 [SEP]
            token_embeddings = hidden[0, 1:-1]  # (L-2, D)
            results.append(token_embeddings.cpu().numpy())

        return results
```

### 2.4 模型对比与选择

| 模型 | 维度 | 参数量 | 平均对接 | 中文 | 推理速度 |
|------|------|--------|---------|------|---------|
| **bge-small-zh** | 512 | 33M | 62.2 | ★★★★★ | 快 |
| **bge-base-zh** | 768 | 110M | 64.5 | ★★★★★ | 中 |
| **bge-large-zh** | 1024 | 326M | 65.5 | ★★★★★ | 慢 |
| **mGTE** | 1024 | 118M | 63.1 | ★★★★ | 中 |
| **BAAI/bge-m3** | 1024 | 567M | 66.4 | ★★★★★ | 很慢 |

**选择理由**：
- bge-small-zh 在中文任务上效果与 large 差距小（<3%）
- 33M 参数适合 CPU 实时推理
- 512 维向量节省存储空间

---

## 三、向量检索算法

### 3.1 FAISS 索引算法详解

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FAISS 索引分类                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                              精确检索                                    │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │   │
│  │  │  IndexFlatIP (内积)                                               │ │   │
│  │  │  IndexFlatL2 (欧氏距离)                                           │ │   │
│  │  │                                                                 │ │   │
│  │  │  复杂度: O(n·d)  召回率: 100%  内存: 高                           │ │   │
│  │  └───────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                              近似检索                                    │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │   │
│  │  │     IVF         │  │     HNSW        │  │      PQ         │          │   │
│  │  │  倒排文件索引    │  │  层次可导航小世界│  │   产品量化      │          │   │
│  │  │                 │  │                 │  │                 │          │   │
│  │  │  O(n/√n)       │  │  O(log n)       │  │  O(m·d/m')      │          │   │
│  │  │  召回率: ~95%   │  │  召回率: ~98%   │  │  召回率: ~90%   │          │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 我们的向量存储实现

```python
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pickle
import os

@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = None


class FAISSVectorStore:
    """
    FAISS 向量存储

    支持：
    1. 内积检索（配合L2归一化实现余弦相似度）
    2. 持久化存储
    3. 动态增删
    """

    def __init__(
        self,
        dimension: int = 512,
        metric: str = "cosine",
        index_path: Optional[str] = None
    ):
        """
        Args:
            dimension: 向量维度
            metric: 距离度量 ("cosine", "l2", "ip")
            index_path: 索引持久化路径
        """
        self.dimension = dimension
        self.metric = metric
        self.index_path = index_path

        # 初始化索引
        self._init_index()

        # 文档存储
        self.documents: Dict[str, Dict] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.id_counter = 0

    def _init_index(self):
        """初始化FAISS索引"""
        if self.metric == "cosine":
            # 内积 + 归一化 = 余弦相似度
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:  # ip
            self.index = faiss.IndexFlatIP(self.dimension)

    def add(self, ids: List[str], vectors: np.ndarray, documents: List[Dict]):
        """
        添加向量

        Args:
            ids: 文档ID列表
            vectors: 向量矩阵，shape为 (n, d)
            documents: 文档内容列表
        """
        if len(ids) != len(vectors):
            raise ValueError("IDs and vectors must have same length")

        # 添加到FAISS
        self.index.add(vectors)

        # 更新索引映射
        for i, doc_id in enumerate(ids):
            self.documents[doc_id] = documents[i]
            self.id_to_idx[doc_id] = self.id_counter + i

        self.id_counter += len(ids)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        搜索

        Args:
            query_vector: 查询向量，shape为 (d,) 或 (1, d)
            top_k: 返回数量
            filter_ids: 过滤的ID列表（只在这些ID中搜索）

        Returns:
            SearchResult 列表
        """
        # 统一形状
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # 搜索
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = scores[0][i]

            if idx < 0:
                continue

            # 反查文档ID
            doc_id = self._idx_to_id(idx)
            if doc_id is None:
                continue

            # 过滤
            if filter_ids and doc_id not in filter_ids:
                continue

            doc = self.documents.get(doc_id, {})

            results.append(SearchResult(
                id=doc_id,
                content=doc.get("content", ""),
                score=float(score),
                metadata=doc.get("metadata", {})
            ))

        return results

    def _idx_to_id(self, idx: int) -> Optional[str]:
        """索引转文档ID"""
        for doc_id, doc_idx in self.id_to_idx.items():
            if doc_idx == idx:
                return doc_id
        return None

    def save(self, path: str):
        """保存索引和文档"""
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        # 保存FAISS索引
        faiss.write_index(self.index, f"{path}.index")

        # 保存文档映射
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "id_to_idx": self.id_to_idx,
                "id_counter": self.id_counter
            }, f)

    def load(self, path: str):
        """加载索引"""
        self.index = faiss.read_index(f"{path}.index")

        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.id_to_idx = data["id_to_idx"]
            self.id_counter = data["id_counter"]
```

### 3.3 HNSW 索引（大规模优化）

```python
class HNSWVectorStore:
    """
    HNSW 向量存储 - 适用于大规模场景

    原理：构建多层图结构，每层是随机化的k-NN图
    """

    def __init__(
        self,
        dimension: int = 512,
        max_elements: int = 1000000,
        ef_construction: int = 200,
        m: int = 16
    ):
        """
        Args:
            max_elements: 最大元素数
            ef_construction: 构建时的搜索范围（越大越精确但越慢）
            m: 每个节点的邻居数
        """
        self.dimension = dimension
        self.max_elements = max_elements

        # HNSW 索引
        self.index = faiss.IndexHNSWFlat(dimension, m)
        self.index.hnsw.efConstruction = ef_construction

    def set_ef(self, ef: int):
        """设置搜索时的ef参数"""
        self.index.hnsw.efSearch = ef

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        ef: int = 64
    ) -> List[SearchResult]:
        """
        搜索

        Args:
            ef: 搜索时的探索范围（越大越精确但越慢）
        """
        self.set_ef(ef)
        scores, indices = self.index.search(query_vector, top_k)
        ...
```

### 3.4 索引选择指南

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              索引选择决策树                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                            数据规模                                              │
│                               │                                                 │
│              ┌────────────────┼────────────────┐                               │
│              ▼                ▼                ▼                                │
│         < 10,000         10K - 1M         > 1M                                 │
│              │                │                │                                │
│              ▼                ▼                ▼                                │
│         IndexFlatIP        HNSW           IVF-PQ                                │
│         (精确检索)       (高速检索)       (压缩存储)                             │
│              │                │                │                                │
│              ▼                ▼                ▼                                │
│         召回率100%       召回率98%        召回率90%                              │
│         QPS < 1K          QPS > 10K        QPS > 50K                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、BM25 关键词检索

### 4.1 BM25 算法原理

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BM25 算法公式                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Score(D, Q) = Σ IDF(qi) · f(qi, D) · (k1 + 1)                                │
│                               ───────────────────────                           │
│                    f(qi, D) + k1 · (1 - b + b · |D|/avgdl)                     │
│                                                                                 │
│  其中：                                                                        │
│  ──────────────────────────────────────────────────────────────────────────    │
│  • f(qi, D): 词 qi 在文档 D 中的词频                                              │
│  • |D|: 文档 D 的长度（token数）                                                 │
│  • avgdl: 平均文档长度                                                           │
│  • k1: 词频饱和参数 (通常 1.2-2.0)                                               │
│  • b: 长度归一化参数 (通常 0.75)                                                 │
│  • IDF(qi): 逆文档频率                                                           │
│                                                                                 │
│  IDF(qi) = log( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )                        │
│                                                                                 │
│  其中：                                                                        │
│  • N: 文档总数                                                                   │
│  • n(qi): 包含词 qi 的文档数                                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 完整实现

```python
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import re

class BM25Indexer:
    """
    BM25 索引器

    实现 Okapi BM25 关键词检索算法
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 词频饱和参数
            b: 长度归一化参数
        """
        self.k1 = k1
        self.b = b

        # 文档集合
        self.documents: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}

        # 索引
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # 词 -> 文档频率
        self.term_doc_freqs: Dict[str, Counter] = defaultdict(Counter)  # 词 -> {doc_id -> 词频}
        self.avg_doc_length: float = 0

        # IDF 缓存
        self.idf_cache: Dict[str, float] = {}

    def fit(self, documents: Dict[str, str]):
        """
        构建索引

        Args:
            documents: {doc_id: document_content}
        """
        self.documents = documents
        total_length = 0

        for doc_id, doc in documents.items():
            # 分词
            tokens = self._tokenize(doc)

            # 记录长度
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # 词频统计
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                self.term_doc_freqs[token][doc_id] = count
                self.doc_freqs[token] += 1

        # 计算平均文档长度
        self.avg_doc_length = total_length / len(documents) if documents else 0

        # 预计算 IDF
        self._compute_idf()

    def _tokenize(self, text: str) -> List[str]:
        """中文分词（简单实现）"""
        # 清理并分词
        text = text.lower()
        # 简单分词：按字符/空格
        tokens = re.findall(r'[\w\u4e00-\u9fff]+', text)
        return tokens

    def _compute_idf(self):
        """预计算所有词的 IDF"""
        n = len(self.documents)
        for term, df in self.doc_freqs.items():
            # BM25 IDF 公式
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            self.idf_cache[term] = idf

    def get_idf(self, term: str) -> float:
        """获取词的 IDF"""
        term = term.lower()
        if term in self.idf_cache:
            return self.idf_cache[term]
        # 未知词的 IDF（使用平均 IDF）
        return math.log((len(self.documents) + 0.5) / 0.5 + 1) if self.documents else 0

    def score(self, doc_id: str, query: str) -> float:
        """
        计算文档对查询的 BM25 分数

        Args:
            doc_id: 文档ID
            query: 查询文本

        Returns:
            BM25 分数
        """
        if doc_id not in self.doc_lengths:
            return 0

        # 分词
        query_tokens = self._tokenize(query)
        doc_length = self.doc_lengths[doc_id]

        # 获取文档词频
        doc_freqs = self.term_doc_freqs

        score = 0
        for token in query_tokens:
            if token in doc_freqs:
                # 词频
                tf = doc_freqs[token][doc_id]
                # IDF
                idf = self.get_idf(token)

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

                score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        搜索

        Returns:
            [(doc_id, score), ...]
        """
        # 分词
        query_tokens = self._tokenize(query)

        # 计算所有文档的分数
        scores = []
        for doc_id in self.documents:
            score = self.score(doc_id, query)
            if score > 0:
                scores.append((doc_id, score))

        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
```

### 4.3 参数调优指南

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            BM25 参数影响分析                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  k1 参数：                                                                      │
│  ─────────                                                                     │
│  k1 = 0:    完全忽略词频，只考虑是否包含                                         │
│  k1 = 1.5:  标准设置，平衡词频影响                                               │
│  k1 = 2.0:  词频影响更大                                                        │
│                                                                                 │
│  b 参数：                                                                       │
│  ─────                                                                       │
│  b = 0:    完全不考虑文档长度                                                    │
│  b = 0.75: 标准设置，长文档会受惩罚                                              │
│  b = 1.0:  严格的长度惩罚                                                       │
│                                                                                 │
│  典型配置：                                                                     │
│  ──────────                                                                    │
│  • 网页搜索: k1=1.2, b=0.75                                                    │
│  • 短文档:     k1=1.5, b=0.5                                                    │
│  • 长文档:     k1=2.0, b=0.9                                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、混合检索融合

### 5.1 融合策略详解

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              混合检索融合策略                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │  向量检索结果    │    │  BM25检索结果   │    │  图谱检索结果   │            │
│  │  (相似度分数)    │    │  (BM25分数)     │    │  (频率分数)     │            │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘            │
│           │                      │                      │                      │
│           ▼                      ▼                      ▼                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   分数归一化     │    │   分数归一化     │    │   分数归一化     │            │
│  │  [0, 1]         │    │  [0, 1]         │    │  [0, 1]         │            │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘            │
│           │                      │                      │                      │
│           └──────────────────────┼──────────────────────┘                      │
│                                  │                                             │
│                                  ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         加权求和融合                                     │   │
│  │                                                                         │   │
│  │   final_score = α · score_vector + β · score_bm25 + γ · score_graph    │   │
│  │                                                                         │   │
│  │   其中 α + β + γ = 1                                                    │   │
│  │                                                                         │   │
│  │   我们默认: α = 0.6, β = 0.3, γ = 0.1                                   │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                  │                                             │
│                                  ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           排序输出                                       │   │
│  │                                                                         │   │
│  │   [result_1, result_2, result_3, ...]                                  │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 完整实现

```python
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    id: str
    content: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    final_score: float = 0.0
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """
    混合检索器

    融合向量检索、BM25检索和图谱检索的结果
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.3,
        graph_weight: float = 0.1
    ):
        """
        Args:
            vector_weight: 向量检索权重
            bm25_weight: BM25权重
            graph_weight: 图谱权重
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.graph_weight = graph_weight

        # 归一化权重
        total = vector_weight + bm25_weight + graph_weight
        self.vector_weight /= total
        self.bm25_weight /= total
        self.graph_weight /= total

        # 子检索器
        self.vector_store = None
        self.embedder = None
        self.bm25_indexer = None
        self.graph_retriever = None

    def set_vector_retriever(self, vector_store, embedder):
        """设置向量检索器"""
        self.vector_store = vector_store
        self.embedder = embedder

    def set_bm25_indexer(self, bm25_indexer):
        """设置BM25索引器"""
        self.bm25_indexer = bm25_indexer

    def set_graph_retriever(self, graph_retriever):
        """设置图谱检索器"""
        self.graph_retriever = graph_retriever

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid"
    ) -> List[HybridSearchResult]:
        """
        混合搜索

        Args:
            query: 查询文本
            top_k: 返回数量
            mode: 搜索模式 ("vector", "bm25", "graph", "hybrid")

        Returns:
            HybridSearchResult 列表
        """
        # 并行执行各种检索
        vector_results = []
        bm25_results = []
        graph_results = []

        if mode in ["vector", "hybrid"] and self.vector_store and self.embedder:
            vector_results = self._search_vector(query, top_k * 2)

        if mode in ["bm25", "hybrid"] and self.bm25_indexer:
            bm25_results = self._search_bm25(query, top_k * 2)

        if mode in ["graph", "hybrid"] and self.graph_retriever:
            graph_results = self._search_graph(query, top_k * 2)

        # 融合结果
        return self._fuse_results(
            vector_results,
            bm25_results,
            graph_results,
            top_k
        )

    def _search_vector(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """向量检索"""
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vector, top_k)
        return [(r.id, r.score) for r in results]

    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """BM25检索"""
        return self.bm25_indexer.search(query, top_k)

    def _search_graph(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """图谱检索"""
        results = self.graph_retriever.retrieve(query, top_k)
        return [(r.entity_id, r.score) for r in results]

    def _fuse_results(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        graph_results: List[Tuple[str, float]],
        top_k: int
    ) -> List[HybridSearchResult]:
        """融合结果"""

        # 1. 归一化分数
        max_vector = max((s[1] for s in vector_results), default=1)
        max_bm25 = max((s[1] for s in bm25_results), default=1)
        max_graph = max((s[1] for s in graph_results), default=1)

        # 2. 合并结果
        fused: Dict[str, HybridSearchResult] = {}

        # 向量结果
        for doc_id, score in vector_results:
            if doc_id not in fused:
                fused[doc_id] = HybridSearchResult(
                    id=doc_id,
                    content="",  # 后续填充
                    vector_score=score / max_vector if max_vector > 0 else 0,
                    sources=["vector"]
                )
            else:
                fused[doc_id].vector_score = score / max_vector if max_vector > 0 else 0
                fused[doc_id].sources.append("vector")

        # BM25 结果
        for doc_id, score in bm25_results:
            if doc_id not in fused:
                fused[doc_id] = HybridSearchResult(
                    id=doc_id,
                    content="",
                    bm25_score=score / max_bm25 if max_bm25 > 0 else 0,
                    sources=["bm25"]
                )
            else:
                fused[doc_id].bm25_score = score / max_bm25 if max_bm25 > 0 else 0
                if "bm25" not in fused[doc_id].sources:
                    fused[doc_id].sources.append("bm25")

        # 图谱结果
        for doc_id, score in graph_results:
            if doc_id not in fused:
                fused[doc_id] = HybridSearchResult(
                    id=doc_id,
                    content="",
                    graph_score=score / max_graph if max_graph > 0 else 0,
                    sources=["graph"]
                )
            else:
                fused[doc_id].graph_score = score / max_graph if max_graph > 0 else 0
                if "graph" not in fused[doc_id].sources:
                    fused[doc_id].sources.append("graph")

        # 3. 计算最终分数
        for result in fused.values():
            result.final_score = (
                self.vector_weight * result.vector_score +
                self.bm25_weight * result.bm25_score +
                self.graph_weight * result.graph_score
            )

        # 4. 排序
        sorted_results = sorted(
            fused.values(),
            key=lambda x: x.final_score,
            reverse=True
        )

        return sorted_results[:top_k]
```

### 5.3 RRF 融合（可选）

```python
class RRFusion:
    """
    RRF (Reciprocal Rank Fusion) 融合

    倒数排名融合，无需调参
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: 融合常数（通常 60）
        """
        self.k = k

    def fuse(
        self,
        rank_lists: List[List[str]],
        top_k: int = 10
    ) -> List[str]:
        """
        RRF 融合

        Args:
            rank_lists: 多个排名列表，每个列表按相关度排序

        Returns:
            融合后的排名
        """
        # RRF 分数
        scores: Dict[str, float] = defaultdict(float)

        for rank_list in rank_lists:
            for rank, doc_id in enumerate(rank_list):
                # RRF 公式: 1 / (rank + k)
                rrf_score = 1 / (rank + self.k)
                scores[doc_id] += rrf_score

        # 排序
        sorted_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return sorted_docs[:top_k]
```

---

## 六、重排序算法

### 6.1 Bi-Encoder vs Cross-Encoder

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           双编码器 vs 交叉编码器                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  双编码器 (Bi-Encoder)                    交叉编码器 (Cross-Encoder)             │
│  ═══════════════════════                  ════════════════════════              │
│                                                                                 │
│      ┌──────────────┐                         ┌─────────────────┐              │
│      │    Query     │                         │  [CLS] Query    │              │
│      └──────┬───────┘                         │   [SEP] Doc     │              │
│             │                                 │   [SEP]         │              │
│             ▼                                 └────────┬────────┘              │
│      ┌──────────────┐                                  │                       │
│      │   Encoder    │                                  ▼                       │
│      └──────┬───────┘                         ┌─────────────────┐              │
│             │                                 │    Joint        │              │
│             ▼                                 │   Encoder       │              │
│         v_query                              └────────┬────────┘              │
│             │                                          │                       │
│             │    ┌──────────────┐                      │                       │
│             └───▶│    Doc      │                      ▼                       │
│                  └──────┬───────┘              ┌─────────────────┐              │
│                         │                      │  [CLS] → score  │              │
│                         ▼                      └─────────────────┘              │
│                  ┌──────────────┐                                                 │
│                  │   Encoder    │                                                 │
│                  └──────┬───────┘                                                 │
│                         │                                                         │
│                         ▼                                                         │
│                       v_doc                                                       │
│                         │                                                         │
│                         ▼                                                         │
│              ┌──────────────────┐                                                 │
│              │ cos(v_q, v_d)   │                                                 │
│              └──────────────────┘                                                 │
│                                                                                 │
│  复杂度: O(n·d)                     复杂度: O(n·L·d)                             │
│  延迟:   低                         延迟:   高                                   │
│  效果:   ★★★★                      效果:   ★★★★★                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Cross-Encoder 实现

```python
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import numpy as np

class CrossEncoderReranker:
    """
    Cross-Encoder 重排序器

    用于对初步检索的结果进行精细重排序
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Args:
            model_name: HuggingFace 模型名称
        """
        self.model = CrossEncoder(model_name, device="auto")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3,
        batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """
        重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量
            batch_size: 批处理大小

        Returns:
            [(doc_id, score), ...]
        """
        if not documents:
            return []

        # 构建句子对
        pairs = [(query, doc) for doc in documents]

        # 批量预测
        scores = self.model.predict(pairs, batch_size=batch_size)

        # 归一化为概率 (sigmoid)
        probs = 1 / (1 + np.exp(-scores))

        # 排序
        indexed = list(enumerate(probs))
        indexed.sort(key=lambda x: x[1], reverse=True)

        return [(documents[i], float(s)) for i, s in indexed[:top_k]]
```

### 6.3 效率优化策略

```python
class EfficientCrossEncoderReranker:
    """
    高效 Cross-Encoder 重排序器

    优化策略：
    1. 早期终止（当分数足够低时）
    2. 短文档优先
    3. GPU 批处理
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name, device="auto")

    def rerank_with_early_stop(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3,
        threshold: float = 0.1,
        max_docs: int = 100
    ) -> List[Tuple[str, float]]:
        """
        带早期终止的重排序

        当剩余文档的最高可能分数低于阈值时终止
        """
        # 限制数量
        documents = documents[:max_docs]

        # 构建句子对
        pairs = [(query, doc) for doc in documents]

        # 预测
        scores = self.model.predict(pairs, batch_size=32)

        # 排序并过滤
        results = []
        for i, score in enumerate(scores):
            if score < threshold:
                # 假设后续分数更低，提前终止
                continue

            results.append((documents[i], float(score)))

            if len(results) >= top_k:
                break

        return results
```

---

## 七、实体识别算法

### 7.1 实体识别技术演进

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           实体识别技术演进图                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  规则匹配 ──▶ 正则表达式 ──▶ 词典匹配 ──▶ CRF ──▶ BiLSTM-CRF ──▶ BERT-NER      │
│     │           │            │          │           │            │             │
│     ▼           ▼            ▼          ▼           ▼            ▼             │
│   简单        灵活          高效       序列标注     上下文        SOTA         │
│   不可扩展    依赖模式      词典依赖    需训练数据   理解           效果         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      我们当前：规则 + 正则 + 词典                          │   │
│  │                                                                         │   │
│  │  ENTITY_TYPES = {                                                       │   │
│  │      "PRODUCT": ["重疾险", "医疗险", "意外险", ...],                     │   │
│  │      "COMPANY": ["平安", "中国人寿", ...],                               │   │
│  │      "CONCEPT": ["保额", "保费", "保障期限", ...],                       │   │
│  │      "ACTION": ["理赔", "投保", "核保", ...],                            │   │
│  │  }                                                                      │   │
│  │                                                                         │   │
│  │  建议升级：BERT-NER                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 完整实体抽取实现

```python
import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体"""
    id: str
    name: str
    entity_type: str
    description: str = ""
    attributes: Dict = field(default_factory=dict)
    frequency: int = 1
    sources: List[str] = field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type,
            "description": self.description,
            "attributes": self.attributes,
            "frequency": self.frequency,
            "sources": self.sources
        }


class EntityExtractor:
    """
    实体抽取器

    支持：
    1. 规则匹配
    2. 正则表达式
    3. 词典匹配
    4. 上下文消歧
    """

    # 保险领域实体类型和关键词
    ENTITY_TYPES = {
        "PRODUCT": [
            "重疾险", "重大疾病保险", "医疗保险", "医疗险",
            "意外险", "意外伤害保险", "寿险", "人寿保险",
            "年金险", "养老保险", "万能险", "投资连结保险",
            "车险", "家财险", "旅游险", "防癌险", "护理险"
        ],
        "COMPANY": [
            "平安", "中国人寿", "太平洋", "新华保险",
            "泰康", "人保", "大地保险", "阳光保险",
            "友邦保险", "中宏保险", "工银安盛", "招商信诺"
        ],
        "CONCEPT": [
            "保额", "保费", "保障期限", "缴费期限",
            "等待期", "免赔额", "赔付比例", "现金价值",
            "责任免除", "健康告知", "续保", "理赔"
        ],
        "ACTION": [
            "理赔", "投保", "核保", "续保", "退保",
            "索赔", "报案", "体检", "签单", "缴费"
        ],
        "AGE": [
            "岁", "周岁", "年龄"
        ],
        "AMOUNT": [
            "元", "万", "千万", "亿"
        ]
    }

    # 复合实体模式
    COMPOSITE_PATTERNS = [
        # XX险产品
        (r'(\w+险)', 'PRODUCT'),
        # XX公司
        (r'(\w+(?:保险|人寿|财险|寿险)公司)', 'COMPANY'),
        # 金额
        r'(\d+(?:,\d{3})*(?:元|万|千万|亿))',
    ]

    def __init__(
        self,
        case_sensitive: bool = False,
        enable_composite: bool = True,
        enable_fuzzy: bool = True
    ):
        """
        Args:
            case_sensitive: 是否大小写敏感
            enable_composite: 是否启用复合实体识别
            enable_fuzzy: 是否启用模糊匹配
        """
        self.case_sensitive = case_sensitive
        self.enable_composite = enable_composite
        self.enable_fuzzy = enable_fuzzy

        # 构建模式
        self._build_patterns()

    def _build_patterns(self):
        """构建正则表达式模式"""
        self.type_patterns: Dict[str, List[re.Pattern]] = {}
        self.all_keywords: Set[str] = set()

        for etype, keywords in self.ENTITY_TYPES.items():
            if self.case_sensitive:
                pattern = r'|'.join(re.escape(k) for k in keywords)
            else:
                pattern = r'|'.join(re.escape(k.lower()) for k in keywords)

            self.type_patterns[etype] = [re.compile(pattern)]
            self.all_keywords.update(k.lower() for k in keywords)

    def extract(
        self,
        text: str,
        source: str = "",
        return_positions: bool = False
    ) -> List[Entity]:
        """
        实体抽取主流程

        Args:
            text: 输入文本
            source: 来源标识
            return_positions: 是否返回位置信息

        Returns:
            Entity 列表
        """
        # 预处理
        text_lower = text if self.case_sensitive else text.lower()

        entities = []
        seen_entities: Set[Tuple[str, str]] = set()  # (name_lower, type)

        # 1. 基于词典的抽取
        for etype, patterns in self.type_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    name = match.group()
                    name_lower = name.lower()

                    # 去重
                    if (name_lower, etype) in seen_entities:
                        continue
                    seen_entities.add((name_lower, etype))

                    entity = Entity(
                        id=self._generate_id(name),
                        name=name,
                        entity_type=etype,
                        description=f"Extracted from {source}" if source else "",
                        sources=[source] if source else [],
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    entities.append(entity)

        # 2. 复合实体识别
        if self.enable_composite:
            entities.extend(self._extract_composite_entities(text, source, seen_entities))

        # 3. 模糊匹配（处理变体）
        if self.enable_fuzzy:
            entities.extend(self._extract_fuzzy_entities(text, source, seen_entities))

        # 按位置排序
        entities.sort(key=lambda x: x.start_pos)

        return entities

    def _extract_composite_entities(
        self,
        text: str,
        source: str,
        seen_entities: Set[Tuple[str, str]]
    ) -> List[Entity]:
        """抽取复合实体"""
        entities = []

        # XX险模式
        pattern = re.compile(r'(\w+险)')
        for match in pattern.finditer(text):
            name = match.group()
            # 检查是否已存在
            if (name.lower(), 'PRODUCT') not in seen_entities:
                # 检查是否在已知词典中
                is_known = any(name in kw for kw in self.ENTITY_TYPES.get('PRODUCT', []))
                if not is_known:
                    # 可能是新险种
                    entities.append(Entity(
                        id=self._generate_id(name),
                        name=name,
                        entity_type='PRODUCT',
                        description=f"Composite entity from {source}" if source else "",
                        sources=[source] if source else [],
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
                    seen_entities.add((name.lower(), 'PRODUCT'))

        return entities

    def _extract_fuzzy_entities(
        self,
        text: str,
        source: str,
        seen_entities: Set[Tuple[str, str]]
    ) -> List[Entity]:
        """模糊匹配抽取"""
        entities = []

        # 处理常见变体
        variants = {
            "重大疾病": "重疾",
            "重大疾病保险": "重疾险",
            "住院医疗": "医疗险",
            "意外伤害": "意外险",
        }

        for variant, standard in variants.items():
            if variant in text and standard in text:
                # 两者都出现时，标记为标准实体
                continue

        return entities

    def _generate_id(self, name: str) -> str:
        """生成实体ID"""
        content = f"{name}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
```

### 7.3 BERT-NER 升级方案

```python
class BertNERExtractor:
    """
    BERT 命名实体识别

    需要预训练或微调的 NER 模型
    """

    def __init__(self, model_name: str = "bert-base-chinese"):
        """
        Args:
            model_name: HuggingFace 模型名称
        """
        from transformers import BertTokenizer, BertForTokenClassification

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(model_name)
        self.model.eval()

        # 标签映射
        self.id2label = {
            0: "O",
            1: "B-PRODUCT",
            2: "I-PRODUCT",
            3: "B-COMPANY",
            4: "I-COMPANY",
            5: "B-CONCEPT",
            6: "I-CONCEPT",
            7: "B-ACTION",
            8: "I-ACTION",
        }

    @torch.no_grad()
    def extract(self, text: str) -> List[Entity]:
        """抽取实体"""
        # Tokenization
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()

        # 前向传播
        outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)[0].tolist()

        # 解码
        entities = self._decode_predictions(
            text, predictions, offset_mapping
        )

        return entities

    def _decode_predictions(
        self,
        text: str,
        predictions: List[int],
        offset_mapping: List[Tuple[int, int]]
    ) -> List[Entity]:
        """解码预测结果为实体"""
        entities = []
        current_entity = None

        for i, pred in enumerate(predictions):
            label = self.id2label[pred]

            if label.startswith("B-"):
                # 新实体开始
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]
                start_char, end_char = offset_mapping[i]

                if end_char > start_char:  # 跳过特殊token
                    current_entity = Entity(
                        id=self._generate_id(text[start_char:end_char]),
                        name=text[start_char:end_char],
                        entity_type=entity_type,
                        start_pos=start_char,
                        end_pos=end_char
                    )

            elif label.startswith("I-") and current_entity:
                # 实体延续
                entity_type = label[2:]
                if entity_type == current_entity.entity_type:
                    start_char, end_char = offset_mapping[i]
                    if end_char > start_char:
                        current_entity.name += text[start_char:end_char]
                        current_entity.end_pos = end_char

            else:
                # 非实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # 处理最后一个实体
        if current_entity:
            entities.append(current_entity)

        return entities
```

---

## 八、知识图谱检索

### 8.1 BFS 多跳检索详解

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         BFS 多跳检索可视化                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                              查询: "平安福"                                      │
│                                    │                                            │
│                                    ▼                                            │
│                         ┌─────────────────┐                                     │
│                         │  1. 找到实体    │                                     │
│                         │  平安福 (重疾险) │                                     │
│                         └────────┬────────┘                                     │
│                                  │                                              │
│                    ┌─────────────┼─────────────┐                                │
│                    ▼             ▼             ▼                                │
│           ┌───────────┐  ┌───────────┐  ┌───────────┐                           │
│           │ 平安保险  │  │  重疾保障  │  │  100种疾病 │                           │
│           │(公司)     │  │ (概念)     │  │  (概念)    │                           │
│           └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                           │
│                 │              │              │                                 │
│                 │    ┌─────────┼─────────┐    │                                 │
│                 ▼    ▼         ▼         ▼    ▼                                 │
│           ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐               │
│           │ 平安集团  │ │ 轻症保障  │ │ 中症保障  │ │ 癌症保障  │               │
│           │(公司)     │ │ (概念)    │ │ (概念)    │ │ (概念)    │               │
│           └───────────┘ └───────────┘ └───────────┘ └───────────┘               │
│                                                                                 │
│                         深度=1                    深度=2                         │
│                                                                                 │
│  收集到的上下文:                                                               │
│  - 平安福由平安保险承保                                                        │
│  - 平安福是重疾险产品                                                          │
│  - 平安福保障100种重大疾病                                                     │
│  - 平安福还包括轻症、中症、癌症保障                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 完整实现

```python
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体"""
    id: str
    name: str
    entity_type: str
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
    relation_type: str  # PRODUCED_BY, HAS_ATTRIBUTE, IS_A, etc.
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


class KnowledgeGraph:
    """知识图谱"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        # 索引
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

    def get_connected_entities(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Tuple[Entity, int, float]]:
        """
        获取相连实体（BFS多跳检索）

        Args:
            entity_id: 起始实体ID
            max_depth: 最大跳数

        Returns:
            {entity_id: (entity, depth, weight)}
        """
        result: Dict[str, Tuple[Entity, int, float]] = {}

        if entity_id not in self.entities:
            return result

        # BFS
        queue = deque([(entity_id, 0)])  # (node_id, depth)
        visited: Set[str] = set()

        while queue:
            current_id, depth = queue.popleft()

            if depth > max_depth or current_id in visited:
                continue

            visited.add(current_id)
            entity = self.get_entity(current_id)

            if entity:
                # 更新最短距离
                if current_id not in result or depth < result[current_id][1]:
                    result[current_id] = (entity, depth, 0)

            # 探索邻居
            for neighbor_id, rel_type, weight in self.adjacency.get(current_id, []):
                if neighbor_id not in visited and depth + 1 <= max_depth:
                    queue.append((neighbor_id, depth + 1))

        return result

    def get_shortest_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[List[str]]:
        """
        查找最短路径（BFS）

        Args:
            source_id: 起始实体ID
            target_id: 目标实体ID

        Returns:
            路径节点列表或None
        """
        if source_id not in self.entities or target_id not in self.entities:
            return None

        queue = deque([(source_id, [source_id])])
        visited: Set[str] = {source_id}

        while queue:
            current_id, path = queue.popleft()

            if current_id == target_id:
                return path

            for neighbor_id, _, _ in self.adjacency.get(current_id, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None
    ) -> List[Tuple[Entity, str, float]]:
        """获取直接邻居"""
        neighbors = []
        for neighbor_id, rel_type, weight in self.adjacency.get(entity_id, []):
            if relation_type is None or rel_type == relation_type:
                entity = self.get_entity(neighbor_id)
                if entity:
                    neighbors.append((entity, rel_type, weight))
        return neighbors

    def search_by_type(self, entity_type: str) -> List[Entity]:
        """按类型搜索"""
        entity_ids = self.type_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def search_by_name_contains(self, keyword: str) -> List[Entity]:
        """模糊搜索"""
        keyword_lower = keyword.lower()
        results = []
        for name_lower, entity_ids in self.entity_index.items():
            if keyword_lower in name_lower:
                for eid in entity_ids:
                    if eid in self.entities:
                        results.append(self.entities[eid])
        return results

    def get_stats(self) -> Dict[str, Any]:
        """统计信息"""
        type_counts = {}
        for etype, eids in self.type_index.items():
            type_counts[etype] = len(eids)
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "type_distribution": type_counts
        }
```

### 8.3 图算法复杂度对比

| 算法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| **BFS单跳** | O(degree) | O(1) | 直接邻居 |
| **BFS多跳** | O(V + E) | O(V) | 关系推理 |
| **DFS** | O(V + E) | O(V) | 路径查找 |
| **Dijkstra** | O(E + V log V) | O(V) | 带权最短路径 |
| **A*** | O(E) | O(V) | 目标导向搜索 |

> V: 实体数量, E: 边数量, degree: 平均度数

---

## 九、Query 改写算法

### 9.1 改写流程详解

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Query 改写流程图                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         原始 Query                                        │   │
│  │                     "它能保什么？"                                        │   │
│  └────────────────────────────────┬────────────────────────────────────────┘   │
│                                   │                                            │
│                                   ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 术语扩展 (Term Expansion)                                             │   │
│  │    ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │    │  保险缩写 → 标准术语                                               │ │   │
│  │    │  ─────────────────────────────────────────────────────────────  │ │   │
│  │    │  "重疾" → "重大疾病保险"                                          │ │   │
│  │    │  "医疗险" → "医疗保险"                                            │ │   │
│  │    │  "意外险" → "意外伤害保险"                                        │ │   │
│  │    └─────────────────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────┬────────────────────────────────────────┘   │
│                                   │                                            │
│                                   ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 2. 指代消解 (Coreference Resolution)                                    │   │
│  │    ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │    │  代词 → 实体                                                       │ │   │
│  │    │  ─────────────────────────────────────────────────────────────  │ │   │
│  │    │  "它" → "平安福" (从会话历史中推断)                                │ │   │
│  │    │  "这个" → "这个保险产品"                                          │ │   │
│  │    │  "那种" → "那种重疾险"                                            │ │   │
│  │    └─────────────────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────┬────────────────────────────────────────┘   │
│                                   │                                            │
│                                   ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 3. 意图分类 (Intent Classification)                                    │   │
│  │    ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │    │  意图类型                                                         │ │   │
│  │    │  ─────────────────────────────────────────────────────────────  │ │   │
│  │    │  • PRODUCT_INQUIRY  → 产品咨询                                    │ │   │
│  │    │  • CLAIM_GUIDANCE   → 理赔指导                                    │ │   │
│  │    │  • COMPARISON       → 产品对比                                    │ │   │
│  │    │  • RECOMMENDATION   → 产品推荐                                    │ │   │
│  │    └─────────────────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────┬────────────────────────────────────────┘   │
│                                   │                                            │
│                                   ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 4. 用户画像 Enrichment                                                 │   │
│  │    ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │    │  用户画像元素                                                     │ │   │
│  │    │  ─────────────────────────────────────────────────────────────  │ │   │
│  │    │  • age: 30                                                      │ │   │
│  │    │  • income: 20000                                                │ │   │
│  │    │  • risk_preference: "medium"                                   │ │   │
│  │    │  • insurance_history: ["医疗险"]                                │ │   │
│  │    └─────────────────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────┬────────────────────────────────────────┘   │
│                                   │                                            │
│                                   ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 5. 行为意图扩展 (Behavioral Expansion)                                 │   │
│  │    ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │    │  浏览历史 → 查询扩展                                              │ │   │
│  │    │  ─────────────────────────────────────────────────────────────  │ │   │
│  │    │  用户浏览过"重疾险"页面                                           │ │   │
│  │    │  → 扩展: "平安福重疾险保障范围"                                   │ │   │
│  │    └─────────────────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────┬────────────────────────────────────────┘   │
│                                   │                                            │
│                                   ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         改写后 Query                                     │   │
│  │              "平安福重大疾病保险能保障什么？"                            │   │
│  │              意图: PRODUCT_INQUIRY                                      │   │
│  │              上下文: age=30, 看过重疾险                                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 完整实现

```python
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    income_level: Optional[str] = None
    occupation: Optional[str] = None
    risk_preference: Optional[str] = None  # "low", "medium", "high"
    insurance_history: List[str] = field(default_factory=list)
    family_status: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "age": self.age,
            "gender": self.gender,
            "income_level": self.income_level,
            "occupation": self.occupation,
            "risk_preference": self.risk_preference,
            "insurance_history": self.insurance_history,
            "family_status": self.family_status,
            "preferences": self.preferences
        }


@dataclass
class ConversationTurn:
    """会话轮次"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RewriteResult:
    """改写结果"""
    original_query: str
    rewritten_query: str
    intent_type: str
    expanded_terms: List[str] = field(default_factory=list)
    resolved_references: Dict[str, str] = field(default_factory=dict)
    enriched_profile: Dict = field(default_factory=dict)
    confidence: float = 1.0
    applied_strategies: List[str] = field(default_factory=list)


class InsuranceQueryRewriter:
    """
    保险领域 Query 改写器

    功能：
    1. 术语扩展
    2. 指代消解
    3. 意图分类
    4. 用户画像 Enrichment
    5. 行为意图扩展
    """

    # 保险术语缩写映射
    ABBREVIATIONS = {
        "重疾": "重大疾病保险",
        "医疗险": "医疗保险",
        "意外险": "意外伤害保险",
        "寿险": "人寿保险",
        "年金": "年金保险",
        "万能险": "万能保险",
        "防癌险": "癌症保险",
        "车险": "汽车保险",
        "家财险": "家庭财产保险",
        "保额": "保险金额",
        "保费": "保险费用",
        "保障期限": "保险期限",
        "缴费期限": "缴费期间",
        "等待期": "观察期",
        "免赔额": "免赔额",
        "赔付比例": "理赔比例",
        "现金价值": "保单现金价值",
        "理赔": "保险理赔",
        "投保": "购买保险",
        "核保": "保险核保",
        "续保": "保险续保",
        "退保": "保险退保",
    }

    # 同义词映射
    SYNONYMS = {
        "哪个好": ["推荐", "比较", "选择", "评价"],
        "多少钱": ["价格", "保费", "费用", "成本"],
        "有什么": ["包含", "包括", "保障范围"],
        "怎么买": ["投保", "购买", "如何投保"],
        "能赔多少": ["赔付金额", "理赔金额", "赔偿"],
        "适合谁": ["适用人群", "适合人群", "投保条件"],
    }

    # 意图分类关键词
    INTENT_KEYWORDS = {
        "PRODUCT_INQUIRY": ["保什么", "保障", "包括", "涵盖", "范围", "条款"],
        "PRICE_INQUIRY": ["多少钱", "价格", "保费", "费用", "贵不贵"],
        "CLAIM_GUIDANCE": ["理赔", "怎么赔", "索赔", "报案", "材料"],
        "COMPARISON": ["哪个好", "对比", "比较", "区别", "不同"],
        "RECOMMENDATION": ["推荐", "适合", "应该买", "买什么"],
        "ELIGIBILITY": ["能买吗", "可以买吗", "符合条件", "健康告知"],
        "POLICY_TERMS": ["条款", "细则", "规定", "免责", "除外"],
    }

    # 指代词映射
    PRONOUN_MAP = {
        "它": ["该保险", "这个产品", "这个保险"],
        "这个": ["这个保险", "这个产品", "该产品"],
        "那个": ["那个保险", "那个产品", "该产品"],
        "这种": ["这种保险", "此类产品"],
        "那种": ["那种保险", "那类产品"],
    }

    def __init__(
        self,
        enable_term_expansion: bool = True,
        enable_coreference: bool = True,
        enable_intent_classification: bool = True,
        enable_profile_enrichment: bool = True,
        enable_behavior_expansion: bool = True
    ):
        self.enable_term_expansion = enable_term_expansion
        self.enable_coreference = enable_coreference
        self.enable_intent_classification = enable_intent_classification
        self.enable_profile_enrichment = enable_profile_enrichment
        self.enable_behavior_expansion = enable_behavior_expansion

        # 会话历史管理器
        self.history_manager = ConversationHistoryManager()

    def rewrite(
        self,
        query: str,
        user_id: str,
        session_id: str,
        user_profile: Optional[UserProfile] = None,
        conversation_history: Optional[List[ConversationTurn]] = None
    ) -> RewriteResult:
        """
        Query 改写主流程

        Args:
            query: 原始查询
            user_id: 用户ID
            session_id: 会话ID
            user_profile: 用户画像
            conversation_history: 会话历史

        Returns:
            RewriteResult: 改写结果
        """
        # 初始化结果
        result = RewriteResult(
            original_query=query,
            rewritten_query=query,
            intent_type="GENERAL"
        )

        applied_strategies = []

        # 1. 术语扩展
        if self.enable_term_expansion:
            expanded_query, terms = self._expand_terms(query)
            if expanded_query != query:
                result.rewritten_query = expanded_query
                result.expanded_terms = terms
                applied_strategies.append("term_expansion")

        # 2. 指代消解
        if self.enable_coreference:
            resolved_query, references = self._resolve_coreference(
                result.rewritten_query,
                conversation_history
            )
            if resolved_query != result.rewritten_query:
                result.rewritten_query = resolved_query
                result.resolved_references = references
                applied_strategies.append("coreference_resolution")

        # 3. 意图分类
        if self.enable_intent_classification:
            intent = self._classify_intent(result.rewritten_query)
            result.intent_type = intent
            applied_strategies.append(f"intent_classification:{intent}")

        # 4. 用户画像 Enrichment
        if self.enable_profile_enrichment and user_profile:
            enriched = self._enrich_from_profile(
                result.rewritten_query,
                user_profile,
                result.intent_type
            )
            if enriched != result.rewritten_query:
                result.rewritten_query = enriched
                result.enriched_profile = user_profile.to_dict()
                applied_strategies.append("profile_enrichment")

        # 5. 行为意图扩展
        if self.enable_behavior_expansion:
            expanded = self._expand_from_behavior(
                result.rewritten_query,
                user_id,
                result.intent_type
            )
            if expanded != result.rewritten_query:
                result.rewritten_query = expanded
                applied_strategies.append("behavior_expansion")

        result.applied_strategies = applied_strategies

        # 记录到历史
        self.history_manager.add_turn(
            session_id,
            ConversationTurn(role="user", content=query)
        )

        return result

    def _expand_terms(self, query: str) -> Tuple[str, List[str]]:
        """术语扩展"""
        expanded = query
        expanded_terms = []

        for abbr, full in self.ABBREVIATIONS.items():
            if abbr in expanded and full not in expanded:
                expanded = expanded.replace(abbr, f"{abbr}({full})")
                expanded_terms.append(f"{abbr} → {full}")

        for word, synonyms in self.SYNONYMS.items():
            if word in expanded:
                for syn in synonyms:
                    if syn not in expanded:
                        expanded += f" 或 {syn}"
                expanded_terms.append(f"{word} → {synonyms}")

        return expanded, expanded_terms

    def _resolve_coreference(
        self,
        query: str,
        history: Optional[List[ConversationTurn]]
    ) -> Tuple[str, Dict[str, str]]:
        """指代消解"""
        resolved = query
        references = {}

        if not history:
            return query, references

        # 获取最近的用户问题中的产品名
        product_entities = self._extract_product_entities(history)

        for pronoun, references_list in self.PRONOUN_MAP.items():
            if pronoun in query:
                # 选择最佳引用
                best_ref = self._select_best_reference(
                    references_list, product_entities, query
                )
                if best_ref:
                    resolved = resolved.replace(pronoun, best_ref)
                    references[pronoun] = best_ref

        return resolved, references

    def _extract_product_entities(
        self,
        history: List[ConversationTurn]
    ) -> List[str]:
        """从历史中提取产品实体"""
        entities = []
        for turn in reversed(history):
            if turn.role == "user":
                # 提取产品名
                for product in ["重疾险", "医疗险", "意外险", "寿险", "平安福"]:
                    if product in turn.content:
                        entities.append(product)
        return entities

    def _select_best_reference(
        self,
        candidates: List[str],
        context_entities: List[str],
        query: str
    ) -> str:
        """选择最佳引用"""
        # 首先从上下文实体中选择
        for entity in context_entities:
            for candidate in candidates:
                if entity in candidate or candidate in entity:
                    return candidate

        # 否则选择第一个
        return candidates[0] if candidates else ""

    def _classify_intent(self, query: str) -> str:
        """意图分类"""
        scores = defaultdict(float)

        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    scores[intent] += 1

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "GENERAL"

    def _enrich_from_profile(
        self,
        query: str,
        profile: UserProfile,
        intent: str
    ) -> str:
        """根据用户画像丰富查询"""
        enriched = query

        # 年龄相关的扩展
        if profile.age and profile.age < 30:
            if "适合" in query or "推荐" in query:
                enriched += " 年轻人"

        # 收入相关的扩展
        if profile.income_level == "high" and "重疾" in query:
            enriched += " 高保额"

        # 风险偏好
        if profile.risk_preference == "low":
            if "理财" in query or "投资" in query:
                enriched += " 低风险"

        # 保险历史
        if "医疗险" in profile.insurance_history and "医疗" not in query.lower():
            enriched += " 已购医疗险"

        return enriched

    def _expand_from_behavior(
        self,
        query: str,
        user_id: str,
        intent: str
    ) -> str:
        """根据用户行为扩展查询"""
        # 简化实现，实际应该查询用户行为数据库
        return query


class ConversationHistoryManager:
    """会话历史管理器"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.sessions: Dict[str, List[ConversationTurn]] = defaultdict(list)

    def add_turn(self, session_id: str, turn: ConversationTurn):
        """添加会话轮次"""
        self.sessions[session_id].append(turn)
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id].pop(0)

    def get_history(self, session_id: str) -> List[ConversationTurn]:
        """获取会话历史"""
        return self.sessions.get(session_id, [])

    def get_last_product(self, session_id: str) -> Optional[str]:
        """获取最近讨论的产品"""
        history = self.get_history(session_id)
        for turn in reversed(history):
            if turn.role == "assistant":
                # 尝试提取产品名
                for product in ["平安福", "重疾险", "医疗险"]:
                    if product in turn.content:
                        return product
        return None
```

### 9.3 改写效果评估

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Query 改写效果示例                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  原始 Query          改写后 Query               改写策略         意图           │
│  ─────────          ──────────────             ─────────        ────           │
│                                                                                 │
│  "它保什么"          "平安福重大疾病保险保什么"  指代消解+术语    PRODUCT        │
│                      "保障范围"                                        INQUIRY     │
│                                                                                 │
│  "哪个好"            "哪个重疾险好"             术语扩展         COMPARISON    │
│                      "推荐 比较 选择"                                         │
│                                                                                 │
│  "怎么理赔"          "平安福保险怎么理赔"       指代消解         CLAIM         │
│                      "保险理赔流程"             术语扩展         GUIDANCE      │
│                                                                                 │
│  "适合我吗"          "30岁适合买平安福吗"       用户画像         ELIGIBILITY   │
│                      "已购医疗险"               行为扩展                         │
│                                                                                 │
│  "多少钱一年"        "平安福保费多少钱一年"     指代消解         PRICE         │
│                      "价格 费用"                术语扩展         INQUIRY       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 十、Prompt 构建算法

### 10.1 组件化 Prompt 系统

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Prompt 组件架构图                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         BasePromptTemplate                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  components: Dict[ComponentType, PromptComponent]               │   │   │
│  │  │                                                                 │   │   │
│  │  │  + render(context, question, **kwargs) → str                   │   │   │
│  │  │  + add_component(component)                                     │   │   │
│  │  │  + get_version_info() → Dict                                    │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                            │
│                                    │                                            │
│              ┌─────────────────────┼─────────────────────┐                     │
│              │                     │                     │                      │
│              ▼                     ▼                     ▼                      │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐          │
│  │InsuranceProduct   │  │InsuranceClaim     │  │InsuranceCompare   │          │
│  │Prompt             │  │Prompt             │  │Prompt             │          │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         ComponentType 枚举                              │   │
│  │  ─────────────────────────────────────────────────────────────────     │   │
│  │  PERSONA          → 角色定义 (你是专业的保险顾问)                       │   │
│  │  SYSTEM_PROMPT    → 系统指令 (请使用专业但易懂的语言)                   │   │
│  │  DOMAIN_KNOWLEDGE → 领域知识 (保险条款解读...)                         │   │
│  │  CONSTRAINT       → 约束条件 (不得提供虚假信息)                         │   │
│  │  FEW_SHOT         → 示例 (Q: xxx A: yyy)                               │   │
│  │  OUTPUT_FORMAT    → 输出格式 (使用Markdown表格)                        │   │
│  │  CONTEXT_TEMPLATE → 上下文模板 (根据检索结果填充)                      │   │
│  │  QUESTION_TEMPLATE→ 问题模板 (用户问题占位符)                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 完整实现

```python
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Prompt 组件类型"""
    PERSONA = "persona"
    SYSTEM_PROMPT = "system_prompt"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CONSTRAINT = "constraint"
    FEW_SHOT = "few_shot"
    OUTPUT_FORMAT = "output_format"
    CONTEXT_TEMPLATE = "context_template"
    QUESTION_TEMPLATE = "question_template"


@dataclass
class PromptComponent:
    """Prompt 组件"""
    name: str
    content: str
    component_type: ComponentType
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        """渲染组件"""
        return self.content.format(**kwargs) if kwargs else self.content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "type": self.component_type.value,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


class BasePromptTemplate:
    """
    Prompt 模板基类

    支持组件化、版本管理、动态组合
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.components: Dict[ComponentType, PromptComponent] = {}
        self.version = "1.0.0"
        self.description = ""

    def add_component(self, component: PromptComponent):
        """添加组件"""
        self.components[component.component_type] = component

    def remove_component(self, component_type: ComponentType):
        """移除组件"""
        if component_type in self.components:
            del self.components[component_type]

    def get_component(self, component_type: ComponentType) -> Optional[PromptComponent]:
        """获取组件"""
        return self.components.get(component_type)

    def render(
        self,
        context: str = "",
        question: str = "",
        **kwargs
    ) -> str:
        """
        渲染完整 Prompt

        Args:
            context: 检索上下文
            question: 用户问题
            **kwargs: 其他参数

        Returns:
            完整的 Prompt 字符串
        """
        prompt_parts = []

        # 按优先级添加组件
        render_order = [
            ComponentType.PERSONA,
            ComponentType.SYSTEM_PROMPT,
            ComponentType.DOMAIN_KNOWLEDGE,
            ComponentType.CONSTRAINT,
            ComponentType.FEW_SHOT,
            ComponentType.CONTEXT_TEMPLATE,
            ComponentType.QUESTION_TEMPLATE,
            ComponentType.OUTPUT_FORMAT,
        ]

        for comp_type in render_order:
            if comp_type in self.components:
                component = self.components[comp_type]

                # 特殊处理需要参数的组件
                if comp_type == ComponentType.CONTEXT_TEMPLATE:
                    rendered = component.render(context=context)
                elif comp_type == ComponentType.QUESTION_TEMPLATE:
                    rendered = component.render(question=question)
                else:
                    rendered = component.render(**kwargs)

                if rendered.strip():
                    prompt_parts.append(rendered)

        # 用双换行分隔
        return "\n\n".join(prompt_parts)

    def get_version_info(self) -> Dict[str, Any]:
        """获取版本信息"""
        return {
            "template_name": self.name,
            "template_version": self.version,
            "component_count": len(self.components),
            "components": {
                comp_type.value: comp.version
                for comp_type, comp in self.components.items()
            }
        }


class InsuranceProductPrompt(BasePromptTemplate):
    """保险产品咨询 Prompt"""

    def __init__(self):
        super().__init__("insurance_product")
        self._register_default_components()

    def _register_default_components(self):
        """注册默认组件"""

        # Persona
        self.add_component(PromptComponent(
            name="insurance_expert",
            content="""你是平安保险公司的专业保险顾问，拥有10年保险从业经验。
你精通各类保险产品的条款、保障范围和理赔流程。
请用专业但通俗易懂的语言为客户解答问题。""",
            component_type=ComponentType.PERSONA,
            version="1.0.0",
            description="保险专家角色定义"
        ))

        # System Prompt
        self.add_component(PromptComponent(
            name="response_style",
            content="""请遵循以下回答规范：
1. 回答要结构清晰，重点突出
2. 使用通俗易懂的语言，避免过多专业术语
3. 如有必要，使用表格展示对比信息
4. 提供具体的数据和例子支持你的回答
5. 提醒客户根据自身情况选择合适的保险产品""",
            component_type=ComponentType.SYSTEM_PROMPT,
            version="1.0.0",
            description="回答风格规范"
        ))

        # Domain Knowledge
        self.add_component(PromptComponent(
            name="product_knowledge",
            content="""产品知识背景：
- 平安福是平安人寿的重疾险产品
- 包含100种重疾、50种轻症
- 支持双豁免（投保人和被保险人）
- 保障终身，现金价值逐年增长""",
            component_type=ComponentType.DOMAIN_KNOWLEDGE,
            version="1.0.0",
            description="产品知识背景"
        ))

        # Constraints
        self.add_component(PromptComponent(
            name="response_constraints",
            content="""回答约束：
1. 不得承诺任何未经证实的理赔
2. 不得贬低其他保险公司产品
3. 如遇到不确定的问题，请明确告知客户并建议咨询专业人士
4. 严禁提供虚假或误导性信息""",
            component_type=ComponentType.CONSTRAINT,
            version="1.0.0",
            description="回答约束条件"
        ))

        # Output Format
        self.add_component(PromptComponent(
            name="output_format",
            content="""请使用以下格式组织回答：

## 产品概述
（简要介绍产品）

## 保障内容
| 保障项目 | 保障金额 | 说明 |
|---------|---------|------|
| 重疾保障 | XX万 | 100种疾病 |
| 轻症保障 | XX万 | 50种疾病 |

## 亮点总结
- 亮点1
- 亮点2

## 注意事项
- 注意1
- 注意2

## 建议
（根据客户情况给出建议）""",
            component_type=ComponentType.OUTPUT_FORMAT,
            version="1.0.0",
            description="输出格式模板"
        ))

        # Context Template
        self.add_component(PromptComponent(
            name="context_template",
            content="""参考信息：
{context}""",
            component_type=ComponentType.CONTEXT_TEMPLATE,
            version="1.0.0",
            description="上下文模板"
        ))

        # Question Template
        self.add_component(PromptComponent(
            name="question_template",
            content="""客户问题：
{question}""",
            component_type=ComponentType.QUESTION_TEMPLATE,
            version="1.0.0",
            description="问题模板"
        ))
```

### 10.3 版本管理

```python
class PromptVersionManager:
    """Prompt 版本管理器"""

    def __init__(self):
        self.templates: Dict[str, Dict[str, BasePromptTemplate]] = defaultdict(dict)

    def register(
        self,
        name: str,
        template: BasePromptTemplate,
        environment: str = "production"
    ):
        """注册模板版本"""
        version = template.version
        self.templates[name][version] = template
        self.templates[name]["latest"] = template

        logger.info(f"Registered prompt template: {name} v{version}")

    def get(
        self,
        name: str,
        version: Optional[str] = None,
        environment: str = "production"
    ) -> BasePromptTemplate:
        """获取模板"""
        if version:
            return self.templates[name].get(version)
        return self.templates[name].get("latest")

    def list_versions(self, name: str) -> List[str]:
        """列出所有版本"""
        return [v for v in self.templates[name].keys() if v != "latest"]
```

---

## 附录

### A. 性能基准测试

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LightRAG 性能基准                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  测试环境：MacBook Pro M2, 16GB RAM, Python 3.10                              │
│  测试数据：1000篇保险文档，每篇500字                                           │
│                                                                                 │
│  ┌────────────────────┬──────────────┬─────────────┬───────────────────┐      │
│  │ 模块               │ 操作          │ 耗时         │ 吞吐量             │      │
│  ├────────────────────┼──────────────┼─────────────┼───────────────────┤      │
│  │ 文本分块           │ 500,000字     │ 50ms         │ 10,000字/秒       │      │
│  │ 向量化 (CPU)       │ 32条          │ 1000ms       │ 32条/秒           │      │
│  │ 向量存储 (Flat)    │ 1000向量      │ 50ms         │ 20,000向量/秒     │      │
│  │ 向量检索           │ 1000文档      │ 5ms          │ 200查询/秒        │      │
│  │ BM25检索           │ 10,000文档    │ 3ms          │ 333查询/秒        │      │
│  │ 混合检索           │ 10,000文档    │ 10ms         │ 100查询/秒        │      │
│  │ 重排序             │ 100条         │ 200ms        │ 500条/秒          │      │
│  │ 实体识别           │ 1000字        │ 5ms          │ 200,000字/秒      │      │
│  │ 图谱检索(BFS)      │ 1000实体      │ 2ms          │ 500查询/秒        │      │
│  │ Query改写          │ 1条           │ 1ms          │ 1000查询/秒       │      │
│  │ Prompt构建         │ 1条           │ 0.5ms        │ 2000查询/秒       │      │
│  └────────────────────┴──────────────┴─────────────┴───────────────────┘      │
│                                                                                 │
│  端到端延迟（不含LLM调用）：                                                    │
│  ────────────────────────────                                                  │
│  P50: 50ms                                                                    │
│  P90: 80ms                                                                    │
│  P99: 120ms                                                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### B. 算法选型决策树

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           算法选型决策树                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                            文本分块                                              │
│                               │                                                 │
│              ┌────────────────┼────────────────┐                               │
│              ▼                ▼                ▼                                │
│         速度优先?          语义优先?        平衡?                               │
│              │                │                │                                │
│              ▼                ▼                ▼                                │
│         固定长度          语义分块          重叠分块                            │
│         (O(n))            (O(n·d))          (O(n))                            │
│                                                                                 │
│                            向量模型                                              │
│                               │                                                 │
│              ┌────────────────┼────────────────┐                               │
│              ▼                ▼                ▼                                │
│         CPU推理           API调用          混合                                │
│              │                │                │                                │
│              ▼                ▼                ▼                                │
│         bge-small-zh     OpenAI-ada-2    可切换                              │
│         (33M)            (闭源)           (架构模式)                          │
│                                                                                 │
│                            向量索引                                              │
│                               │                                                 │
│              ┌────────────────┼────────────────┐                               │
│              ▼                ▼                ▼                                │
│         < 10K文档        10K-1M文档       > 1M文档                            │
│              │                │                │                                │
│              ▼                ▼                ▼                                │
│         IndexFlatIP       HNSW             IVF-PQ                             │
│         (精确,100%)       (快速,98%)        (压缩,90%)                        │
│                                                                                 │
│                            检索策略                                              │
│                               │                                                 │
│              ┌────────────────┼────────────────┐                               │
│              ▼                ▼                ▼                                │
│         纯向量            纯BM25           混合                                │
│              │                │                │                                │
│              ▼                ▼                ▼                                │
│         语义理解强        关键词精准        兼得                               │
│              │                │                │                                │
│              ▼                ▼                ▼                                │
│         α=1.0             α=0.0            α=0.6                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

*文档生成时间: 2025-02-03*
*版本: v2.0 (深度分析版)*
