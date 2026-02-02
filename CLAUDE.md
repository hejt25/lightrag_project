# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python3 -m pytest tests/ -v

# Run a single test
python3 -m pytest tests/test_pipeline.py::TestConfig -v

# Run pipeline example
python3 src/pipeline.py

# CLI usage
python3 src/run.py --data ./data --index --query "什么是AI？"
```

## Architecture

LightRAG is a modular RAG (Retrieval-Augmented Generation) framework with full monitoring chain.

### Pipeline Flow
```
Raw Data → Ingestion → Embedding → Storage
                                   ↓
User Query → Retrieval ← ← ← ← ← ← ← ←
                 ↓                  ↑
            Reranking               |
                 ↓                  |
            Generation → Answer ← ←
                      ↓
              Monitoring ← ← ← ← ←
```

### Core Components (src/)

1. **ingestion/** - Document loading and chunking
   - `DataIngestionPipeline` - Main ingestion orchestrator
   - `Chunker` - Overlapping text chunking
   - Loaders: TextLoader, MarkdownLoader, JSONLoader, PDFLoader

2. **embedding/** - Text vectorization
   - `EmbeddingPipeline` - Batch embedding orchestration
   - `SentenceTransformerEmbedder` - Local model support
   - `OpenAIEmbedder` - API-based embedding

3. **storage/** - Vector storage
   - `FAISSVectorStore` - FAISS-based storage (primary)
   - `InMemoryVectorStore` - Simple in-memory fallback

4. **retrieval/** - Document retrieval
   - `RetrievalPipeline` - Retrieval orchestrator
   - `VectorRetriever` - Dense vector similarity search
   - `HybridRetriever` - Vector + BM25 hybrid search
   - `Reranker` - Cross-encoder reranking

5. **generation/** - LLM generation
   - `GenerationPipeline` - Generation orchestrator
   - `LocalLLM` / `OpenAILLM` - LLM backends
   - `PromptBuilder` - RAG prompt construction

6. **monitoring/** - Full monitoring chain
   - `Monitor` - Central monitoring hub
   - `TraceContext` - Distributed tracing
   - `MetricsCollector` - Metrics aggregation
   - Tracks: latency, tokens, retrieval results, events

### Main Entry Points

- `src/pipeline.py` - `LightRAGPipeline` class for full workflow
- `src/__init__.py` - Exported public API
- `src/run.py` - CLI interface

### Configuration

`src/config.py` defines `LightRAGConfig` dataclass with nested configs for:
- `embedding` - Model name, device, batch size
- `vector_store` - Storage type, dimension, metric
- `llm` - Model name, API base, max tokens
- `monitoring` - Tracing, metrics, log level

Also `config/config.json` for JSON config loading.

## Key Patterns

- All components use context managers for monitoring (`with monitor.span(...)`)
- `PerformanceMonitor` wrapper measures operation latency
- Retrieval returns `RetrievalResult` with document_id, content, score, metadata
- Query response includes metrics: latency_ms, tokens used
- Use `pytest.importorskip()` for optional dependencies in tests
