# LightRAG

A lightweight, modular RAG (Retrieval-Augmented Generation) framework with full monitoring chain.

## Features

- **Modular Design**: Easy to customize and extend
- **Multiple Document Formats**: Support TXT, Markdown, JSON, PDF
- **Hybrid Retrieval**: Vector + BM25 hybrid search
- **Reranking**: Optional cross-encoder reranking
- **Full Monitoring**: Tracing, metrics, and logging
- **Local LLM Support**: OpenAI-compatible API

## Architecture

```
LightRAG Pipeline
=================

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

## Quick Start

```python
from src import LightRAGPipeline, LightRAGConfig

# Initialize
config = LightRAGConfig()
rag = LightRAGPipeline(config)

# Build Index
rag.index("./data")

# Query
result = rag.query("What is AI?")
print(result["answer"])

# Check Stats
print(rag.get_stats())
```

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- python >= 3.8
- sentence-transformers
- faiss-cpu
- openai
- rank-bm25
- torch

## Project Structure

```
lightrag_project/
├── src/
│   ├── __init__.py         # Main package
│   ├── config.py           # Configuration
│   ├── pipeline.py         # Main pipeline
│   ├── ingestion/          # Data ingestion
│   ├── embedding/          # Text embedding
│   ├── storage/            # Vector storage
│   ├── retrieval/          # Retrieval logic
│   ├── generation/         # LLM generation
│   └── monitoring/         # Monitoring & tracing
├── docs/
│   └── FLOWCHART.md        # Architecture diagrams
├── tests/
├── config/
├── data/
│   ├── raw/                # Raw documents
│   └── vector_store/       # Vector index
└── logs/
    └── rag_pipeline.log
```

## Monitoring

```python
# Get monitoring stats
stats = rag.get_stats()

# Export trace and metrics
rag.export_monitoring_data("./monitoring")
```

## License

MIT
