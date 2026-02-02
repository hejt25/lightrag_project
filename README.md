# LightRAG

A lightweight, modular **RAG (Retrieval-Augmented Generation)** framework with full monitoring chain and insurance domain support.

## Highlights

### 1. Professional Prompt Template System

Industry-leading prompt management with **modular component design**:

```
Prompt = Persona + System Prompt + Domain Knowledge + Constraints + Context + Output Format
```

- **Component-based architecture**: Each part is independently configurable
- **Version control**: Track changes, rollback support, A/B testing
- **Multi-domain templates**: General + Insurance-specific templates
- **Dynamic composition**: Build prompts at runtime

```python
from src.prompt_templates import InsuranceProductPrompt, PromptBuilder

# Use pre-built templates
template = InsuranceProductPrompt()
prompt = template.render(context, question)

# Or build dynamically
builder = PromptBuilder()
builder.register_template(template)
result = builder.build(context, question, template_name="InsuranceProductPrompt")
```

### 2. Insurance Domain Intelligence

Specialized query rewriting for insurance scenarios:

```python
from src import InsuranceQueryRewriter, UserProfile

# Create user profile
profile = UserProfile(
    user_id="user_001",
    age=35,
    income_level="medium",
    insurance_history=["医疗险"]
)

# Rewrite query with context
rewriter = InsuranceQueryRewriter()
result = rewriter.rewrite(
    query="有什么推荐",
    user_id="user_001",
    session_id="session_001",
    user_profile=profile
)
# Output: "有什么推荐 重疾险 寿险 医疗险 教育金"
```

**Features**:
- User profile enrichment (age, income, insurance history)
- Conversation context resolution (pronoun消解)
- User behavior tracking (views, clicks, searches)
- Insurance term expansion (e.g., "重疾" -> "重大疾病保险")
- Intent classification for insurance domain

### 3. Full Monitoring Chain

End-to-end observability for production:

- **Tracing**: Span-based distributed tracing
- **Metrics**: Latency, token usage, retrieval counts
- **Events**: Custom event logging
- **Performance tracking**: Per-operation timing

```python
from src import Monitor

monitor = Monitor(enable_tracing=True, enable_metrics=True)

# Automatic tracking
with monitor.span("retrieval"):
    results = retriever.retrieve(query)

# Manual metrics
monitor.record_latency("generation", 1.5)
monitor.record_tokens(prompt_tokens, completion_tokens)

# Export for analysis
monitor.export_trace("./traces")
monitor.export_metrics("./metrics")
```

### 4. Modular Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LightRAG Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  Ingestion → Embedding → Storage → Retrieval → Generation  │
│      ↓           ↓           ↓           ↓           ↓      │
│  TXT/MD/JSON  Sentence    FAISS     Vector/Hybrid   Local/ │
│  PDF Loader   Transformers In-Memory  BM25         OpenAI   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Query Rewrite Module                       │
│  UserProfile + ConversationHistory + UserBehaviorTracker    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Prompt Templates                           │
│  Persona + System + Constraints + Output Format + Examples  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Monitoring                              │
│  Tracing + Metrics + Events + Logging                        │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Modular Design**: Easy to customize and extend
- **Multiple Document Formats**: TXT, Markdown, JSON, PDF
- **Hybrid Retrieval**: Vector similarity + BM25 keyword search
- **Reranking**: Cross-encoder support for better results
- **Local LLM Support**: OpenAI-compatible API (Qwen, Llama, etc.)
- **Production Ready**: Full monitoring, logging, metrics

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

# Get Stats
print(rag.get_stats())
```

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- sentence-transformers
- faiss-cpu
- openai
- rank-bm25
- torch
- transformers

## Project Structure

```
lightrag_project/
├── src/
│   ├── __init__.py              # Main package
│   ├── config.py                # Configuration
│   ├── pipeline.py              # Main pipeline
│   ├── ingestion/               # Data ingestion
│   ├── embedding/               # Text embedding
│   ├── storage/                 # Vector storage
│   ├── retrieval/               # Retrieval logic
│   ├── generation/              # LLM generation + Prompt Templates
│   ├── query_rewrite/           # Insurance query rewriting
│   └── monitoring/              # Monitoring & tracing
├── src/prompt_templates/        # Professional prompt templates
│   ├── core/                    # Base classes & builder
│   ├── insurance/               # Insurance domain templates
│   └── common/                  # General templates
├── docs/
│   └── FLOWCHART.md             # Architecture diagrams
├── tests/
├── config/
├── data/
└── logs/
```

## Insurance Use Case Example

```python
from src import (
    LightRAGPipeline,
    InsuranceQueryRewriter,
    UserProfile
)

# Initialize RAG pipeline
rag = LightRAGPipeline()

# Build index with insurance documents
rag.index("./data/insurance")

# Create user profile
user = UserProfile(
    user_id="u001",
    age=30,
    income_level="high",
    risk_preference="moderate",
    insurance_history=["意外险"]
)

# Initialize query rewriter
rewriter = InsuranceQueryRewriter()

# Process user query with context
query = "那个保险怎么样？"
rewritten = rewriter.rewrite(query, "u001", "s001", user)

# Query RAG system
result = rag.query(rewritten.rewritten_query)

print(f"Answer: {result['answer']}")
print(f"Metrics: {result['metrics']}")
```

## Monitoring

```python
# Get monitoring stats
stats = rag.get_stats()

# Export trace and metrics
rag.export_monitoring_data("./monitoring")

# Monitor metrics summary
print(stats["monitoring"]["summaries"])
```

## License

MIT
