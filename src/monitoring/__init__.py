"""
Monitoring Module
=================
数据监控链路模块 - 提供全链路监控、日志和指标追踪
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
import logging
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """追踪跨度"""
    name: str
    span_id: str
    parent_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status
        }


@dataclass
class Metric:
    """监控指标"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


class TraceContext:
    """追踪上下文"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.spans: List[Span] = []
        self.current_span: Optional[Span] = None
        self.span_stack: List[Span] = []
        self._span_id_counter = 0
        self._initialized = True

    def generate_span_id(self) -> str:
        """生成span ID"""
        self._span_id_counter += 1
        return f"span_{self._span_id_counter}"

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """开始一个追踪跨度"""
        span = Span(
            name=name,
            span_id=self.generate_span_id(),
            parent_id=self.current_span.span_id if self.current_span else None,
            start_time=datetime.now(),
            end_time=None,
            attributes=attributes or {},
            events=[]
        )

        self.span_stack.append(self.current_span)
        self.current_span = span

        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.events.append({
                "name": "exception",
                "timestamp": datetime.now().isoformat(),
                "attributes": {"exception.type": type(e).__name__, "exception.message": str(e)}
            })
            raise
        finally:
            span.end_time = datetime.now()
            self.spans.append(span)
            self.current_span = self.span_stack.pop() if self.span_stack else None

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """添加事件"""
        if self.current_span:
            self.current_span.events.append({
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "attributes": attributes or {}
            })

    def get_current_trace(self) -> Dict[str, Any]:
        """获取当前完整追踪"""
        return {
            "trace_id": "trace_1",
            "spans": [s.to_dict() for s in self.spans],
            "start_time": self.spans[0].start_time.isoformat() if self.spans else None,
            "end_time": self.spans[-1].end_time.isoformat() if self.spans else None
        }


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics: List[Metric] = []
        self.metrics_lock = threading.Lock()

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录指标"""
        with self.metrics_lock:
            self.metrics.append(Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {}
            ))

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录瞬时值"""
        self.record(name, value, labels)

    def counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """记录计数器"""
        self.record(name, value, labels)

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录直方图值"""
        self.record(name, value, labels)

    def get_metrics(self) -> List[Dict[str, Any]]:
        """获取所有指标"""
        with self.metrics_lock:
            return [asdict(m) for m in self.metrics]

    def get_summary(self, metric_name: str) -> Dict[str, Any]:
        """获取指标摘要"""
        values = [m.value for m in self.metrics if m.name == metric_name]
        if not values:
            return {}

        return {
            "name": metric_name,
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }


class Monitor:
    """监控中心"""

    def __init__(self, enable_tracing: bool = True, enable_metrics: bool = True, log_dir: str = "./logs"):
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.trace_context = TraceContext()
        self.metrics_collector = MetricsCollector()

        # 日志处理器
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'rag_pipeline.log'),
                logging.StreamHandler()
            ]
        )

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """开始追踪跨度"""
        if not self.enable_tracing:
            yield None
            return

        with self.trace_context.start_span(name, attributes) as span:
            yield span

    def log_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """记录事件"""
        self.trace_context.add_event(name, attributes)
        logger.info(f"Event: {name}", extra=attributes or {})

    def record_latency(self, operation: str, duration: float):
        """记录延迟"""
        self.metrics_collector.histogram(f"{operation}.latency_ms", duration * 1000)
        logger.info(f"{operation} latency: {duration * 1000:.2f}ms")

    def record_tokens(self, prompt_tokens: int, completion_tokens: int):
        """记录token使用"""
        self.metrics_collector.counter("tokens.prompt", prompt_tokens)
        self.metrics_collector.counter("tokens.completion", completion_tokens)
        self.metrics_collector.counter("tokens.total", prompt_tokens + completion_tokens)

    def record_retrieval(self, query: str, results_count: int, latency: float):
        """记录检索指标"""
        self.metrics_collector.histogram("retrieval.latency_ms", latency * 1000)
        self.metrics_collector.gauge("retrieval.results_count", results_count)
        self.metrics_collector.counter("retrieval.total")

        logger.info(f"Retrieval: query='{query[:50]}...', results={results_count}, latency={latency*1000:.2f}ms")

    def record_generation(self, latency: float, tokens: int):
        """记录生成指标"""
        self.metrics_collector.histogram("generation.latency_ms", latency * 1000)
        self.metrics_collector.gauge("generation.tokens", tokens)
        self.metrics_collector.counter("generation.total")

    def record_data_ingestion(self, files_processed: int, chunks_created: int):
        """记录数据摄入指标"""
        self.metrics_collector.counter("ingestion.files", files_processed)
        self.metrics_collector.counter("ingestion.chunks", chunks_created)
        logger.info(f"Ingestion: files={files_processed}, chunks={chunks_created}")

    def record_embedding(self, documents_count: int, latency: float):
        """记录向量化指标"""
        self.metrics_collector.histogram("embedding.latency_ms", latency * 1000)
        self.metrics_collector.gauge("embedding.documents_count", documents_count)
        self.metrics_collector.counter("embedding.total")

    def get_trace(self) -> Dict[str, Any]:
        """获取追踪数据"""
        return self.trace_context.get_current_trace()

    def get_metrics(self) -> List[Dict[str, Any]]:
        """获取指标数据"""
        return self.metrics_collector.get_metrics()

    def export_trace(self, path: str):
        """导出追踪数据到文件"""
        trace_data = self.get_trace()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Trace exported to {path}")

    def export_metrics(self, path: str):
        """导出指标数据到文件"""
        metrics_data = self.get_metrics()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics exported to {path}")

    def get_stats(self) -> Dict[str, Any]:
        """获取完整统计信息"""
        return {
            "tracing": {
                "enabled": self.enable_tracing,
                "total_spans": len(self.trace_context.spans)
            },
            "metrics": {
                "enabled": self.enable_metrics,
                "total_metrics": len(self.metrics_collector.metrics)
            },
            "summaries": {
                "retrieval_latency": self.metrics_collector.get_summary("retrieval.latency_ms"),
                "generation_latency": self.metrics_collector.get_summary("generation.latency_ms"),
                "embedding_latency": self.metrics_collector.get_summary("embedding.latency_ms"),
                "total_tokens": self.metrics_collector.get_summary("tokens.total")
            }
        }


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, monitor: Monitor):
        self.monitor = monitor

    @contextmanager
    def measure(self, operation: str, **attributes):
        """测量操作耗时"""
        start_time = time.time()
        try:
            with self.monitor.span(operation, attributes):
                yield
        finally:
            duration = time.time() - start_time
            self.monitor.record_latency(operation, duration)
