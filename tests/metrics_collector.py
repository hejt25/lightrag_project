"""
Test metrics collection and reporting module for LightRAG.
"""
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAIL = "xfail"


@dataclass
class TestMetrics:
    """Metrics for a single test case."""
    test_id: str
    test_name: str
    test_category: str
    status: str
    duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "test_category": self.test_category,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "custom_metrics": self.custom_metrics,
            "metadata": self.metadata,
        }
        if self.error_message:
            data["error"] = {
                "message": self.error_message,
                "type": self.error_type,
                "traceback": self.traceback
            }
        return data


@dataclass
class CategoryMetrics:
    """Aggregated metrics for a test category."""
    category_name: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0

    def update(self, test: TestMetrics):
        """Update metrics with a new test result."""
        self.total_tests += 1
        self.total_duration_ms += test.duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.total_tests
        self.min_duration_ms = min(self.min_duration_ms, test.duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, test.duration_ms)

        status = test.status
        if status == TestStatus.PASSED.value:
            self.passed += 1
        elif status == TestStatus.FAILED.value:
            self.failed += 1
        elif status == TestStatus.ERROR.value:
            self.errors += 1
        elif status == TestStatus.SKIPPED.value:
            self.skipped += 1
        elif status == TestStatus.XFAIL.value:
            self.xfailed += 1


class TestMetricsCollector:
    """
    Central metrics collector for the test suite.

    Thread-safe implementation for concurrent test execution.
    """

    def __init__(self, suite_name: str = "LightRAG Test Suite"):
        self.suite_name = suite_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._lock = threading.Lock()
        self._test_results: List[TestMetrics] = []
        self._category_metrics: Dict[str, CategoryMetrics] = {}

    def start_suite(self):
        """Mark the start of a test suite execution."""
        self.start_time = time.time()

    def end_suite(self):
        """Mark the end of a test suite execution."""
        self.end_time = time.time()

    def record_test(self, test: TestMetrics):
        """Record a test result (thread-safe)."""
        with self._lock:
            self._test_results.append(test)

            # Update category metrics
            category = test.test_category
            if category not in self._category_metrics:
                self._category_metrics[category] = CategoryMetrics(category_name=category)
            self._category_metrics[category].update(test)

    def record_pass(self, test_id: str, test_name: str, test_category: str,
                    duration_ms: float, custom_metrics: Dict = None,
                    metadata: Dict = None):
        """Record a passed test."""
        test = TestMetrics(
            test_id=test_id,
            test_name=test_name,
            test_category=test_category,
            status=TestStatus.PASSED.value,
            duration_ms=duration_ms,
            custom_metrics=custom_metrics or {},
            metadata=metadata or {}
        )
        self.record_test(test)

    def record_fail(self, test_id: str, test_name: str, test_category: str,
                    duration_ms: float, error_message: str, error_type: str = None,
                    traceback: str = None, custom_metrics: Dict = None,
                    metadata: Dict = None):
        """Record a failed test."""
        test = TestMetrics(
            test_id=test_id,
            test_name=test_name,
            test_category=test_category,
            status=TestStatus.FAILED.value,
            duration_ms=duration_ms,
            error_message=error_message,
            error_type=error_type,
            traceback=traceback,
            custom_metrics=custom_metrics or {},
            metadata=metadata or {}
        )
        self.record_test(test)

    def record_error(self, test_id: str, test_name: str, test_category: str,
                    duration_ms: float, error_message: str, error_type: str = None,
                    traceback: str = None, custom_metrics: Dict = None,
                    metadata: Dict = None):
        """Record an errored test (exception during execution)."""
        test = TestMetrics(
            test_id=test_id,
            test_name=test_name,
            test_category=test_category,
            status=TestStatus.ERROR.value,
            duration_ms=duration_ms,
            error_message=error_message,
            error_type=error_type,
            traceback=traceback,
            custom_metrics=custom_metrics or {},
            metadata=metadata or {}
        )
        self.record_test(test)

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the test suite."""
        total = len(self._test_results)
        passed = sum(1 for t in self._test_results if t.status == TestStatus.PASSED.value)
        failed = sum(1 for t in self._test_results if t.status == TestStatus.FAILED.value)
        errors = sum(1 for t in self._test_results if t.status == TestStatus.ERROR.value)
        skipped = sum(1 for t in self._test_results if t.status == TestStatus.SKIPPED.value)

        durations = [t.duration_ms for t in self._test_results]

        return {
            "suite_name": self.suite_name,
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "pass_rate": (passed / total * 100) if total > 0 else 0,
                "failure_rate": (failed / total * 100) if total > 0 else 0,
                "total_duration_ms": sum(durations),
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "min_duration_ms": min(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
            },
            "categories": {
                name: {
                    "total_tests": cat.total_tests,
                    "passed": cat.passed,
                    "failed": cat.failed,
                    "pass_rate": (cat.passed / cat.total_tests * 100) if cat.total_tests > 0 else 0,
                    "avg_duration_ms": cat.avg_duration_ms,
                }
                for name, cat in self._category_metrics.items()
            }
        }

    def get_functional_metrics(self) -> Dict[str, Any]:
        """Calculate functional correctness metrics."""
        total = len(self._test_results)
        if total == 0:
            return {"success_rate": 0, "crash_rate": 0, "exception_rate": 0}

        passed = sum(1 for t in self._test_results if t.status == TestStatus.PASSED.value)
        errors = sum(1 for t in self._test_results if t.status == TestStatus.ERROR.value)
        failed = sum(1 for t in self._test_results if t.status == TestStatus.FAILED.value)

        return {
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "crash_rate": (errors / total * 100) if total > 0 else 0,
            "failure_rate": (failed / total * 100) if total > 0 else 0,
            "exception_rate": (errors / total * 100) if total > 0 else 0,
            "total_tests": total,
            "passed": passed,
            "failed": failed + errors,
        }

    def get_boundary_metrics(self) -> Dict[str, Any]:
        """Calculate boundary handling metrics."""
        boundary_tests = [t for t in self._test_results if t.test_category == "boundary"]
        if not boundary_tests:
            return {}

        passed = sum(1 for t in boundary_tests if t.status == TestStatus.PASSED.value)
        return {
            "boundary_tests_total": len(boundary_tests),
            "boundary_passed": passed,
            "boundary_pass_rate": (passed / len(boundary_tests) * 100) if boundary_tests else 0,
        }

    def get_retrieval_metrics(self) -> Dict[str, Any]:
        """Calculate retrieval quality metrics if applicable."""
        retrieval_tests = [t for t in self._test_results if t.test_category == "retrieval"]
        if not retrieval_tests:
            return {}

        # Extract custom metrics from retrieval tests
        recalls = []
        precisions = []
        mrrs = []
        hit_rates = []

        for test in retrieval_tests:
            if "recall" in test.custom_metrics:
                recalls.append(test.custom_metrics["recall"])
            if "precision" in test.custom_metrics:
                precisions.append(test.custom_metrics["precision"])
            if "mrr" in test.custom_metrics:
                mrrs.append(test.custom_metrics["mrr"])
            if "hit_rate" in test.custom_metrics:
                hit_rates.append(test.custom_metrics["hit_rate"])

        return {
            "retrieval_tests_total": len(retrieval_tests),
            "avg_recall": sum(recalls) / len(recalls) if recalls else 0,
            "avg_precision": sum(precisions) / len(precisions) if precisions else 0,
            "avg_mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
            "avg_hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0,
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        perf_tests = [t for t in self._test_results if t.test_category == "performance"]
        if not perf_tests:
            return {}

        # Extract concurrent safety info
        concurrent_tests = [t for t in perf_tests if "concurrent" in t.test_name.lower()]

        return {
            "performance_tests_total": len(perf_tests),
            "avg_duration_ms": sum(t.duration_ms for t in perf_tests) / len(perf_tests),
            "concurrent_safe_count": sum(1 for t in concurrent_tests
                                       if t.status == TestStatus.PASSED.value),
            "concurrent_tests_total": len(concurrent_tests),
        }

    def get_detailed_results(self) -> List[Dict[str, Any]]:
        """Get detailed test results."""
        return [test.to_dict() for test in self._test_results]

    def generate_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        report = {
            "metadata": {
                "suite_name": self.suite_name,
                "generated_at": datetime.now().isoformat(),
                "total_duration_ms": (self.end_time - self.start_time) * 1000 if self.end_time else 0,
            },
            "summary": self.get_summary(),
            "functional_metrics": self.get_functional_metrics(),
            "boundary_metrics": self.get_boundary_metrics(),
            "retrieval_metrics": self.get_retrieval_metrics(),
            "performance_metrics": self.get_performance_metrics(),
            "detailed_results": self.get_detailed_results(),
        }

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def reset(self):
        """Reset all metrics for a new test run."""
        with self._lock:
            self._test_results = []
            self._category_metrics = {}
            self.start_time = None
            self.end_time = None


# Convenience function for creating metrics collectors
def create_metrics_collector(suite_name: str = "LightRAG Test Suite") -> TestMetricsCollector:
    """Create a new metrics collector instance."""
    return TestMetricsCollector(suite_name)


# Pytest plugin integration
@pytest.fixture
def metrics_collector():
    """Pytest fixture providing a metrics collector."""
    collector = create_metrics_collector()
    collector.start_suite()
    yield collector
    collector.end_suite()
