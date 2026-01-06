"""Tracking utilities for policy evaluation metrics.

This module provides thread-safe trackers for inference latency and action
discarding metrics during policy evaluation. Inspired by LeRobot's RTC
latency tracking implementation.
"""

from threading import Lock

import numpy as np
import rerun as rr


class LatencyTracker:
    """Tracks inference latency and logs metrics to rerun (thread-safe)."""

    def __init__(self):
        self.latencies_ms: list[float] = []
        self.chunk_idx: int = 0
        self.lock = Lock()

    def record(self, latency_ms: float, log_to_rerun: bool = True) -> None:
        """Record a single inference latency measurement."""
        with self.lock:
            self.latencies_ms.append(latency_ms)
            if log_to_rerun:
                rr.set_time("inference_idx", sequence=self.chunk_idx)
                rr.log("metrics/inference_latency", rr.Scalars(latency_ms))
            self.chunk_idx += 1

    def get_stats(self) -> dict:
        """Compute summary statistics."""
        with self.lock:
            if not self.latencies_ms:
                return {}
            arr = np.array(self.latencies_ms)
            return {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
            }

    def log_summary_to_rerun(self) -> None:
        """Log summary statistics and histogram to rerun."""
        with self.lock:
            if not self.latencies_ms:
                return

            stats = self.get_stats()

            # Log summary as markdown text
            summary_text = f"""**Inference Latency Stats**
- Count: {stats["count"]}
- Mean: {stats["mean"]:.1f}ms
- Std: {stats["std"]:.1f}ms
- Min: {stats["min"]:.1f}ms
- Max: {stats["max"]:.1f}ms
- P50: {stats["p50"]:.1f}ms
- P95: {stats["p95"]:.1f}ms
"""
            rr.log(
                "metrics/latency_summary",
                rr.TextDocument(summary_text, media_type=rr.MediaType.MARKDOWN),
            )

            # Log histogram
            hist, _ = np.histogram(self.latencies_ms, bins=20)
            rr.log("metrics/latency_histogram", rr.BarChart(hist))

    def reset(self) -> None:
        """Reset tracker for a new episode."""
        with self.lock:
            self.latencies_ms = []
            self.chunk_idx = 0


class DiscardTracker:
    """Tracks discarded actions and logs metrics to rerun (thread-safe)."""

    def __init__(self):
        self.discarded_counts: list[int] = []
        self.chunk_idx: int = 0
        self.lock = Lock()

    def record(self, n_discarded: int, log_to_rerun: bool = True) -> None:
        """Record number of discarded actions for a chunk."""
        with self.lock:
            self.discarded_counts.append(n_discarded)
            if log_to_rerun:
                rr.set_time("inference_idx", sequence=self.chunk_idx)
                rr.log("metrics/discarded_actions", rr.Scalars(n_discarded))
            self.chunk_idx += 1

    def get_stats(self) -> dict:
        """Compute summary statistics."""
        with self.lock:
            if not self.discarded_counts:
                return {}
            arr = np.array(self.discarded_counts)
            return {
                "count": len(arr),
                "total_discarded": int(np.sum(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
            }

    def log_summary_to_rerun(self) -> None:
        """Log summary statistics to rerun."""
        with self.lock:
            if not self.discarded_counts:
                return

            stats = self.get_stats()

            # Log summary as markdown text
            summary_text = f"""**Discarded Actions Stats**
- Chunks: {stats["count"]}
- Total discarded: {stats["total_discarded"]}
- Mean per chunk: {stats["mean"]:.1f}
- Std: {stats["std"]:.1f}
- Min: {stats["min"]}
- Max: {stats["max"]}
"""
            rr.log(
                "metrics/discard_summary",
                rr.TextDocument(summary_text, media_type=rr.MediaType.MARKDOWN),
            )

            # Log histogram
            if max(self.discarded_counts) > 0:
                hist, _ = np.histogram(
                    self.discarded_counts,
                    bins=min(20, max(self.discarded_counts) + 1),
                )
                rr.log("metrics/discard_histogram", rr.BarChart(hist))

    def reset(self) -> None:
        """Reset tracker for a new episode."""
        with self.lock:
            self.discarded_counts = []
            self.chunk_idx = 0
