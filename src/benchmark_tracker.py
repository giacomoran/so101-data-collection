"""
Benchmark metrics tracking for SO-101 data collection experiments.

Tracks metrics across data collection sessions and persists them to CSV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SessionMetrics:
    """Metrics for a single data collection session."""

    task: str
    setup: str
    session_start: float  # Unix timestamp
    session_end: float  # Unix timestamp
    collection_time_s: float  # Wall-clock time minus encoding time
    encoding_time_s: float  # Time spent encoding videos
    episodes_recorded: int
    total_frames: int
    mistakes: int  # Number of re-recorded episodes

    @property
    def session_start_iso(self) -> str:
        """Session start as ISO formatted string."""
        return datetime.fromtimestamp(self.session_start).isoformat()

    @property
    def session_end_iso(self) -> str:
        """Session end as ISO formatted string."""
        return datetime.fromtimestamp(self.session_end).isoformat()

    def to_csv_row(self) -> dict[str, str | int | float]:
        """Convert to CSV row dict."""
        return {
            "task": self.task,
            "setup": self.setup,
            "session_start": self.session_start_iso,
            "session_end": self.session_end_iso,
            "collection_time_s": round(self.collection_time_s, 2),
            "encoding_time_s": round(self.encoding_time_s, 2),
            "episodes_recorded": self.episodes_recorded,
            "total_frames": self.total_frames,
            "mistakes": self.mistakes,
        }


class BenchmarkTracker:
    """
    Tracks and persists benchmark metrics across collection sessions.

    Maintains a CSV file that accumulates metrics from all collection runs.
    """

    CSV_COLUMNS = [
        "task",
        "setup",
        "session_start",
        "session_end",
        "collection_time_s",
        "encoding_time_s",
        "episodes_recorded",
        "total_frames",
        "mistakes",
    ]

    def __init__(self, csv_path: Path | str = "benchmark_metrics.csv"):
        """
        Initialize the benchmark tracker.

        Args:
            csv_path: Path to the CSV file for storing metrics.
        """
        self.csv_path = Path(csv_path)
        self._ensure_csv_exists()

    def _ensure_csv_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()

    def log_session(self, metrics: SessionMetrics) -> None:
        """
        Log a session's metrics to the CSV file.

        Args:
            metrics: The session metrics to log.
        """
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writerow(metrics.to_csv_row())

    def load_all_sessions(self) -> list[dict]:
        """
        Load all recorded sessions from the CSV file.

        Returns:
            List of session dictionaries.
        """
        if not self.csv_path.exists():
            return []

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_summary_by_condition(self) -> dict[tuple[str, str], dict]:
        """
        Get aggregated summary statistics by (task, setup) condition.

        Returns:
            Dictionary mapping (task, setup) to aggregated stats.
        """
        sessions = self.load_all_sessions()
        summary: dict[tuple[str, str], dict] = {}

        for session in sessions:
            key = (session["task"], session["setup"])
            if key not in summary:
                summary[key] = {
                    "total_collection_time_s": 0.0,
                    "total_encoding_time_s": 0.0,
                    "total_episodes": 0,
                    "total_frames": 0,
                    "total_mistakes": 0,
                    "sessions": 0,
                }

            summary[key]["total_collection_time_s"] += float(
                session["collection_time_s"]
            )
            summary[key]["total_encoding_time_s"] += float(session["encoding_time_s"])
            summary[key]["total_episodes"] += int(session["episodes_recorded"])
            summary[key]["total_frames"] += int(session["total_frames"])
            summary[key]["total_mistakes"] += int(session["mistakes"])
            summary[key]["sessions"] += 1

        return summary

    def print_summary(self) -> None:
        """Print a formatted summary of all benchmark data."""
        summary = self.get_summary_by_condition()

        if not summary:
            print("No benchmark data recorded yet.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        for (task, setup), stats in sorted(summary.items()):
            print(f"\n{task} / {setup}:")
            print(f"  Sessions:          {stats['sessions']}")
            print(f"  Total episodes:    {stats['total_episodes']}")
            print(f"  Total frames:      {stats['total_frames']}")
            print(f"  Collection time:   {stats['total_collection_time_s']:.1f}s")
            print(f"  Encoding time:     {stats['total_encoding_time_s']:.1f}s")
            print(f"  Total mistakes:    {stats['total_mistakes']}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Print summary when run directly
    tracker = BenchmarkTracker()
    tracker.print_summary()
