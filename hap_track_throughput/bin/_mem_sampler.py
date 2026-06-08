import threading
from time import perf_counter_ns
from typing import Optional, TextIO


def _sample_rss() -> int:
    """RSS of this process + all descendants (bytes)."""
    import psutil

    try:
        proc = psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total
    except psutil.NoSuchProcess:
        return 0


class RssTimeSeriesSampler:
    """Sample total RSS over time, writing one `<prefix>,<elapsed_ns>,<rss>` row
    per sample to an open CSV (flushed each row, so the growth curve survives a
    mid-run kill). Tracks peak/avg too. Used for the memory-growth measurement."""

    def __init__(self, fh: TextIO, row_prefix: str, interval_s: float = 0.5):
        self.fh = fh
        self.row_prefix = row_prefix
        self.interval_s = interval_s
        self.peak = 0
        self._sum = 0
        self._count = 0
        self.avg = 0
        self._t0 = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _write(self, elapsed_ns: int, rss: int) -> None:
        self.fh.write(f"{self.row_prefix},{elapsed_ns},{rss}\n")
        self.fh.flush()
        if rss > self.peak:
            self.peak = rss
        self._sum += rss
        self._count += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._write(perf_counter_ns() - self._t0, _sample_rss())

    def __enter__(self) -> "RssTimeSeriesSampler":
        self._t0 = perf_counter_ns()
        self._write(0, _sample_rss())
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._write(perf_counter_ns() - self._t0, _sample_rss())
        self.avg = self._sum // self._count if self._count else self.peak


class PeakRssSampler:
    """Context manager that samples RSS of this process + all descendants at ~20 Hz.

    Used only in --measure-memory mode, never in the timed throughput path.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.peak: int = 0
        self.avg: int = 0
        self._sum: int = 0
        self._count: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> int:
        import psutil

        try:
            proc = psutil.Process()
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            return total
        except psutil.NoSuchProcess:
            return 0

    def _record(self, v: int) -> None:
        if v > self.peak:
            self.peak = v
        self._sum += v
        self._count += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._record(self._sample())

    def __enter__(self) -> "PeakRssSampler":
        self.peak = 0
        self._sum = 0
        self._count = 0
        self._record(self._sample())
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._record(self._sample())
        self.avg = self._sum // self._count if self._count else self.peak
