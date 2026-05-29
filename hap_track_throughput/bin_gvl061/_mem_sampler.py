import threading
from typing import Optional


class PeakRssSampler:
    """Context manager sampling RSS of this process + descendants at ~20 Hz.

    Records peak and average RSS. Used only in memory-measurement mode.
    Ported verbatim from the 0.24.1 harness (hap_track_throughput/bin/_mem_sampler.py)
    so the 0.6.1 memory numbers are directly comparable.
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
