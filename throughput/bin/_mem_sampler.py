import threading
from typing import Optional


class PeakRssSampler:
    """Context manager that samples RSS of this process + all descendants at ~20 Hz.

    Used only in --measure-memory mode, never in the timed throughput path.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.peak: int = 0
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

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            v = self._sample()
            if v > self.peak:
                self.peak = v

    def __enter__(self) -> "PeakRssSampler":
        self.peak = self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        v = self._sample()
        if v > self.peak:
            self.peak = v
