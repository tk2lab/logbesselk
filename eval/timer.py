import time


class Timer:

    def __init__(self):
        self._start = None
        self.duration = []

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.duration.append(time.time() - self._start)
