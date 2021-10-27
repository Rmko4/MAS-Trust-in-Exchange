from collections import OrderedDict


class LimitedDict(OrderedDict):
    def __init__(self, max_size=10, other=(), /, **kwds) -> None:
        super().__init__(other, **kwds)
        self.max_size = max_size

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        self._prune()

    def update(self, *args, **kwargs) -> None:
        super().update(args, kwargs)
        self._prune()

    def _prune(self) -> None:
        while len(self) > self.max_size:
            self.popitem()
