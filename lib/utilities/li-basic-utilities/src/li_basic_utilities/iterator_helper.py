from collections import deque
import threading
from typing import Callable, Iterable, Iterator

from li_basic_utilities.static_class import StaticClass


class IteratorHelper(StaticClass):
    @staticmethod
    def tee_filtered[T](iterable: Iterable[T], 
                        *filters: Callable[[T], bool]) -> tuple[Iterator[T], ...]:
        it: Iterator[T] = iter(iterable)
        queues_and_filters: list[tuple[deque[T], Callable[[T], bool]]] = [
            (deque(), filter) for filter in filters
        ]

        def generator(q: deque[T]):
            while True:
                while len(q) == 0:
                    try:
                        next_value: T = next(it)
                    except StopIteration:
                        return
                    
                    queue: deque[T]
                    filter: Callable[[T], bool]
                    for queue, filter in queues_and_filters:
                        if filter(next_value):
                            queue.append(next_value)
                yield q.popleft()

        return tuple(generator(queue) for queue, _ in queues_and_filters)
    
    # https://github.com/more-itertools/more-itertools/blob/master/more_itertools/more.py#L5396
    @staticmethod
    def tee_concurrent[T](iterable: Iterable[T], n: int = 2) -> tuple[Iterable[T], ...]:
        class _tee_concurrent:
            __slots__: tuple = ('iterator', 'link', 'lock')

            def __init__(self, iterable: Iterable[T]):
                if isinstance(iterable, _tee_concurrent):
                    self.iterator = iterable.iterator
                    self.link: list = iterable.link
                    self.lock = iterable.lock
                else:
                    self.iterator = iter(iterable)
                    self.link = [None, None]
                    self.lock = threading.Lock()

            def __iter__(self):
                return self

            def __next__(self) -> T:
                link: list = self.link
                if link[1] is None:
                    with self.lock:
                        if link[1] is None:
                            link[0] = next(self.iterator)
                            link[1] = [None, None]
                self.link = link[1]
                return link[0]

        if n <= 0:
            return ()
        iterator: _tee_concurrent = _tee_concurrent(iterable)
        result: list[_tee_concurrent] = [iterator]
        for _ in range(n - 1):
            result.append(_tee_concurrent(iterator))
        return tuple(result)

