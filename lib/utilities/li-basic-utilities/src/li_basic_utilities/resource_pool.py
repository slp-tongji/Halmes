import contextlib
import queue
from typing import Iterable


class ResourcePool[T]:
    def __init__(self, resources: Iterable[T]) -> None:
        self.__resources = queue.Queue[T]()
        resource: T
        for resource in resources:
            self.__resources.put(resource)

    @contextlib.contextmanager
    def borrow(self):
        resource: T = self.__resources.get()
        try:  
            yield resource
        finally:  
            self.__resources.put(resource)
