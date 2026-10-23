from random import Random
from typing import Iterable


class StringRandom:
    def __init__(self, 
                 generator: Random,
                 length: int,
                 allow_duplicate: bool = False,
                 characters: str = "abcdefghijklmnopqrstuvwxyz") -> None:
        self.__random: Random = generator

        self.__history: set[str] | None = set()
        if allow_duplicate:
            self.__history = None
        
        self.__characters = characters
        self.__length = length
    
    def _next_character_list(self):
        for _ in range(self.__length):
            yield self.__random.choice(self.__characters)

    def next(self) -> str:
        result_list: Iterable[str] = self._next_character_list()
        result: str = "".join(result_list)

        if self.__history is None:
            return result
        
        if result in self.__history:
            return self.next()
        
        self.__history.add(result)
        return result