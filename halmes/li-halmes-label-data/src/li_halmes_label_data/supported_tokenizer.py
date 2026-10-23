import abc
import itertools


class SupportedTokenizer(abc.ABC):
    def __init__(self, target_id: str) -> None:
        super().__init__()
        self.__target_id = target_id

    @property
    def target_id(self):
        return self.__target_id

    def _decode(self, ids: list[int]) -> str:
        ...

    def de_tokenize(self, tokens: list[int]) -> tuple[str, list[int]]:
        # Tokens and characters can have the following relationships:
        # 1. One-to-many: One token corresponds to multiple characters (e.g., a English word).
        # 2. One-to-one: One token corresponds to one character (e.g., a Chinese character).
        # 3. Many-to-one (or many): Multiple tokens correspond to one character (e.g., a rare Chinese character).
        #
        # In the many-to-one case, say, if `tokens[1:3]` corresponds to one character,
        # we assume that:
        # `decode(tokens[1:2])` (the incomplete stuff) will not be shorter than `decode(tokens[1:3])`;
        # otherwise, the algorithm would consider `tokens[2]` to correspond to the character,
        # as `tokens[1]` is more likely to be a skipped special token.

        response: str = self._decode(tokens)

        previous_length: int = len(response)
        text_to_token: list[int] = []
        index: int
        for index in range(len(tokens) - 1, -1, -1):
            current_length: int = len(self._decode(tokens[:index]))
            contribution: int = previous_length - current_length
            if contribution > 0:
                text_to_token.extend(itertools.repeat(index, contribution))
                previous_length = current_length
        text_to_token.reverse()
        return response, text_to_token
    