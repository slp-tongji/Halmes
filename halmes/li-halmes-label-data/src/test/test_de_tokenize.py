from li_halmes_configurations import HalmesTargetLlmConfiguration

from li_halmes_label_data.supported_tokenizer import SupportedTokenizer
from li_halmes_label_data.tokenizers.huggingface_tokenizer import HuggingfaceTokenizer


class FakeTokenizer(SupportedTokenizer):
    def __init__(self) -> None:
        super().__init__("fake")

    @property
    def model_id(self) -> str:
        raise NotImplementedError()

    def _decode(self, ids: list[int]) -> str:
        result: str = ""
        id: int
        for id in ids:
            if id == 0:
                continue    # special token
            if id < 0:
                result += "-"    # many-to-one: it will be removed at next positive id
                continue

            negative: int = result.find("-")
            if negative != -1:
                assert len(str(id)) == 1    # avoid many-to-many
                result = result[:negative]
            result += str(id)    # one-to-one (0-9) and one-to-many (>10)
        return result


fake: FakeTokenizer = FakeTokenizer()
print(fake._decode([1, 2, 0, 345, 67, 89]))    # 123456789
print(fake._decode([1, 2, 0, 345, -67, 8, 9]))    # 1234589
print(fake._decode([1, 2, 0, 345, -67]))    # 12345- (incomplete)

print(fake.de_tokenize([1, 2, 0, 345, 67, 89]))    # 1: 0, 2: 1, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5
print(fake.de_tokenize([1, 2, 0, 345, -67, 8, 9]))    # 1: 0, 2: 1, 3: 3, 4: 3, 5: 3, 8: 4, 9: 6
print(fake.de_tokenize([1, -2, 3]))    # 1: 0, 3: 1


print(fake.de_tokenize([0, 0, 0]))    # ""
print(fake.de_tokenize([-1, -2, -3]))    # -: 0, -: 1, -: 2
print(fake.de_tokenize([1, -2, -3, 4]))   # 1: 0, 4: 1
print(fake.de_tokenize([123, 0, 4567, 8, 910]))    # 1: 0, 2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 4, 1: 4, 0: 4
print(fake.de_tokenize([0, 1, -2, 3]))    # 1: 1, 3: 2
print(fake.de_tokenize([1, 0, 222, -3, -4, 5, 0, 0, 678]))    # 1: 0, 2: 2, 2: 2, 2: 2, 5: 3, 6: 8, 7: 8, 8: 8

real: HuggingfaceTokenizer = HuggingfaceTokenizer(HalmesTargetLlmConfiguration.qwen3_vl_8b())
codes: list[int] = real.inner_tokenizer.encode("你好，this is my 龘.").ids
i: int
for i in range(len(codes) + 1):
    print(codes[:i])
    print(real.de_tokenize(codes[:i]))
