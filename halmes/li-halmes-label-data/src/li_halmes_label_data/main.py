from collections import defaultdict
import datetime
import itertools
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence

import csdir
from li_basic_utilities import CsvFileWriter, StaticClass, ThreadPoolOrCurrentThreadExecutor, ResourcePool
from li_halmes_configurations import HalmesTargetLlm
from li_halmes_data_preparer import HalmesDataManager, HalmesRawData
from li_halmes_model import HalmesFeature, HalmesFeatureExtractor
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLog, TomlLogger
import more_itertools
from statictorch import Tensor2d
import torch

from li_halmes_label_data.supported_judge import SupportedJudge
from li_halmes_label_data.supported_tokenizer import SupportedTokenizer


class Main(StaticClass):
    @staticmethod
    def __pack_features(
        judgement: Iterable[SupportedJudge.Word],
        features: Sequence[HalmesFeature],
        text_to_token: list[int],
        min_length: int = 2,
        max_length: int = 50,
        max_repeat_non_hallucination: int | None = 1,
        max_repeat_hallucination: int | None = 2,
    ) -> Iterable[tuple[bool, Tensor2d]]:
        word_count: dict[str, int] = defaultdict(lambda: 0)
        seen_hallucination: bool = False

        word: str
        character_index: int
        is_hallucination: bool
        for word, character_index, is_hallucination in sorted(judgement, key=lambda x: x[1]):
            if is_hallucination:
                seen_hallucination = True
            else:
                if seen_hallucination:
                    continue    # skip all non-hallucinated words after hallucination happens

            token_index: int = text_to_token[character_index]

            max_repeat: int | None = max_repeat_hallucination if is_hallucination else max_repeat_non_hallucination
            if max_repeat is not None:
                if word_count[word] >= max_repeat:
                    continue
                word_count[word] += 1

            if token_index < min_length:
                continue

            yield is_hallucination, Tensor2d(torch.stack([
                HalmesFeatureExtractor.to_single_tensor(f)
                for f in features[max(token_index - max_length, 0):token_index]
            ], dim=0))

    class _Resource(NamedTuple):
        tokenizer: SupportedTokenizer
        judge: SupportedJudge
        input: HalmesDataManager[HalmesRawData]
        output_hallucination: HalmesDataManager[Tensor2d]
        output_non_hallucination: HalmesDataManager[Tensor2d]
    
    @staticmethod
    def __normalize_bag(feature_bag: list[tuple[bool, Tensor2d]]):
        if len(feature_bag) == 0:
            return
        all: torch.Tensor = torch.cat([f for _, f in feature_bag], dim=0)
        mean: torch.Tensor = all.mean(dim=0)
        std: torch.Tensor = all.std(dim=0).clamp_min(1e-6)
        item: tuple[bool, Tensor2d]
        for item in feature_bag:
            yield item[0], Tensor2d((item[1] - mean) / std)

    @staticmethod
    def __run(resources: ResourcePool[_Resource], 
            raw_data_id: str,
            log: TomlLog, 
            csv_log: CsvFileWriter):
        resource: Main._Resource
        with (resources.borrow() as resource, log):
            log.set_document_value(raw_data_id, "raw_data_id")

            raw: HalmesRawData | None
            _, raw = resource.input.get(raw_data_id)
            assert raw is not None

            text: str
            text_to_token: list[int]
            text, text_to_token = resource.tokenizer.de_tokenize(raw["output"].tolist())
            log.set_document_value(text, "raw_data_output")

            judgement: Iterable[SupportedJudge.Word] = resource.judge.judge(
                raw["data_id"], 
                LlmAssistantMessage(text, None), 
                log.create_sub_log("judge")
            )
            
            results: Iterable[tuple[bool, Tensor2d]] = Main.__pack_features(
                judgement,
                raw["features"],
                text_to_token
            )
            results = Main.__normalize_bag(list(results))

            index_width: int = len(str(len(text)))

            index: int
            is_hallucination: bool
            feature: Tensor2d
            for index, (is_hallucination, feature) in enumerate(results):
                id: str = f"{raw_data_id}_{index:0{index_width}}"
                if is_hallucination:
                    resource.output_hallucination.upsert(id, feature)
                    log.set_document_value("hallucination", ["result", id])
                else:
                    resource.output_non_hallucination.upsert(id, feature)
                    log.set_document_value("non_hallucination", ["result", id])

            csv_log.write_row([raw_data_id, log.name])

    @staticmethod
    def main(output_directory: Path, 
             tokenizer: Callable[[], SupportedTokenizer], 
             input: Path,
             multithreading: int | None,
             judge: Callable[[], SupportedJudge]) -> None:
        target_id: str = tokenizer().target_id

        log_directory: Path = csdir.create_directory(
            output_directory / "logs" / target_id / 
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"Logs: {log_directory}")

        logger: TomlLogger = TomlLogger(log_directory / "logs")

        executor: ThreadPoolOrCurrentThreadExecutor
        log_csv: CsvFileWriter
        with (ThreadPoolOrCurrentThreadExecutor(multithreading) as executor,
              CsvFileWriter(log_directory / "log.csv") as log_csv):
            log_csv.write_row(["input", "log"])

            resource_pool: ResourcePool[Main._Resource] = ResourcePool(
                (
                    Main._Resource(
                        tokenizer(), 
                        judge(), 
                        HalmesDataManager(input / target_id, True),
                        HalmesDataManager(output_directory / target_id / "hallucination", False),
                        HalmesDataManager(output_directory / target_id / "non_hallucination", False),
                    )
                    for _ in range(executor.worker_count if executor.worker_count is not None else 1)
                )
            )

            input_data_manager: HalmesDataManager[HalmesRawData]
            with resource_pool.borrow() as (_, _, input_data_manager, _, _):
                ids: list[str] = input_data_manager.find()
                ids.sort()

            more_itertools.consume(executor.lazy_submit_result(
                Main.__run, 
                itertools.repeat(resource_pool),
                sorted(ids),
                more_itertools.repeatfunc(logger.new_log),
                itertools.repeat(log_csv)
            ))
            
        print("Done.")