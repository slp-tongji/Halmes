import datetime
import gc
from pathlib import Path
from random import Random
from typing import Iterable

import csdir
from li_basic_utilities import CsvFileWriter, StaticClass
from li_halmes_configurations import HalmesTargetLlm
from li_data_loader import DatasetInputProvider, DatasetInputWithId
from li_llm_messages import LlmMessages
from li_logging import TomlLog, TomlLogger
import torch
from statictorch import Tensor1d
import transformers
from li_halmes_model import HalmesFeature, HalmesFeatureExtractor
from li_halmes_data_preparer import HalmesRawData, HalmesDataManager

from li_halmes_generate_data.supported_llm import SupportedLlm


class Main(StaticClass):
    @staticmethod
    def __run(index: str,
              model: SupportedLlm, 
              data: DatasetInputWithId, 
              max_new_tokens: int, 
              output_manager: HalmesDataManager[HalmesRawData], 
              skip_if_exists: bool,
              log: TomlLog,
              log_csv: CsvFileWriter):
        with log:
            log.set_document_value(index, "id")
            log.set_document_value(model.target_id, "model_id")
            log.set_document_value(data.id, "data_id")

            if skip_if_exists and output_manager.get(index)[0]:
                log.set_document_value("skipped", "status")
                log_csv.write_row([index, model.target_id, data.id, log.name, "skipped"])
                return

            LlmMessages.log_messages(data.input, log.create_sub_log("input"))

            def execute() -> None:
                output: SupportedLlm.Output = model.chat(data.input, max_new_tokens)
                features: Iterable[HalmesFeature] = HalmesFeatureExtractor.extract(
                    output["attentions"],
                    output["mm_token_type_ids"]
                )
                output_manager.upsert(index, {
                    "data_id": data.id,
                    "output": Tensor1d(output["sequence"][output["mm_token_type_ids"].size(0):]),
                    "features": list(features)
                })

            try:
                try:
                    execute()
                except torch.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    execute()

                log.set_document_value("succeeded", "status")
                log_csv.write_row([index, model.target_id, data.id, log.name, "succeeded"])
            
                ex: Exception
            except Exception as ex:
                log.set_document_value("failed", "status")
                log.set_document_value(ex, "error")
                log_csv.write_row([index, model.target_id, data.id, log.name, "failed"])

    @staticmethod
    def main(random_seed: str, 
             target: SupportedLlm, 
             model_new_tokens: int,
             output_directory: Path,
             skip_if_exists: bool,
             dataset: DatasetInputProvider) -> None:
        log_directory: Path = csdir.create_directory(
            output_directory / "logs" / target.target_id / 
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"Logs: {log_directory}")

        transformers.set_seed(Random(random_seed).randrange(1000))

        max_new_tokens: int = model_new_tokens

        output: HalmesDataManager[HalmesRawData] = HalmesDataManager(output_directory / target.target_id, False)

        logger: TomlLogger = TomlLogger(log_directory / "logs")
        csv: CsvFileWriter
        with CsvFileWriter(log_directory / "log.csv") as csv:
            csv.write_row(["index", "model", "data", "log", "status"])

            data_count: int = len(list(dataset.ids))
            index_width: int = len(str(data_count - 1))
            index: int
            data: DatasetInputWithId
            for index, data in enumerate(dataset.enumerate()):
                Main.__run(f"{index:0{index_width}}", 
                           target, data, max_new_tokens, 
                           output, skip_if_exists, logger.new_log(), csv)
                
        print("Done.")