import datetime
import gc
from pathlib import Path
from typing import Callable, Iterable, Sequence

import PIL
import PIL.Image
import csdir
from li_basic_utilities import CsvFileWriter, StaticClass
from li_data_loader import DatasetInputProvider, DatasetInputWithId
from li_image_operating import InMemoryImage
from li_halmes_model import HalmesModel, HalmesModelState
from li_halmes_configurations import HalmesTargetLlm
from li_llm_messages import LlmAssistantMessage, LlmMessage, LlmMessages
from li_logging import TomlLog, TomlLogger
from sqlite_utils import Database
import sqlite_utils
from sqlite_utils.db import Table, View
import torch

from li_halmes_inference.supported_llm import SupportedLlm
from li_halmes_inference.supported_llm_provider import SupportedLlmProvider


class Main(StaticClass):
    @staticmethod
    def __resize_image(image: InMemoryImage):
        image_: PIL.Image.Image = image.pillow
        image_.thumbnail((512, 512))
        return InMemoryImage.from_pillow(image_)
    
    @staticmethod
    def __skip(skip_if_exists: bool,
               table: Table, 
               model: SupportedLlm, 
               data: DatasetInputWithId):
        if not skip_if_exists:
            return False

        try:
            return not table.get((model.variant_id, data.id))["image_resized"]
        except sqlite_utils.db.NotFoundError:
            return False

    @staticmethod
    def __run(model: SupportedLlm | Exception, 
              data: DatasetInputWithId, 
              max_new_tokens: int, 
              table: Table | None, 
              skip_if_exists: bool,
              log: TomlLog,
              log_csv: CsvFileWriter):
        with log:
            if isinstance(model, Exception):
                log.set_document_value("failed", "status")
                log.set_document_value(model, "error")
                log_csv.write_row(["", data.id, log.name, "failed"])
                return

            assert table is not None
            try:
                log.set_document_value(model.variant_id, "model")
                log.set_document_value(data.id, "data")

                if Main.__skip(skip_if_exists, table, model, data):
                    log.set_document_value("skipped", "status")
                    log_csv.write_row([model.variant_id, data.id, log.name, "skipped"])
                    return
                
                LlmMessages.log_messages(data.input, log.create_sub_log("input"))

                resized: bool = False
                try:
                    try:
                        response: LlmAssistantMessage = model.chat(data.input, max_new_tokens)
                    except torch.OutOfMemoryError:
                        gc.collect()
                        torch.cuda.empty_cache()
                        response = model.chat(data.input, max_new_tokens)
                except torch.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    response = model.chat(LlmMessages.modify_images(data.input, Main.__resize_image), max_new_tokens)
                    resized = True

                log.set_document_value(resized, "image_resized")
                log.set_document_value(response.content, "response")

                table.upsert({
                    "model": model.variant_id,
                    "data": data.id,
                    "output": response.content,
                    "image_resized": resized
                })

                log.set_document_value("succeeded", "status")
                log_csv.write_row([model.variant_id, data.id, log.name, "succeeded"])

                ex: Exception
            except Exception as ex:
                log.set_document_value("failed", "status")
                log.set_document_value(ex, "error")
                log_csv.write_row([model.variant_id, data.id, log.name, "failed"])


    @staticmethod
    def main(target: HalmesTargetLlm,
             features: HalmesModel.HalmesFeatureSelector,
             llms: Iterable[SupportedLlmProvider],
             data: DatasetInputProvider,
             input_path: Path,
             output_path: Path,
             skip_if_exists: bool,
             max_new_tokens: int):
        output_path = csdir.create_directory(output_path)

        log_directory: Path = csdir.create_directory(
            output_path / "logs" / target["id"] / features.id /
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"Logs: {log_directory}")

        logger: TomlLogger = TomlLogger(log_directory / "logs")
        
        cpu: torch.device = torch.device("cpu")
        input_path = input_path / target["id"] / features.id / "halmes.pt"
        def halmes(device: torch.device):
            state: HalmesModelState = torch.load(input_path, 
                                                 map_location=cpu, 
                                                 weights_only=True)
            return HalmesModel.load(state, device)

        output: Database = Database(csdir.create_directory(output_path / target["id"] / features.id) / f"outputs.db")

        log_csv: CsvFileWriter
        with CsvFileWriter(log_directory / "log.csv") as log_csv:
            log_csv.write_row(["model", "data", "log", "status"])

            llm_factory: SupportedLlmProvider
            for llm_factory in llms:
                try:
                    model: SupportedLlm | Exception = llm_factory(target, features, halmes)
                    table: Table | View | None = output.table(model.variant_id, pk=("model", "data"))
                    assert isinstance(table, Table)
                    ex: Exception
                except Exception as ex:
                    model = ex
                    table = None

                input: DatasetInputWithId
                for input in data.enumerate():
                    Main.__run(model, input, max_new_tokens, table, skip_if_exists, logger.new_log(), log_csv)

                del model
                gc.collect()
                torch.cuda.empty_cache()

        print("Done.")
