import datetime
import itertools
from pathlib import Path
from typing import Callable, Iterable, Mapping

import csdir
from li_basic_utilities import CsvFileWriter, ResourcePool, StaticClass, ThreadPoolOrCurrentThreadExecutor
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLog, TomlLogger
import more_itertools
from sqlite_utils import Database
import sqlite_utils
from sqlite_utils.db import Table, View

from li_halmes_score_inference_outputs.supported_judge import SupportedJudge


class Main(StaticClass):
    @staticmethod
    def __run(resource: ResourcePool[SupportedJudge],
              model: str,
              data: str,
              response: str,
              output_path: Path,
              log: TomlLog,
              log_csv: CsvFileWriter,
              skip_if_exists: bool):
        judge: SupportedJudge
        with (log, resource.borrow() as judge):
            try:
                log.set_document_value(model, "model")
                log.set_document_value(data, "data")
                log.set_document_value(response, "response")

                database_id: str = judge.get_database_id(data)
                database: Database = Database(output_path / f"{database_id}.db")
                log.set_document_value(database_id, "database")

                table: Table | View = database.table(model, pk=("_model", "_data"))
                assert isinstance(table, Table)

                if skip_if_exists:
                    try:
                        _ = table.get((model, data))
                        log.set_document_value("skipped", "status")
                        log_csv.write_row([model, data, log.name, "skipped"])
                        return
                    except sqlite_utils.db.NotFoundError:
                        pass

                judgement: Mapping[str, str | float | int] 
                judgement = judge.judge(data, LlmAssistantMessage(response, None), log.create_sub_log("judge"))
                log.set_document_value(judgement, "judgement")

                judgement = {
                    key: float(judgement[key]) if isinstance(judgement[key], int) else judgement[key]
                    for key in judgement
                }
                if "_model" in judgement:
                    raise RuntimeError("The key '_model' is preserved to indicate the model.")
                if "_data" in judgement:
                    raise RuntimeError("The key '_data' is preserved to indicate the data.")

                judgement["_model"] = model
                judgement["_data"] = data

                table.upsert(judgement)

                log.set_document_value("succeeded", "status")
                log_csv.write_row([model, data, log.name, "succeeded"])

                ex: Exception
            except Exception as ex:
                log.set_document_value("failed", "status")
                log.set_document_value(ex, "error")
                log_csv.write_row([model, data, log.name, "failed"])
    
    @staticmethod
    def __enumerate_input(path: Path) -> Iterable[tuple[str, str, str]]:
        database: Database = Database(path)
        table: Table | View
        for table in database.tables:
            row: dict
            for row in table.rows:
                yield row["model"], row["data"], row["output"]

    @staticmethod
    def main(target: tuple[str, str],
             judge: Callable[[], SupportedJudge],
             input_path: Path,
             output_path: Path,
             skip_if_exists: bool,
             multithreading: int | None):
        output_path = csdir.create_directory(output_path)

        log_directory: Path = csdir.create_directory(
            output_path / "logs" / target[0] / target[1] /
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"Logs: {log_directory}")

        logger: TomlLogger = TomlLogger(log_directory / "logs")
        
        executor: ThreadPoolOrCurrentThreadExecutor
        log_csv: CsvFileWriter
        with (ThreadPoolOrCurrentThreadExecutor(multithreading) as executor,
              CsvFileWriter(log_directory / "log.csv") as log_csv):
            log_csv.write_row(["model", "data", "log", "status"])

            resource_pool: ResourcePool[SupportedJudge] = ResourcePool(
                judge()
                for _ in range(executor.worker_count if executor.worker_count is not None else 1)
            )
            
            input: Iterable[tuple[str, str, str]]
            input = Main.__enumerate_input(input_path / target[0] / target[1] / f"outputs.db")
            input1: Iterable[tuple[str, str, str]]
            input2: Iterable[tuple[str, str, str]]
            input3: Iterable[tuple[str, str, str]]
            input1, input2, input3 = itertools.tee(input, 3)

            output_path = csdir.create_directory(output_path / target[0] / target[1])
            more_itertools.consume(
                executor.lazy_submit_result(Main.__run,
                                            itertools.repeat(resource_pool),
                                            (model for model, _, _ in input1),
                                            (data for _, data, _ in input2),
                                            (response for _, _, response in input3),
                                            itertools.repeat(output_path),
                                            more_itertools.repeatfunc(logger.new_log),
                                            itertools.repeat(log_csv),
                                            itertools.repeat(skip_if_exists))
            )

        print("Done.")
