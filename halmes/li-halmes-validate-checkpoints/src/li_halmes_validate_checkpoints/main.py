import datetime
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import csdir
from li_basic_utilities import CsvFileWriter, StaticClass
from li_halmes_configurations import HalmesTargetLlm
from li_halmes_data_preparer import HalmesDataManager, HalmesDatasetItem
from li_halmes_model import HalmesModel, HalmesModelState
import numpy

import sklearn.metrics
from statictorch import Tensor0d, Tensor2d
import torch
import sklearn


class Main(StaticClass):
    class _Evaluation(NamedTuple):
        auroc: float
        accuracy: float
        recall: float
        pr_auc: float
        fpr: float

    @staticmethod
    @torch.no_grad()
    def __evaluate(
        model: HalmesModel,
        dataset: Iterable[HalmesDatasetItem]
    ):
        truth: list[int] = []
        prediction: list[float] = []

        item: HalmesDatasetItem
        for item in dataset:
            logit: Tensor0d = model.predict_logit(Tensor2d(item["feature"].to(model.device)))
            probability: Tensor0d = Tensor0d(logit.sigmoid())
            prediction.append(float(probability))
            truth.append(int(item["is_hallucination"]))
        
        truth_: numpy.ndarray = numpy.array(truth)
        prediction_: numpy.ndarray = numpy.array(prediction)

        positive: numpy.ndarray = prediction_[truth_ == 1]
        negative: numpy.ndarray = prediction_[truth_ == 0]

        threshold: float = 0.5
        def _():
            nonlocal threshold
            
            thresholds: numpy.ndarray = numpy.linspace(0.1, 0.9, 81)
            best_score: float | numpy.ndarray = -1.0
            current_threshold: float
            for current_threshold in thresholds:
                tp: numpy.ndarray = (positive >= current_threshold).sum()
                tn: numpy.ndarray = (negative < current_threshold).sum()
                score: numpy.ndarray = tp / max(len(positive), 1) + tn / max(len(negative), 1) - 1
                if score > best_score:
                    best_score = score
                    threshold = current_threshold
        _()

        prediction_ = (prediction_ >= threshold).astype(int)
        return Main._Evaluation(float(sklearn.metrics.roc_auc_score(truth, prediction)),
                                float((prediction_ == truth_).mean()),
                                float((positive >= threshold).mean()),
                                float((negative >= threshold).mean()),
                                float(sklearn.metrics.average_precision_score(truth, prediction)))

    @staticmethod
    def main(data: Path, checkpoints: Path, 
             target_id: str, features: HalmesModel.HalmesFeatureSelector, 
             device: torch.device, output: Path):
        log_directory: Path = csdir.create_directory(
            output / "logs" / target_id / features.id /
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"Logs: {log_directory}")

        dataset: Sequence[HalmesDatasetItem] = torch.load(data / target_id / "validation.pt", 
                                                          weights_only=True, 
                                                          map_location=torch.device("cpu"))
        dataset = [
            {
                "feature": features.select(d["feature"]),
                "is_hallucination": d["is_hallucination"]
            }
            for d in dataset
        ]

        input_manager: HalmesDataManager[HalmesModelState] = HalmesDataManager(checkpoints / target_id / features.id, True)
        log: CsvFileWriter
        with CsvFileWriter(log_directory / "log.csv") as log:
            log.write_row(["epoch", "auroc", "accuracy", "recall", "fpr", "pr_auc"])

            best: tuple[str, float] = ("", -1.)

            epoch: str
            for epoch in input_manager.find():
                state: HalmesModelState | None
                _, state = input_manager.get(epoch)
                assert state is not None
                model: HalmesModel = HalmesModel.load(state, device)
                evaluation: Main._Evaluation = Main.__evaluate(model, dataset)

                if evaluation.auroc >= best[1]:
                    best = (epoch, evaluation.auroc)

                log.write_row([epoch, evaluation.auroc, evaluation.accuracy, 
                               evaluation.recall, evaluation.fpr, evaluation.pr_auc])
        
        _, state = input_manager.get(best[0])
        assert state is not None
        torch.save(state, csdir.create_directory(output / target_id / features.id) / "halmes.pt")

        print(f"Done.")
