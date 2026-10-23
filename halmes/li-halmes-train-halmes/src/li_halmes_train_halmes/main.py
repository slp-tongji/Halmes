import datetime
import math
from pathlib import Path
from random import Random
from typing import Callable, Iterable, Sequence

import csdir
from li_basic_utilities import CsvFileWriter, StaticClass
from li_halmes_configurations import HalmesTargetLlm
from li_halmes_data_preparer import HalmesDataManager, HalmesDatasetItem
from li_halmes_model import HalmesModel, HalmesFocalBceLoss, HalmesModelState
import more_itertools

from statictorch import Tensor0d, Tensor2d
import torch


class Main(StaticClass):
    @staticmethod
    def __take_tail(features: Tensor2d, ratio: float = 0.7) -> Tensor2d:
        length: int = features.shape[0]
        if ratio >= 1 or length <= 1:
            return features
        length = math.ceil(length * ratio)
        return Tensor2d(features[-length:])

    @staticmethod
    def __train(
        model: HalmesModel,
        dataset: Sequence[HalmesDatasetItem],
        random: Random,
        total_epochs: int = 30,
        warmup_epochs: int = 5,
        batch_size: int = 32,
        grad_clip: float = 0.5,
    ):
        negative_data: Iterable[HalmesDatasetItem]
        positive_data: Iterable[HalmesDatasetItem]
        negative_data, positive_data = more_itertools.partition(lambda x: x["is_hallucination"], dataset)
        negative_data = list(negative_data)
        positive_data = list(positive_data)

        loss: HalmesFocalBceLoss = HalmesFocalBceLoss().to(model.device)

        optimizer: torch.optim.AdamW = torch.optim.AdamW(
            model.get_parameters(), lr=3e-4, weight_decay=1e-2,
        )

        def lr_lambda(epoch: int):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress: float = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, grad_clip)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler: torch.optim.lr_scheduler.LambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        model.update_prior_values([d["feature"] for d in dataset])

        batch_per_epoch: int = len(dataset) // batch_size
        _: int
        for _ in range(total_epochs):
            losses: list[float] = []

            for _ in range(batch_per_epoch):
                result: list[torch.Tensor] = []

                for _ in range(batch_size):
                    sample: HalmesDatasetItem = random.choice(random.choice([positive_data, negative_data]))
                    logit: Tensor0d = model.predict_logit(Main.__take_tail(
                        Tensor2d(sample["feature"].to(model.device))
                    ))
                    result.append(loss(logit, sample["is_hallucination"]))

                assert not isinstance(result, int)

                loss_value: torch.Tensor = torch.stack(result, dim=0).mean()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                losses.append(float(loss_value.detach()))
            scheduler.step()

            yield sum(losses) / len(losses)

    @staticmethod
    def main(input: Path, target_id: str, 
             random_seed: str, device: torch.device, 
             output: Path, features: HalmesModel.HalmesFeatureSelector):
        log_directory: Path = csdir.create_directory(
            output / "logs" / target_id / features.id /
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"Logs: {log_directory}")

        dataset: Sequence[HalmesDatasetItem] = torch.load(input / target_id / "train.pt", 
                                                          weights_only=True, 
                                                          map_location=torch.device("cpu"))
        dataset = [
            {
                "feature": features.select(d["feature"]),
                "is_hallucination": d["is_hallucination"]
            }
            for d in dataset
        ]

        model: HalmesModel = HalmesModel(device, dataset[0]["feature"].size(1))

        random: Random = Random(random_seed)
        torch.manual_seed(random.randrange(1000))
        torch.cuda.manual_seed_all(random.randrange(1000))

        output_manager: HalmesDataManager[HalmesModelState] = HalmesDataManager(output / target_id / features.id, False)
        log: CsvFileWriter
        with CsvFileWriter(log_directory / "log.csv") as log:
            log.write_row(["epoch", "loss"])
            epoch: int
            loss: float
            for epoch, loss in enumerate(Main.__train(model, dataset, random)):
                output_manager.upsert(str(epoch), model.export())
                log.write_row([epoch, loss])

        print(f"Done.")
