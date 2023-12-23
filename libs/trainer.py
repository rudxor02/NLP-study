from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer(BaseModel, ABC):
    model: nn.Module
    train_dataloader: DataLoader
    test_dataloader: Optional[DataLoader] = None
    batch_size: int = 32
    dataset_shuffle: bool = True
    dataset_num_workers: int = 0
    criterion: nn.Module
    optimizer: Optimizer

    class Config:
        arbitrary_types_allowed = True

    @property
    def is_test_enabled(self) -> bool:
        return self.test_dataloader is not None

    @abstractmethod
    def run(
        self,
        num_epoch: int,
        device: str,
        model_save_path: str,
        model_load_path: Optional[str] = None,
        verbose: bool = False,
    ) -> tuple[list[float], Optional[list[float]]]:
        raise NotImplementedError
