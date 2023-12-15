from abc import ABC, abstractmethod
from typing import Optional

import torch
from pydantic import BaseModel
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset


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
        self, num_epoch: int, model_path: str
    ) -> tuple[list[float], Optional[list[float]]]:
        pass


class ClassificationTrainer(Trainer):
    def run(
        self,
        num_epoch: int,
        device: str,
        model_save_path: str,
        model_load_path: Optional[str] = None,
        verbose: bool = False,
    ) -> tuple[list[float], Optional[list[float]]]:
        train_losses: list[float] = []
        test_losses: Optional[list[float]] = [] if self.is_test_enabled else None

        if model_load_path is not None:
            self.model.load_state_dict(torch.load(model_load_path))

        for epoch in range(num_epoch):
            self.model.train()
            train_loss = 0.0
            num_correct = 0
            for i, (x_data, y_data) in enumerate(self.train_dataloader):
                x_data, y_data = x_data.to(device), y_data.to(device)
                self.optimizer.zero_grad()
                y_pred = self.model(x_data)
                loss = self.criterion(y_pred, y_data)
                loss.backward()
                self.optimizer.step()
                batch_loss = loss.item()
                train_loss += batch_loss
                num_correct += (y_pred.argmax(1) == y_data.argmax(1)).sum().item()
                if verbose and i % 100 == 0:
                    print(
                        f"epoch: {epoch + 1} loss: {batch_loss} {i} / {len(self.train_dataloader)}"
                    )
            train_loss /= len(self.train_dataloader)
            train_losses.append(train_loss)
            train_acc = num_correct / len(self.train_dataloader.dataset)

            print(
                f"epoch: {epoch + 1}, train loss: {train_loss}, train acc: {train_acc}"
            )
            if self.is_test_enabled:
                self.model.eval()
                test_loss = 0.0
                num_correct = 0
                with torch.no_grad():
                    for x_data, y_data in self.test_dataloader:
                        x_data, y_data = x_data.to(device), y_data.to(device)
                        y_pred = self.model(x_data)
                        loss = self.criterion(y_pred, y_data)
                        test_loss += loss.item()
                        num_correct += (
                            (y_pred.argmax(1) == y_data.argmax(1)).sum().item()
                        )
                test_loss /= len(self.test_dataloader)
                test_losses.append(test_loss)
                test_acc = num_correct / len(self.test_dataloader.dataset)

                print(
                    f"epoch: {epoch + 1}, test loss: {test_loss}, test acc: {test_acc}"
                )
            torch.save(self.model.state_dict(), model_save_path + f".{epoch + 1}")
        return train_losses, test_losses
