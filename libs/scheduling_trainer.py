from typing import Optional

import torch
from torch.optim.lr_scheduler import LRScheduler

from libs.trainer import Trainer


class LRStepSchedulingTrainer(Trainer):
    lr_scheduler: LRScheduler | None = None

    def run(
        self,
        num_epoch: int,
        device: str,
        model_save_path: str,
        model_version: str = "v1",
        model_load_path: Optional[str] = None,
        verbose: bool = False,
    ) -> tuple[list[float], Optional[list[float]]]:
        train_losses: list[float] = []

        self.model.to(device)

        if model_load_path is not None:
            self.model.load_state_dict(torch.load(model_load_path))

        for epoch in range(num_epoch):
            self.model.train()
            train_loss = 0.0
            for step, (*data, label) in enumerate(self.train_dataloader):
                if data[0].device != device:
                    data = [d.to(device) for d in data]
                    label = label.to(device)
                self.optimizer.zero_grad()
                pred = self.model(*data)
                pred = torch.transpose(pred, 1, 2)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()
                batch_loss = loss.item()
                train_loss += batch_loss
                if verbose and step % 100 == 0:
                    print(
                        f"epoch: {epoch + 1} loss: {batch_loss} step: {step} / {len(self.train_dataloader)}"
                    )
                    torch.save(
                        self.model.state_dict(),
                        model_save_path + f".{model_version}.epoch_{epoch}.step_{step}",
                    )
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            train_loss /= len(self.train_dataloader)
            train_losses.append(train_loss)

            print(f"epoch: {epoch + 1}, train loss: {train_loss}")
            print(f"epoch: {epoch + 1}, train losses: {train_losses}")
            torch.save(
                self.model.state_dict(),
                model_save_path + f".{model_version}.epoch_{epoch}.step_0",
            )
        return train_losses, None
