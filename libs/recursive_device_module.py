from torch import nn


class RecursiveDeviceModule(nn.Module):
    def to(self, device: str):
        self._device = device
        for module in self.children():
            module.to(device)
