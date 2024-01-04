from torch import nn


class RecursiveDeviceModule(nn.Module):
    def to(self, device: str):
        self._device = device
        for module in self.children():
            module.to(device)


class StrNumOfParamsModule(nn.Module):
    def __str__(self):
        n_params = sum(p.numel() for p in self.parameters())
        return (
            super().__str__()
            + "\n"
            + ("number of parameters: %.2fM" % (n_params / 1e6,))
        )
