from torch import nn


class RecursiveDeviceModule(nn.Module):
    def to(self, device: str):
        self._device = device
        for module in self.children():
            module.to(device)


class StrNumOfParamsModule(nn.Module):
    def __str__(self):
        n_params = sum(p.numel() for p in self.parameters())
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print()
        return (
            super().__str__()
            + "\n"
            + ("number of parameters: %.2fM" % (n_params / 1e6,))
            + "\n"
            + "model size: {:.3f}MB".format(size_all_mb)
        )
