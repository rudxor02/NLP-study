from pydantic import BaseModel
from torch import cuda, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from libs.scheduling_trainer import LRStepSchedulingTrainer
from week3.model import Transformer
from week3.vocab import WMT14Dataset, load_data, load_tokenizer


class TransformerConfig(BaseModel):
    seq_len: int = 50
    num_epochs: int = 6
    device: str = "cuda:0" if cuda.is_available() else "cpu"
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    p_dropout: float = 0.1
    d_ff: int = 2048
    betas: tuple[float, float] = (0.9, 0.98)
    eps_adam: float = 1e-9
    eps_label_smoothing: float = 0.1
    warmup_steps: int = 4000

    @property
    def batch_size(self):
        # return 25000 // self.seq_len
        return 300


config = TransformerConfig()


class TransformerScheduler(LRScheduler):
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        """
        lr updates per step, not per epoch
        """
        return (
            [
                self.d_model ** (-0.5)
                * min(
                    self.last_epoch ** (-0.5),
                    self.last_epoch * self.warmup_steps ** (-1.5),
                )
                for _ in self.base_lrs
            ]
            if self.last_epoch != 0
            else self.base_lrs
        )


#
def train():
    tokenizer = load_tokenizer()
    train_en, train_de = load_data(train=True)
    dataset = WMT14Dataset(
        tokenizer,
        train_en,
        train_de,
        seq_length=config.seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    padding_idx = dataset.pad

    model = Transformer(
        padding_idx=padding_idx,
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=config.seq_len,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        p_dropout=config.p_dropout,
        num_layers=config.num_layers,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.eps_label_smoothing)

    optimizer = Adam(
        model.parameters(), betas=config.betas, eps=config.eps_adam, lr=1e-7
    )
    scheduler = TransformerScheduler(optimizer, config.d_model, config.warmup_steps)

    trainer = LRStepSchedulingTrainer(
        model=model,
        train_dataloader=dataloader,
        batch_size=config.batch_size,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    print(model)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))

    trainer.run(
        num_epoch=config.num_epochs,
        device=config.device,
        model_save_path="week3/data/transformer_model",
        verbose=True,
    )


if __name__ == "__main__":
    train()
