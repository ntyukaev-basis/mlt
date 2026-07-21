import argparse
import os
import time

import air_dataset
import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки (конфигурация обучения)."""
    p = argparse.ArgumentParser(description="MLT-05 PyTorch Lightning training")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Каталог примонтированного датасета MLT-03 (внутри — папки версий с wine.csv)",
    )
    p.add_argument(
        "--checkpoints-dir",
        default="/checkpoints",
        help="Каталог для last.ckpt (том air-checkpoints; по умолчанию /checkpoints)",
    )
    p.add_argument(
        "--storage-uri",
        default=os.environ.get("AIR_DATASET_STORAGE_URI"),
        help="Расположение отслеживаемой версии по данным платформы; последний "
        "сегмент пути — имя папки версии (по умолчанию $AIR_DATASET_STORAGE_URI)",
    )
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--epoch-delay",
        type=float,
        default=0.0,
        help="Пауза (сек) после каждой эпохи — растягивает обучение, чтобы "
        "успеть выполнить Suspend в середине",
    )
    return p.parse_args()


class EpochProgress(Callback):
    """Печатает номер эпохи в stdout.

    В поде нет TTY, а прогресс-бар выключен (``enable_progress_bar=False``,
    иначе tqdm заспамит лог ``\\r``-переводами) — поэтому без явного print
    в логах не видно, на какой эпохе идёт обучение. Печатаем на входе и
    выходе каждой эпохи, с ``flush=True`` (иначе строки застрянут в буфере
    до конца процесса).
    """

    def on_train_epoch_start(self, trainer, pl_module):
        """Старт эпохи — сразу видно, что обучение зашло в новую эпоху."""
        print(
            f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        """Конец эпохи — печатаем номер и текущий train_loss (из autolog-метрик)."""
        loss = trainer.callback_metrics.get("train_loss")
        loss_s = f" train_loss={float(loss):.4f}" if loss is not None else ""
        print(
            f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs} done{loss_s}",
            flush=True,
        )


class EpochDelay(Callback):
    """Пауза после каждой эпохи — даёт время на Suspend в демо MLT-05."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def on_train_epoch_end(self, trainer, pl_module):
        """Спит заданное число секунд по завершении эпохи."""
        if self.seconds > 0:
            time.sleep(self.seconds)


def load_wine(
    data_dir: str, storage_uri: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Читает нужную версию wine.csv и готовит тензоры (X стандартизован).

    Выбор папки версии — в ``air_dataset``: сортировка имён здесь давала не
    «последнюю», а «наибольшую строку», и на стенде это молча уводило обучение
    на версию с другой схемой.
    """
    path = air_dataset.resolve_csv(data_dir, "wine.csv", storage_uri)
    print(f"training on: {path}", flush=True)
    df = pd.read_csv(path)
    air_dataset.require_columns(df, ["quality"], path)

    y = (df["quality"] >= 6).astype("int64").to_numpy()
    x = df.drop(columns=["quality"]).astype("float32")
    # Стандартизация признаков — иначе MLP на «сырых» шкалах не сходится.
    x = (x - x.mean()) / (x.std().replace(0.0, 1.0))
    return (
        torch.tensor(x.to_numpy(), dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


class WineMLP(pl.LightningModule):
    """Двухслойный MLP: бинарная классификация quality >= 6."""

    def __init__(self, in_features: int, hidden: int, lr: float):
        super().__init__()
        # save_hyperparameters() кладёт гиперпараметры в чекпоинт и делает их
        # видимыми для autolog — своих mlflow-вызовов не требуется.
        self.save_hyperparameters()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        # self.log — это Lightning, НЕ mlflow.log_*; autolog подхватит train_loss.
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    x, y = load_wine(args.data_dir, args.storage_uri)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=args.batch_size,
        shuffle=True,
    )

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    last_ckpt = os.path.join(args.checkpoints_dir, "last.ckpt")

    # Чекпоинт каждую эпоху, чтобы Suspend В СЕРЕДИНЕ обучения оставлял
    # last.ckpt для resume.
    #
    # ВАЖНО: ``save_top_k=0`` тут НЕЛЬЗЯ — при нём Lightning пишет last.ckpt
    # ТОЛЬКО по завершении ``fit()`` (проверено на 2.6: после эпох 1/2/3
    # файла нет, появляется лишь в конце). Тогда прерывание в середине не
    # оставляет чекпоинта и resume стартует с эпохи 1. ``save_top_k=1``
    # заставляет писать чекпоинт (и обновлять last.ckpt) каждую эпоху;
    # держится один ротируемый ``epoch=*.ckpt`` + last.ckpt.
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        save_last=True,
        every_n_epochs=1,
        save_top_k=1,
    )

    # Авто-resume: если чекпоинт уже есть — продолжаем с него.
    resume_from = last_ckpt if os.path.exists(last_ckpt) else None
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        # epoch в чекпоинте — число уже завершённых эпох.
        done = int(ckpt.get("epoch", 0))
        print(f"resumed from epoch {done}", flush=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, EpochProgress(), EpochDelay(args.epoch_delay)],
        enable_progress_bar=False,
        log_every_n_steps=1,
        accelerator="auto",
        devices=1,
    )
    trainer.fit(
        model=WineMLP(in_features=x.shape[1], hidden=args.hidden, lr=args.lr),
        train_dataloaders=loader,
        ckpt_path=resume_from,
    )

    print(f"training finished at epoch {trainer.current_epoch}", flush=True)


if __name__ == "__main__":
    main()
