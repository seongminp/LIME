import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict

from pytorch_lightning.strategies import DDPSpawnStrategy

logging.getLogger("lightning").setLevel(logging.ERROR)

import numpy as np
import pytorch_lightning as pl
import torch
from datamodule import PseudoDataset
from model import BertCategorizerModel as Model
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin, DDPShardedPlugin
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast


def load_model(kwargs, weights=None):

    model_name = "kobert" if kwargs["language"] == "kr" else "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    if weights is not None:
        model = Model.load_from_checkpoint(weights, **kwargs, strict=False)
    else:
        model = Model(**kwargs)
    return model, tokenizer


def get_class_weights(data):
    count = {class_name: 0 for class_name in data["classes"]}
    for sample in data["data"]:
        count[data["classes"][sample["prediction"]]] += 1
    class_counts = torch.tensor([count[class_name] for class_name in data["classes"]])
    class_weights = 1 / class_counts * class_counts.sum()
    class_weights = torch.clamp(torch.tensor(class_weights), 1, 5)
    return class_weights


def train(
    data,
    supervised,
    use_soft_labels,
    confidence_threshold,
    quick,
    batch_size,
    max_epochs,
    learning_rate,
    warmup_ratio,
    weights,
    name,
    language,
    seed,
):
    seed_everything(seed)

    # Initialize trainer.
    grad_batches = 1
    gpus = 0 if quick else -1
    strategy = None if quick else DDPSpawnStrategy(find_unused_parameters=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        mode="max",
        save_weights_only=True,
        save_top_k=-1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # callbacks = [early_stop_callback, checkpoint_callback, lr_monitor]
    callbacks = [checkpoint_callback, lr_monitor]

    version = f"{name}" if not quick else "quick_test"
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=version,
        name="lightning_logs",
    )

    trainer = pl.Trainer(
        gpus=gpus,
        strategy=strategy,
        callbacks=callbacks,
        # plugins=plugins,
        log_every_n_steps=grad_batches,
        logger=logger,
        accumulate_grad_batches=grad_batches,
        deterministic=True,
        # max_epochs=max_epochs,
        max_epochs=1,
        # precision=16,
    )

    # Calculate class imbalance:
    num_classes = len(data["classes"])
    if supervised:
        filtered = data["data"]
    else:
        # class_to_index = {class_name: i for i, class_name in enumerate(data["classes"])}
        filtered = [
            sample
            for sample in data["data"]
            if sample["confidence"][sample["prediction"]] >= confidence_threshold
        ]
    data["data"] = filtered
    # class_weights = get_class_weights(data)
    class_weights = None

    training_steps = len(data["data"]) // grad_batches // trainer.devices * max_epochs
    # warmup_steps = int(training_steps * 0.01)
    warmup_steps = int(training_steps * warmup_ratio)
    print("============================================")
    print(f"Training {name}")
    print("===============Dataset Stats================")
    print("Number of classes:", num_classes)
    print("Class weights:", class_weights)
    print("==============Training Stats================")
    print("Train dataloader size:", len(data["data"]))
    print("Supervised training:", supervised)
    print("Soft labels:", use_soft_labels)
    print("Learning rate:", learning_rate)
    print("Warmup ratio:", warmup_ratio)
    print("Max epochs:", max_epochs)
    print("Training steps:", training_steps)
    print("Warmup steps:", warmup_steps)
    print("============================================")
    # class_weights = None

    model_kwargs = {
        "num_classes": num_classes,
        "use_soft_labels": use_soft_labels,
        "language": language,
        "class_weights": class_weights,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "training_steps": training_steps,
    }
    model, tokenizer = load_model(model_kwargs, weights)

    cpu_count = os.cpu_count()
    train_set = PseudoDataset(data, tokenizer, use_pseudo=not supervised)
    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=cpu_count // 4,
    )
    # validate_dataloader = DataLoader(
    # validate_set, batch_size=batch_size, num_workers=cpu_count // 4
    # )

    # Run training.
    print(f"Training {version}")

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        # val_dataloaders=validate_dataloader,
    )

    best_model_path = checkpoint_callback.best_model_path
    if trainer.is_global_zero:
        print(f"Best model checkpoint: {best_model_path}")
        shutil.move(best_model_path, f"models/{version}.bin")
        print(f"Best model saved as", f"models/{version}.bin")
        print("SAVED, PID", os.getpid())

        return best_model_path
    else:
        # print("Exiting", os.getpid())
        sys.exit(0)
    time.sleep(3)
    # model = Model.load_from_checkpoint(
    # best_model_path, num_classes=num_classes, language=language
    # )

    return best_model_path


def predict(data, weights, batch_size, language):

    num_classes = len(data["classes"])

    model_kwargs = {"num_classes": num_classes, "language": language}
    model, tokenizer = load_model(model_kwargs, weights)

    # Load dataset.
    cpu_count = os.cpu_count()
    test_set = PseudoDataset(data, tokenizer, use_pseudo=False, shuffle=False)
    test_dataloader = DataLoader(
        test_set, batch_size=batch_size, num_workers=1, shuffle=False
    )

    # Initialize trainer.
    # gpus = 0 if quick else -1
    gpus = -1
    strategy = "dp"

    trainer = pl.Trainer(
        gpus=gpus,
        strategy=strategy,
        # plugins=plugins,
        deterministic=True,
        # precision=16,
    )

    # Run predictions.
    out = trainer.predict(model, dataloaders=test_dataloader)

    # indices = torch.cat([i for i, _ in out])
    preds = torch.cat([p for _, p in out]).tolist()
    # preds = [data["classes"][pred] for pred in preds]
    # For DDP. When DP is finally discontinued.
    # if trainer.is_global_zero:
    # world_size = torch.distributed.get_world_size()
    # gathered_indices = [indices.clone() for _ in range(world_size)]
    # gathered_preds = [preds.clone() for _ in range(world_size)]
    # else:
    # gathered_indices = None
    # gathered_preds = None
    # torch.distributed.gather(indices, gather_list=gathered_indices)
    # torch.distributed.gather(preds, gather_list=gathered_preds)
    # if trainer.is_global_zero:
    # print(gathered_indices.shape)
    # print(gathered_preds.shape)

    return preds


if __name__ == "__main__":
    # Example:
    # python scripts/emnlp22/train.py \
    #     -t \
    #     -df data/AGNews/preds_xclass.json \
    #     -c 0.5 \
    #     -bs 256

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-df",
        "--data-file",
        help="File with text data",
    )
    parser.add_argument(
        "-sp",
        "--supervised",
        action="store_true",
        help="Use ground truth label if True (supervised training)",
    )
    parser.add_argument(
        "-sl",
        "--use-soft-labels",
        action="store_true",
        help="Use soft labels in cross entropy if true",
    )
    parser.add_argument(
        "-c",
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Use only data above this threshold (inclusive)",
    )
    parser.add_argument(
        "-t", "--do-train", default=False, action="store_true", help="Run training"
    )
    parser.add_argument(
        "-q", "--quick", default=False, action="store_true", help="Do a quick test run"
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-me", "--max-epochs", default=5, type=int, help="Max epochs")
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate"
    )
    parser.add_argument(
        "-wr", "--warmup-ratio", default=0.05, type=float, help="Scheduler ratio"
    )
    parser.add_argument("-w", "--weights", default=None, help="Pretrained weights")
    parser.add_argument("-n", "--name", default="classifier", help="Name of experiment")
    parser.add_argument(
        "-l",
        "--language",
        default="en",
        help="en or kr",
    )
    parser.add_argument("-s", "--seed", default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.data_file) as rf:
        data = json.load(rf)

    if args.do_train:
        best_model_path = train(
            data,
            args.supervised,
            args.use_soft_labels,
            args.confidence_threshold,
            args.quick,
            args.batch_size,
            args.max_epochs,
            args.learning_rate,
            args.warmup_ratio,
            args.weights,
            args.name,
            args.language,
            args.seed,
        )
    # model_path = best_model_path if args.do_train else
    else:
        predict(
            data,
            args.weights,
            args.batch_size,
            args.language,
        )
