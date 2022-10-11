import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch import optim
from transformers import BertForSequenceClassification


class BertCategorizerModel(LightningModule):
    def __init__(
        self,
        num_classes,
        use_soft_labels=False,
        language="en",
        class_weights=None,
        warmup_steps=0,
        training_steps=0,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.use_soft_labels = use_soft_labels
        self.language = language
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        model_name = "kobart" if language == "kr" else "bert-base-uncased"
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

        self.pad_token_id = self.model.config.pad_token_id
        self.ce_loss = nn.CrossEntropyLoss()
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

    def forward(self, input_ids, input_mask):
        x = self.model(input_ids=input_ids, attention_mask=input_mask)
        return x

    def run_batch(self, batch, batch_idx, predicting=False):
        label, confidence, input_ids, input_mask = batch
        target = confidence if self.use_soft_labels else label

        out = self(input_ids, input_mask).logits
        if not predicting:
            loss = self.ce_loss(out.view(-1, self.num_classes), target)
        else:
            loss = None
        return loss, out

    def training_step(self, batch, batch_idx):
        loss, _ = self.run_batch(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.run_batch(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, out = self.run_batch(batch, batch_idx)
        self.log("test_loss", loss)
        return loss, indices, preds

    def predict_step(self, batch, batch_idx):
        _, out = self.run_batch(batch, batch_idx, predicting=True)
        indices = batch[0]
        preds = out.argmax(dim=-1)
        return indices, preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument")
    args = parser.parse_args()
