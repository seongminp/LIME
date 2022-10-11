import argparse

import torch
from torch.utils.data import Dataset


class PseudoDataset(Dataset):
    def __init__(self, data, tokenizer, use_pseudo=True, shuffle=True):

        self.data = data
        self.tokenizer = tokenizer
        self.use_pseudo = use_pseudo
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, index):

        sample = self.data["data"][index]

        label_key = "prediction" if self.use_pseudo else "label"
        label = sample[label_key]
        text = sample["text"]

        input = self.tokenizer(
            text.lower(),
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = input.input_ids.squeeze(0)
        input_mask = input.attention_mask.squeeze(0)
        confidence = torch.tensor(sample["confidence"]) if "confidence" in sample else 0

        return label, confidence, input_ids, input_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument")
    args = parser.parse_args()
