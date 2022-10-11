import argparse
import json

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def topic_verbalizer(category):
    hypothesis = f"This text is about {category.lower()}."
    return hypothesis


def location_verbalizer(category):
    hypothesis = f"This happened in {category.lower().capitalize()}."
    return hypothesis


def review_verbalizer(category):
    if category == "good":
        return "I recommend this place."
    elif category == "bad":
        return "I don't recommend this place."
    else:
        raise ValueError("Invalid category:", category)


def clickbait_verbalizer(category, model=None):
    if category == "news":
        return "This is news."
    elif category == "gossip":
        return "This is gossip."
    else:
        raise ValueError("Invalid category:", category)


def emotion_verbalizer(category, model=None):
    if category == "angry":
        return "I'm angry."
    elif category == "happy":
        return "I'm happy."
    elif category == "optimistic":
        if model == "qa":
            return "Looks good."
        else:
            return "I'm fine."
    elif category == "sad":
        return "I'm sad."
    else:
        raise ValueError("Invalid category:", category)


def classify_nsp(model, tokenizer, categories, verbalizer, text):
    hypotheses = [(text, verbalizer(category)) for category in categories]
    tokenized = tokenizer(
        hypotheses,
        truncation="only_first",
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True,
    )
    logits = model(**tokenized.to(model.device))[0]
    nsp_probs = logits[:, 0]
    probs = nsp_probs.softmax(dim=0)
    max_index = torch.argmax(probs)
    category = categories[max_index]
    return max_index, probs.tolist()


def classify_reverse_nsp(model, tokenizer, categories, verbalizer, text):
    hypotheses = [(verbalizer(category, "rnsp"), text) for category in categories]
    tokenized = tokenizer(
        hypotheses,
        truncation="only_second",
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True,
    )
    logits = model(**tokenized.to(model.device))[0]
    nsp_probs = logits[:, 0]
    probs = nsp_probs.softmax(dim=0)
    max_index = torch.argmax(probs)
    category = categories[max_index]
    return max_index, probs.tolist()


def classify_entailment(model, tokenizer, categories, verbalizer, text):
    hypotheses = [(text, verbalizer(category)) for category in categories]
    tokenized = tokenizer(
        hypotheses,
        truncation="only_first",
        padding="longest",
        return_tensors="pt",
    )
    logits = model(**tokenized.to(model.device))[0]
    logits_label_is_true = logits[:, 2]
    max_index = torch.argmax(logits_label_is_true)
    category = categories[max_index]
    probs = torch.softmax(logits_label_is_true, dim=0)
    return max_index, probs.tolist()


def classify_qa(model, tokenizer, categories, verbalizer, text):
    hypotheses = [
        (text, verbalizer(category, "qa"))
        for category in categories
    ]
    tokenized = tokenizer(
        hypotheses,
        truncation="only_first",
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
    )
    input_ids = tokenized.input_ids.unsqueeze(0).to(model.device)
    masks = tokenized.attention_mask.unsqueeze(0).to(model.device)
    logits = model(
        input_ids=input_ids,
        attention_mask=masks,
    )[0]
    logits = logits.squeeze(0)
    max_index = torch.argmax(logits)
    probs = torch.softmax(logits, dim=0)
    return max_index, probs.tolist()


if __name__ == "__main__":

    # Example:
    #
    # python scripts/emnlp22/create_pseudo_labels.py \
    #     -d data/20News/data.json \
    #     -o data/20News/preds_nsp.json \
    #     -v topic \
    #     -m nsp

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data-file", help="Data file")
    parser.add_argument("-o", "--out-file", help="Pseudo label output")
    parser.add_argument("-v", "--verbalizer", help="Verbalizer type")
    parser.add_argument(
        "-m", "--model-type", help="Model type (entailment, nsp, or rnsp)"
    )
    parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=1)
    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=42)
    parser.add_argument("-dv", "--device", help="Cuda device number", default=0)
    args = parser.parse_args()

    if args.model_type == "entailment":
        model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        )
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        classify = classify_entailment
    elif args.model_type == "nsp":
        model = AutoModelForNextSentencePrediction.from_pretrained("bert-large-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
        classify = classify_nsp
    elif args.model_type == "rnsp":
        model = AutoModelForNextSentencePrediction.from_pretrained("bert-large-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
        classify = classify_reverse_nsp
    elif args.model_type == "qa":
        tokenizer = AutoTokenizer.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )
        model = AutoModelForMultipleChoice.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
            # "ehdwns1516/bert-base-uncased_SWAG"
        )
        classify = classify_qa
    else:
        raise ValueError("Wrong model type.")

    model.to(f"cuda:{args.device}")

    if args.verbalizer == "topic":
        verbalizer = topic_verbalizer
    elif args.verbalizer == "location":
        verbalizer = location_verbalizer
    elif args.verbalizer == "review":
        verbalizer = review_verbalizer
    elif args.verbalizer == "clickbait":
        verbalizer = clickbait_verbalizer
    elif args.verbalizer == "emotion":
        verbalizer = emotion_verbalizer
    else:
        raise ValueError("Wrong verbalizer.")

    with open(args.data_file) as rf:
        data = json.load(rf)

    classes = data["classes"]

    print(f"Classes ({len(classes)}): {classes}")

    correct = total = 0
    samples = []
    for i, sample in enumerate(tqdm(data["data"])):
        text = sample["text"]
        prediction, probs = classify(model, tokenizer, classes, verbalizer, text)
        # predicted_class = classes[prediction]
        if prediction.item() == sample["label"]:
            correct += 1
        total += 1
        prob = probs[prediction]
        out_sample = {
            "label": sample["label"],
            "prediction": prediction.item(),
            "confidence": probs,
            "text": sample["text"],
        }
        samples.append(out_sample)
    print(f"Stats for {args.model_type}, {args.out_file}: {correct / total}")

    out_data = {"classes": classes, "data": samples}
    with open(args.out_file, "w") as wf:
        json.dump(out_data, wf, indent=4, ensure_ascii=False)
