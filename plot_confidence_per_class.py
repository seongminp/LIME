import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"
# color_list = [CB91_Purple, CB91_Green, CB91_Amber, CB91_Blue, CB91_Pink, CB91_Violet]
color_list = [CB91_Blue, CB91_Pink, CB91_Amber, CB91_Green, CB91_Pink, CB91_Violet]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_list)
plt.rcParams["font.family"] = "Times New Roman"

markers = ["x", "+", "o", "8", "s", "X", "D", "p", "P", "d"]


x_axis = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def get_ax_index(name):
    if name == "entailment":
        return 0
    if name == "nsp":
        return 1
    if name == "rnsp":
        return 2
    if name == "qa":
        return 3
    if name == "qa_article":
        return 3
    if name == "xclass":
        return 4
    if name == "lotclass":
        return 5


def normalize(scores, reference):
    diff = reference - scores[5]
    new_scores = [score + diff for score in scores]
    return new_scores


def get_model_key(dataset, model, threshold, is_soft):
    model_key = f"{dataset}_{model}_{float(threshold)}"
    model_key = f"{model_key}_soft" if is_soft else model_key
    return model_key


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset")
    args = parser.parse_args()

    all_data = {}
    models = ["entailment", "nsp", "rnsp", "qa", "xclass", "lotclass"]
    for model in models:
        data_file = f"data/{args.dataset}/preds_{model}.json"
        with open(data_file) as rf:
            model_data = json.load(rf)
        classes = model_data["classes"]
        data = {c: [] for c in classes}
        for sample in model_data["data"]:
            pred = sample["prediction"]
            confidence = sample["confidence"][pred]
            data[classes[pred]].append(confidence)
        all_data[model] = data
        # data.append((model, all, correct, wrong))

    fig, ax = plt.subplots(1, 6, figsize=(20, 5))
    # fig.suptitle(
    # f"Pseudo-Label Confidence Distribution",
    # fontweight="bold",
    ## pad=30,
    # fontsize=20,
    # )
    # ax.set_ylim([0, 4])
    # ax.set_xlabel("Pseudo label confidence", style="italic", fontsize=20, labelpad=10)
    # ax.set_ylabel(f"Count", style="italic", fontsize=20, labelpad=10)
    # ax.set_xticks(x_axis)
    # ax.set_xticklabels(
    # ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    # )
    # ax.tick_params(axis="y", labelsize=15)
    # ax.tick_params(axis="x", labelsize=15)

    for i, (model, model_preds) in enumerate(all_data.items()):
        to_plot = [np.array(v) for _, v in model_preds.items()]
        ax[i].boxplot(to_plot, showfliers=False)
        ax[i].set_xlabel(model, style="italic", fontsize=15)
        ax[i].set_ylabel("Confidence", style="italic", fontsize=15)
        ax[i].set_title(
            "Confidence Distribution By Class",
            style="italic",
            fontsize=15,
            fontweight="bold",
        )

    # Cleanup.
    # ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # ax.tick_params(
    # axis="both",
    # which="both",
    # bottom="off",
    # top="off",
    # labelbottom="on",
    # left="off",
    # right="off",
    # labelleft="on",
    # size=5,
    # )

    name = f"{args.dataset}_box"
    fig_name = f"{name.replace(' ', '_')}.svg"
    fig.tight_layout()
    fig.savefig(fig_name)
    print(f"Saved as {fig_name}")
