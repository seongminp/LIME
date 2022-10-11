import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from plot_constants import dataset_name, model_name
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

params = {'mathtext.default': 'regular' } 
plt.rcParams.update(params)

markers = ["x", "+", "o", "8", "s", "X", "D", "p", "P", "d"]


x_axis = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def get_ax_index(name):
    if name == "entailment":
        return 0
    if name == "nsp":
        return 1
    if name == "rnsp":
        return 1
    if name == "qa":
        return 2
    if name == "qa_article":
        return 3
    if name == "qa_what":
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
    parser.add_argument("-d", "--data-directory", help="Data directory")
    # parser.add_argument("-ds", "--datasets", help="Dataset names, separated with a comma")
    args = parser.parse_args()

    data = []
    # datasets = [d.strip() for d in args.datasets.split(",")]
    datasets = ["agnews", "yahoo", "dbpedia", "clickbait"]
    # models = ["entailment", "nsp", "rnsp", "qa", "xclass", "lotclass"]
    models = ["entailment", "rnsp", "qa", "xclass", "lotclass"]
    # models = ["entailment", "rnsp", "qa", "qa_what"]
    difficulty = {}
    raw_corrects = {}
    raw_confidences = {}
    for dataset in datasets:

        data_file = f"data/{dataset}/preds_entailment.json"
        with open(data_file) as rf:
            dataset_data = json.load(rf)

        dataset_size = len(dataset_data["data"])
        difficulty[dataset] = [0] * dataset_size
        raw_corrects[dataset] = {model: [False] * dataset_size for model in models}
        raw_confidences[dataset] = {model: [None] * dataset_size for model in models}

        for model in models:

            data_file = f"data/{dataset}/preds_{model}.json"
            with open(data_file) as rf:
                model_data = json.load(rf)
            for i, sample in enumerate(model_data["data"]):
                pred = sample["prediction"]
                label = sample["label"]
                c = sample["confidence"][pred]
                if label == pred:
                    difficulty[dataset][i] += 1
                    raw_corrects[dataset][model][i] = True
                raw_confidences[dataset][model][i] = c

    difficulty_counts = {dataset: [] for dataset in datasets}
    corrects = {dataset: {} for dataset in datasets}
    confidences = {dataset: {} for dataset in datasets}
    for dataset in datasets:
        for model in models:
            difficulty_counts[dataset] = [0 for _ in range(len(models) + 1)]
            corrects[dataset][model] = [0 for _ in range(len(models) + 1)]
            confidences[dataset][model] = [[] for _ in range(len(models) + 1)]
            for d, cor, conf in zip(
                difficulty[dataset],
                raw_corrects[dataset][model],
                raw_confidences[dataset][model],
            ):
                diff = len(models) - d
                difficulty_counts[dataset][diff] += 1
                if cor:
                    corrects[dataset][model][diff] += 1
                confidences[dataset][model][diff].append(conf)
            for d in range(len(models) + 1):
                confidence_list = confidences[dataset][model][d]
                if confidence_list:
                    confidences[dataset][model][d] = sum(confidence_list) / len(
                        confidence_list
                    )
                else:
                    confidences[dataset][model][d] = 0

    #fig, ax = plt.subplots(2, 4, figsize=(12, 5))
    fig = plt.figure(constrained_layout=True, figsize=(12, 7))
    fig.set_constrained_layout_pads(hspace=0.1)
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle("(a) Confidence", fontsize=14)
    subfigs[1].suptitle("(b) Percentage correct", fontsize=14)
    ax1 = subfigs[0].subplots(1, len(datasets))
    ax2 = subfigs[1].subplots(1, len(datasets))
    #fig.text(0.5,0.5, "(a) Confidence", ha="center", va="bottom", fontsize=14)
    #fig.text(0.5,0.0, "(b) Percentage correct", ha="center", va="bottom", fontsize=14)
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

    for i, dataset in enumerate(datasets):

        models_reversed = models[::-1]

        cor = [
            np.array(corrects[dataset][model]) / difficulty_counts[dataset]
            for model in models_reversed
        ]
        cor = np.array(cor)

        conf = [np.array(confidences[dataset][model]) for model in models_reversed]
        conf = np.array(conf)

        col = i  # get_ax_index(name)
        # row = datasets.index(dataset)
        ax1[col].set_xlabel(f"Difficulty", style="italic", fontsize=13, labelpad=13)
        #ax[0, col].set_ylabel("Confidence", style="italic", fontsize=15)
        ax1[col].set_title(dataset_name[dataset], style="italic", fontsize=15, fontweight="bold")
        ax1[col].set_xticks(np.arange(conf.shape[1]) + 0.5, minor=False)
        ax1[col].set_yticks(np.arange(conf.shape[0]) + 0.5, minor=False)
        ax1[col].set_xticklabels(np.arange(conf.shape[0]+1))
        ax1[col].set_yticklabels([model_name[m] for m in models_reversed])
        c = ax1[col].pcolormesh(conf, cmap="YlOrRd")
        fig.colorbar(c, ax=ax1[col], location="bottom")

        # ax[row, col].set_xlabel(f"Confidence", style="italic", fontsize=15)
        # ax[row, col].set_ylabel("Count", style="italic", fontsize=15)
        # ax[row, col].set_title(name, style="italic", fontsize=15, fontweight="bold")
        # ax[row, col].set_xticks(x_axis)
        # ax[row, col].set_xticklabels(
        # ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        # )
        # ax[row, col].hist(wrong, x_axis, color=CB91_Pink)
        ax2[col].set_xlabel(f"Difficulty", style="italic", fontsize=14, labelpad=13)
        #ax[1, col].set_ylabel("Percentage correct", style="italic", fontsize=15)
        ax2[col].set_title(dataset_name[dataset], style="italic", fontsize=15, fontweight="bold")
        ax2[col].set_xticks(np.arange(cor.shape[1]) + 0.5, minor=False)
        ax2[col].set_yticks(np.arange(cor.shape[0]) + 0.5, minor=False)
        ax2[col].set_xticklabels(np.arange(cor.shape[0]+1))
        #ax[1, col].set_yticklabels(models_reversed)
        ax2[col].set_yticklabels([model_name[m] for m in models_reversed])
        c = ax2[col].pcolormesh(cor, cmap="YlOrRd")
        fig.colorbar(c, ax=ax2[col], location="bottom")

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

    name = f"heatmap"
    fig_name = f"{name.replace(' ', '_')}.svg"
    #fig.tight_layout()
    # Cleanup.
    fig.savefig(fig_name)
    print(f"Saved as {fig_name}")
