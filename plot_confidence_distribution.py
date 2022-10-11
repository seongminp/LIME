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

from plot_constants import model_name
params = {'mathtext.default': 'regular' } 
plt.rcParams.update(params)


def get_ax_index(name):
    if name == "entailment":
        return 2
    if name == "nsp":
        return 1
    if name == "rnsp":
        return 3
    if name == "qa":
        return 4
    if name == "qa_article":
        return 2
    if name == "qa_what":
        return 2
    if name == "xclass":
        return 1
    if name == "lotclass":
        return 0


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
    parser.add_argument(
        "-ds", "--datasets", help="Dataset names, separated with a comma"
    )
    args = parser.parse_args()

    data = []
    datasets = [d.strip() for d in args.datasets.split(",")]
    # models = ["entailment", "nsp", "rnsp", "qa", "xclass", "lotclass"]
    models = ["entailment", "rnsp", "qa", "xclass", "lotclass"]
    for dataset in datasets:
        for model in models:
            data_file = f"data/{dataset}/preds_{model}.json"
            with open(data_file) as rf:
                model_data = json.load(rf)
            all_confidences = []
            correct_confidences = []
            wrong_confidences = []
            for sample in model_data["data"]:
                pred = sample["prediction"]
                label = sample["label"]
                c = sample["confidence"][pred]
                if label == pred:
                    correct_confidences.append(c)
                else:
                    wrong_confidences.append(c)
                all_confidences.append(c)

                # for i, c in enumerate(sample["confidence"]):
                # if i == pred:
                # correct_confidences.append(c)
                # else:
                # wrong_confidences.append(c)
                # all_confidences.append(c)

            correct = np.array(correct_confidences)
            wrong = np.array(wrong_confidences)
            all = np.array(all_confidences)
            data.append((dataset, model, all, correct, wrong))

    fig = plt.figure(constrained_layout=True, figsize=(15, 5))
    fig.set_constrained_layout_pads(hspace=0.1)
    #fig = plt.figure(figsize=(20, 4))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    subfigs[0].suptitle("(a) Correct predictions", fontsize=15)
    subfigs[1].suptitle("(b) Wrong predictions", fontsize=15)
    ax1 = subfigs[0].subplots(len(datasets), len(models))
    ax2 = subfigs[1].subplots(len(datasets), len(models))
    #fig.subplots_adjust(bottom=0.1)
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

    for i, (dataset, name, all, correct, wrong) in enumerate(data):
        col = get_ax_index(name)
        row = datasets.index(dataset)
        ax1[col].set_xlabel(f"Confidence", style="italic", fontsize=15)
        ax1[col].set_ylabel("Count", style="italic", fontsize=15)
        ax1[col].set_title(model_name[name], style="italic", fontsize=15, fontweight="bold")
        ax1[col].set_xticks(x_axis)
        ax1[col].set_xticklabels(
            ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        )
        ax1[col].hist(correct, x_axis, color=CB91_Blue)

        ax2[col].set_xlabel(f"Confidence", style="italic", fontsize=15)
        ax2[col].set_ylabel("Count", style="italic", fontsize=15)
        ax2[col].set_title(model_name[name], style="italic", fontsize=15, fontweight="bold")
        ax2[col].set_xticks(x_axis)
        ax2[col].set_xticklabels(
            ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        )
        ax2[col].hist(wrong, x_axis, color=CB91_Pink)
        # ax[row, col].set_xlabel(f"Confidence", style="italic", fontsize=15)
        # ax[row, col].set_ylabel("Count", style="italic", fontsize=15)
        # ax[row, col].set_title(name, style="italic", fontsize=15, fontweight="bold")
        # ax[row, col].set_xticks(x_axis)
        # ax[row, col].set_xticklabels(
        # ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        # )
        # ax[row, col].hist(wrong, x_axis, color=CB91_Pink)

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

    name = f"{'_'.join(datasets)}_dist"
    fig_name = f"{name.replace(' ', '_')}.svg"
    fig.tight_layout()
    fig.savefig(fig_name)
    print(f"Saved as {fig_name}")
