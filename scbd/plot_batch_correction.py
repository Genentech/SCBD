import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import auc
from utils.file import load_file
from utils.plot import *


def main(args):
    algorithms = ["SCBD", "CellProfiler", r"iVAE ($q_\phi(z_s \mid x, e)$)", r"iVAE ($q_\phi(z_s \mid x)$)", "mcVAE", "VAE"]
    well_f1_score, corum_auprc = [], []
    for algorithm in algorithms:
        well_f1_score_row, corum_auprc_row = [], []
        for seed in range(args.n_seeds):
            if algorithm ==  "SCBD":
                _, f1_score = load_file(os.path.join(args.results_dpath, "resnet18,alpha=8", f"version_{seed}", "batch_effects.pkl"))
                well_f1_score_row.append(f1_score.loc["Plate and well", "zc"].item())
                corum = load_file(os.path.join(args.results_dpath, "resnet18,alpha=8", f"version_{seed}", "corum.pkl"))
                corum_auprc_row.append(auc(corum["zc"].recall, corum["zc"].precision).item())
            elif algorithm == "CellProfiler":
                _, f1_score = load_file(os.path.join(args.results_dpath, "resnet18,alpha=0", f"version_{seed}", "batch_effects.pkl"))
                well_f1_score_row.append(f1_score.loc["Plate and well", "CellProfiler"].item())
                corum = load_file(os.path.join(args.results_dpath, "resnet18,alpha=0", f"version_{seed}", "corum.pkl"))
                corum_auprc_row.append(auc(corum["CellProfiler"].recall, corum["CellProfiler"].precision).item())
            elif algorithm == r"iVAE ($q_\phi(z_s \mid x, e)$)":
                _, f1_score = load_file(os.path.join(args.results_dpath, "ivae_cond", f"version_{seed}", "batch_effects.pkl"))
                well_f1_score_row.append(f1_score.loc["Plate and well", "zc"].item())
                corum = load_file(os.path.join(args.results_dpath, "ivae_cond", f"version_{seed}", "corum.pkl"))
                corum_auprc_row.append(auc(corum["zc"].recall, corum["zc"].precision).item())
            elif algorithm == r"iVAE ($q_\phi(z_s \mid x)$)":
                _, f1_score = load_file(os.path.join(args.results_dpath, "ivae_uncond", f"version_{seed}", "batch_effects.pkl"))
                well_f1_score_row.append(f1_score.loc["Plate and well", "zc"].item())
                corum = load_file(os.path.join(args.results_dpath, "ivae_uncond", f"version_{seed}", "corum.pkl"))
                corum_auprc_row.append(auc(corum["zc"].recall, corum["zc"].precision).item())
            elif algorithm == "mcVAE":
                _, f1_score = load_file(os.path.join(args.results_dpath, "mcvae", f"version_{seed}", "batch_effects.pkl"))
                well_f1_score_row.append(f1_score.loc["Plate and well", "z"].item())
                corum = load_file(os.path.join(args.results_dpath, "mcvae", f"version_{seed}", "corum.pkl"))
                corum_auprc_row.append(auc(corum["z"].recall, corum["z"].precision).item())
            else:
                assert algorithm == "VAE"
                _, f1_score = load_file(
                    os.path.join(args.results_dpath, "vae", f"version_{seed}", "batch_effects.pkl"))
                well_f1_score_row.append(f1_score.loc["Plate and well", "z"].item())
                corum = load_file(os.path.join(args.results_dpath, "vae", f"version_{seed}", "corum.pkl"))
                corum_auprc_row.append(auc(corum["z"].recall, corum["z"].precision).item())
        well_f1_score.append(well_f1_score_row)
        corum_auprc.append(corum_auprc_row)
    well_f1_score = pd.DataFrame(np.array(well_f1_score).T).melt()
    corum_auprc = pd.DataFrame(np.array(corum_auprc).T).melt()

    well_auroc_means = well_f1_score.groupby("variable")["value"].mean()
    well_auroc_sds = well_f1_score.groupby("variable")["value"].std()

    corum_auprc_means = corum_auprc.groupby("variable")["value"].mean()
    corum_auprc_sds = corum_auprc.groupby("variable")["value"].std()

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    for i in range(len(algorithms)):
        axes[0].errorbar(i, well_auroc_means[i], yerr=well_auroc_sds[i], fmt="o", color=sns.color_palette()[i],
            markersize=10, capsize=10, linewidth=2, elinewidth=2, label=algorithms[i])
        axes[1].errorbar(i, corum_auprc_means[i], yerr=corum_auprc_sds[i], fmt="o", color=sns.color_palette()[i],
            markersize=10, capsize=10, linewidth=2, elinewidth=2)

    for ax in axes:
        ax.set_xticks([])

    axes[0].set_ylabel("Well (F1 score)")
    axes[1].set_ylabel("CORUM (AUPRC)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.225)
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "batch_correction.svg"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dpath", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=3)
    main(parser.parse_args())