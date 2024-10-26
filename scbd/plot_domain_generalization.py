import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def main(args):
    values_val, values_test = [], []
    for alpha in args.alpha_range:
        values_val_row, values_test_row = [], []
        for seed in range(args.n_seeds):
            df = pd.read_csv(os.path.join(args.results_dpath, f"alpha={alpha}", f"version_{seed}", "metrics.csv"))
            opt_val_idx = df.val_loss.argmin().item()
            values_val_row.append(df.val_acc_y_rc.iloc[opt_val_idx])
            values_test_row.append(df.test_acc_y_rc.iloc[-1])
        values_val.append(values_val_row)
        values_test.append(values_test_row)
    values_val = pd.DataFrame(np.array(values_val).T).melt()
    values_test = pd.DataFrame(np.array(values_test).T).melt()

    fig, ax_val = plt.subplots(1, 1, figsize=(6, 4))
    ax_test = ax_val.twinx()

    sns.lineplot(data=values_val, x="variable", y="value", errorbar="sd", legend=False, ax=ax_val, label="Val",
        color=sns.color_palette()[0])
    sns.lineplot(data=values_test, x="variable", y="value", errorbar="sd", legend=False, ax=ax_test, label="Test",
        color=sns.color_palette()[1])

    ax_val.set_xlabel(r"$\alpha$")
    ax_val.set_xticks(range(len(args.alpha_range)))
    ax_val.set_xticklabels(args.alpha_range)
    ax_val.set_ylabel("Val accuracy", labelpad=10)
    ax_test.set_ylabel("Test accuracy", labelpad=10)
    ax_val.tick_params(axis="y", colors=sns.color_palette()[0])
    ax_test.tick_params(axis="y", colors=sns.color_palette()[1])

    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.325)

    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "domain_generalization.pdf"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dpath", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--alpha_range", nargs="+", type=int, default=[0, 64, 128, 192])
    main(parser.parse_args())