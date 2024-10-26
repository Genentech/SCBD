import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def main(args):
    hparams_range = os.listdir(args.results_dpath)
    val_accs, test_accs = [], []
    for hparams in hparams_range:
        val_acc_hparams, test_acc_hparams = [], []
        for seed in range(args.n_seeds):
            df = pd.read_csv(os.path.join(args.results_dpath, hparams, f"version_{seed}", "metrics.csv"))
            val_acc_hparams.append(df.val_acc.max().item())
            test_acc_hparams.append(df.test_acc.iloc[-1].item())
        val_accs.append(np.mean(val_acc_hparams))
        test_accs.append(np.mean(test_acc_hparams))
    print(f"Correlation: {100 * np.corrcoef(val_accs, test_accs)[0, 1]:.1f}%")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(val_accs, test_accs, s=100, edgecolors="black")
    ax.set_xlabel("Val accuracy")
    ax.set_ylabel("Test accuracy")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.325)

    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "domain_generalization_baseline.pdf"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dpath", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--alpha_range", nargs="+", type=int, default=[0, 64, 128, 256])
    main(parser.parse_args())