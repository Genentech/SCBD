import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from batch_effects import get_embed
from scipy.spatial import distance
from sklearn.metrics import recall_score, precision_score
from utils.file import save_file


def get_corum_graph(corum_fpath, genes):
    rows = open(corum_fpath).readlines()
    gene_groups = [row.replace("\n", "") for row in rows]
    gene_groups = [gene_group.split("\t")[1].split(" ") for gene_group in gene_groups]
    gene_groups = [[gene for gene in gene_group if gene in genes] for gene_group in gene_groups]
    unique_genes = set(list(itertools.chain(*gene_groups)))
    gene_graph = nx.Graph()
    parent_to_children = {x: set() for x in unique_genes}
    for gene_group in gene_groups:
        for gene in gene_group:
            parent_to_children[gene].update(gene_group)
            parent_to_children[gene].remove(gene)
    for parent, children in parent_to_children.items():
        for child in children:
            gene_graph.add_edge(parent, child)
    return gene_graph


def get_precision_and_recall(embeddings, gene_graph, percentile_range):
    sim_mat = 1 - distance.cdist(embeddings, embeddings, "cosine")
    sim_mat_flat = sim_mat.flatten()
    target, pred_sim = get_target_and_pred_sim(gene_graph, sim_mat)
    df = {"precision": [], "recall": []}
    for percentile in percentile_range:
        upper_threshold = np.percentile(sim_mat_flat, percentile)
        lower_threshold = np.percentile(sim_mat_flat, 100 - percentile)
        pred_binary = pred_sim_to_binary(pred_sim, upper_threshold, lower_threshold)
        df["precision"].append(precision_score(target, pred_binary))
        df["recall"].append(recall_score(target, pred_binary))
    return pd.DataFrame(df)


def get_target_and_pred_sim(gene_graph, sim_mat):
    target, pred_sim = [], []
    adj_mat = nx.adjacency_matrix(gene_graph).toarray()
    for i in range(len(adj_mat)):
        target_elem = adj_mat[i].tolist()
        pred_sim_elem = sim_mat[i].tolist()
        target_elem.pop(i)
        pred_sim_elem.pop(i)
        target += target_elem
        pred_sim += pred_sim_elem
    return target, pred_sim


def pred_sim_to_binary(pred, upper_threshold, lower_threshold):
    return [1 if ((elem >= upper_threshold) or (elem <= lower_threshold)) else 0 for elem in pred]


def main(args):
    metadata_df, zc, zs, cellprofiler = get_embed(args.s3_bucket_name, args.train_embed_dpath, args.val_embed_dpath,
        args.test_embed_dpath, args.cellprofiler_fpath)
    metadata_df = metadata_df[["sgRNA_0", "gene_symbol_0"]]

    embed_name_to_precision_recall = {}
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for embed_name in ("zc", "zs", "CellProfiler"):
        if embed_name == "zc":
            embed = zc
        elif embed_name == "zs":
            embed = zs
        else:
            assert embed_name == "CellProfiler"
            embed = cellprofiler
        embed_df = pd.DataFrame(embed)

        df = pd.concat((metadata_df, embed_df), axis=1)
        df = df.groupby("sgRNA_0").agg({col: "mean" if col != "gene_symbol_0" else "first" for col in df.columns[1:]})
        df = df.groupby("gene_symbol_0").mean()
        df = df - df.loc["nontargeting"]
        df = (df - df.mean(axis=0)) / df.std(axis=0)

        genes = df.index.values
        gene_graph = get_corum_graph(args.corum_fpath, genes)

        idxs = []
        for node in list(gene_graph.nodes):
            idxs.append(np.where(genes == node)[0][0])

        df = df.iloc[idxs]
        assert list(df.index.values) == list(gene_graph.nodes)

        precision_recall = get_precision_and_recall(
            df.values,
            gene_graph,
            np.arange(80, 100, 1)
        )
        ax.plot(precision_recall.recall, precision_recall.precision, label=embed_name)
        embed_name_to_precision_recall[embed_name] = precision_recall
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.225)

    os.makedirs(args.results_dpath, exist_ok=True)
    plt.savefig(os.path.join(args.results_dpath, "corum.png"))
    save_file(embed_name_to_precision_recall, os.path.join(args.results_dpath, "corum.pkl"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dpath", type=str, required=True)
    parser.add_argument("--s3_bucket_name", type=str, required=True)
    parser.add_argument("--train_embed_dpath", type=str, required=True)
    parser.add_argument("--val_embed_dpath", type=str, required=True)
    parser.add_argument("--test_embed_dpath", type=str, required=True)
    parser.add_argument("--cellprofiler_fpath", type=str, required=True)
    parser.add_argument("--corum_fpath", type=str, required=True)
    main(parser.parse_args())