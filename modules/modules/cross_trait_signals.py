"""
Author: Fatemeh Monfared

This module identifies genomic loci shared across multiple traits
based on shared GWAS signals, provides clean visualization utilities,
and performs community detection on trait networks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.algorithms import community


# ---------------------------
# Core function
# ---------------------------
def find_cross_trait_signals(shared_signals_df, min_traits=2):

    loci = (
        shared_signals_df
        .groupby(["chrom", "start"])["trait"]
        .nunique()
        .reset_index(name="n_traits")
    )

    shared = loci[loci["n_traits"] >= min_traits]

    trait_map = (
        shared_signals_df
        .groupby(["chrom","start"])["trait"]
        .unique()
        .reset_index(name="traits")
    )

    return shared.merge(trait_map, on=["chrom","start"])


# ---------------------------
# Helper: filter strong traits
# ---------------------------
def _filter_strong_traits(shared_df, min_total_overlap=50):

    pivot = (
        shared_df.assign(val=1)
        .pivot_table(index=["chrom","start"], columns="trait", values="val", fill_value=0)
    )

    overlap = pivot.T @ pivot

    strong = overlap.sum(axis=1)
    top_traits = strong[strong > min_total_overlap].index

    return shared_df[shared_df["trait"].isin(top_traits)]


# ---------------------------
# Visualization 1: scatter
# ---------------------------
def plot_shared_loci_scatter(shared_signals_df):

    counts = (
        shared_signals_df
        .groupby(["chrom","start"])["trait"]
        .nunique()
        .reset_index(name="n_traits")
    )

    plt.figure(figsize=(10,5))
    plt.scatter(counts["start"], counts["n_traits"], alpha=0.5)

    plt.xlabel("Genomic Position")
    plt.ylabel("Number of Traits")
    plt.title("Cross-Trait Shared Loci")

    plt.tight_layout()
    plt.show()


# ---------------------------
# Visualization 2: highlight
# ---------------------------
def plot_shared_loci_highlight(shared_signals_df, threshold=3):

    counts = (
        shared_signals_df
        .groupby(["chrom","start"])["trait"]
        .nunique()
        .reset_index(name="n_traits")
    )

    plt.figure(figsize=(10,5))

    plt.scatter(counts["start"], counts["n_traits"], alpha=0.3)

    top = counts[counts["n_traits"] >= threshold]

    plt.scatter(top["start"], top["n_traits"], s=80)

    plt.xlabel("Position")
    plt.ylabel("#Traits")
    plt.title("Highly Shared Loci")

    plt.tight_layout()
    plt.show()


# ---------------------------
# Visualization 3: heatmap
# ---------------------------
def plot_trait_overlap_heatmap(shared_signals_df):

    df = _filter_strong_traits(shared_signals_df)

    pivot = (
        df.assign(val=1)
        .pivot_table(index=["chrom","start"], columns="trait", values="val", fill_value=0)
    )

    overlap = pivot.T @ pivot

    plt.figure(figsize=(10,8))
    sns.heatmap(overlap, cmap="viridis")

    plt.title("Trait Overlap Matrix (Filtered)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


# ---------------------------
# Build graph (shared utility)
# ---------------------------
def build_trait_graph(shared_signals_df, min_overlap=10):

    df = _filter_strong_traits(shared_signals_df)

    pivot = (
        df.assign(val=1)
        .pivot_table(index=["chrom","start"], columns="trait", values="val", fill_value=0)
    )

    overlap = pivot.T @ pivot

    G = nx.Graph()

    for t1 in overlap.index:
        for t2 in overlap.columns:
            if t1 != t2 and overlap.loc[t1, t2] >= min_overlap:
                G.add_edge(t1, t2, weight=overlap.loc[t1, t2])

    return G, overlap


# ---------------------------
# Visualization 4: network
# ---------------------------
def plot_trait_network(shared_signals_df, min_overlap=10):

    G, _ = build_trait_graph(shared_signals_df, min_overlap)

    plt.figure(figsize=(10,8))

    pos = nx.spring_layout(G, k=0.6)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1500,
        font_size=8
    )

    plt.title("Trait Network (Filtered Shared Loci)")

    plt.tight_layout()
    plt.show()


# ---------------------------
# Community detection 🔥
# ---------------------------
def detect_communities(shared_signals_df, min_overlap=10):

    G, _ = build_trait_graph(shared_signals_df, min_overlap)

    communities = community.greedy_modularity_communities(G)

    result = []
    for i, comm in enumerate(communities):
        for trait in comm:
            result.append({
                "trait": trait,
                "community": i
            })

    return pd.DataFrame(result)


# ---------------------------
# Community summary (loci-level)
# ---------------------------
def community_summary(shared_signals_df, community_df):

    merged = shared_signals_df.merge(community_df, on="trait")

    summary = (
        merged
        .groupby(["community","chrom","start"])
        .agg(
            n_traits=("trait","nunique"),
            traits=("trait", lambda x: list(set(x)))
        )
        .reset_index()
    )

    return summary


# ---------------------------
# Save utilities
# ---------------------------
def save_communities(community_df, path):

    community_df.to_csv(path, index=False)


def save_community_loci(summary_df, path):

    summary_df.to_csv(path, index=False)