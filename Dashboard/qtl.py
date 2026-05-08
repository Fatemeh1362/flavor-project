# manhattan_module.py
# ------------------------------------------------------------
# Utilities for:
# - loading phenotypes
# - listing traits from GWAS table
# - fetching a lightweight GWAS subset for one trait
# - plotting a Manhattan plot
# ------------------------------------------------------------

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql import DataFrame as SparkDataFrame


# ----------------------------
# Phenotype loading / reshaping
# ----------------------------

def load_phenotypes_csv(pheno_path: str) -> pd.DataFrame:
    """
    Load phenotype matrix from CSV.
    Expected: a column 'Variety' that becomes the index.
    Cleans whitespace and removes duplicated Variety IDs (keeps first).
    Converts trait columns to numeric (non-numeric -> NaN).
    """
    df = pd.read_csv(pheno_path)

    if "Variety" not in df.columns:
        raise ValueError("Column 'Variety' not found in phenotype file.")

    df["Variety"] = df["Variety"].astype(str).str.strip()
    df = df.set_index("Variety")
    df = df[~df.index.duplicated(keep="first")]

    # Optional: convert all columns to numeric (safe for stats/plots)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def phenotypes_to_long(df_pheno: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide phenotype matrix (index=Variety, columns=traits)
    to long format: Variety, trait, trait_value
    """
    df_long = (
        df_pheno.reset_index()
        .melt(id_vars="Variety", var_name="trait", value_name="trait_value")
        .dropna(subset=["trait_value"])
    )
    return df_long


# ----------------------------
# GWAS trait listing / fetching
# ----------------------------

def get_traits(
    spark,
    gwas_table: str = "bmqg.gwas.run_local_20251207",
    trait_col_candidates: tuple[str, ...] = ("trait", "phenotype"),
) -> list[str]:
    """
    Return distinct trait names from GWAS table.
    """
    df0 = spark.table(gwas_table)
    trait_col = None
    for c in trait_col_candidates:
        if c in df0.columns:
            trait_col = c
            break
    if trait_col is None:
        raise ValueError(f"No trait column found in {gwas_table}. Tried: {trait_col_candidates}")

    traits = (
        df0.select(F.col(trait_col).cast("string").alias("trait"))
           .dropna()
           .distinct()
           .toPandas()["trait"]
           .astype(str)
           .tolist()
    )
    return traits


def load_trait_gwas_subset(
    spark,
    trait: str,
    source_table: str = "bmqg.gwas.run_local_20251207",
    trait_col_candidates: tuple[str, ...] = ("trait", "phenotype"),
    chrom_col: str = "chrom",
    pos_col: str = "start",
    p_col: str = "p_wald",
    p_sig: float = 1e-6,
    random_fraction: float = 0.05,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Fetch a small GWAS dataframe for one trait:
      - all significant rows (p < p_sig)
      - plus a random sample of background rows for context

    Returns pandas with columns: chrom, start, p_wald (names match pos_col/p_col).
    """
    df0: SparkDataFrame = spark.table(source_table)

    trait_col = None
    for c in trait_col_candidates:
        if c in df0.columns:
            trait_col = c
            break
    if trait_col is None:
        raise ValueError(f"No 'trait' or 'phenotype' column found in {source_table}.")

    df = (
        df0.filter(F.col(trait_col) == F.lit(str(trait)))
           .select(
               F.col(chrom_col).cast("string").alias("chrom"),
               F.col(pos_col).cast("long").alias("start"),
               F.col(p_col).cast("double").alias("p_wald"),
           )
           .dropna(subset=["chrom", "start", "p_wald"])
    )

    df_sig = df.filter(F.col("p_wald") < F.lit(float(p_sig)))
    df_rand = df.sample(withReplacement=False, fraction=float(random_fraction), seed=int(seed))

    df_small = df_sig.unionByName(df_rand).dropDuplicates(["chrom", "start", "p_wald"])
    pdf = df_small.toPandas()
    return pdf


# ----------------------------
# Manhattan plotting
# ----------------------------

def _natural_chrom_key(chrom: str):
    """
    Sort key for chrom strings like 'ST4.03ch01', 'ST4.03ch12', etc.
    Extracts last number; falls back to string.
    """
    s = str(chrom)
    m = re.search(r"(\d+)\s*$", s)
    if m:
        return (re.sub(r"\d+\s*$", "", s), int(m.group(1)))
    m2 = re.search(r"ch(\d+)", s)
    if m2:
        return (re.sub(r"ch\d+", "ch", s), int(m2.group(1)))
    return (s, 10**9)


def plot_manhattan(
    pdf: pd.DataFrame,
    trait: str,
    p_threshold: float = 1e-6,
    point_size: int = 7,
    figsize: tuple[int, int] = (18, 6),
):
    """
    Manhattan plot from a pandas dataframe that has:
      chrom, start, p_wald

    p_threshold is a p-value (e.g., 1e-6). A red line is drawn at -log10(p_threshold).
    """

    # Flatten weird values (lists/arrays)
    for c in ["chrom", "start", "p_wald"]:
        if c in pdf.columns:
            pdf[c] = pdf[c].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

    pdf = pdf.copy()
    pdf["start"] = pd.to_numeric(pdf["start"], errors="coerce")
    pdf["p_wald"] = pd.to_numeric(pdf["p_wald"], errors="coerce")
    pdf["chrom"] = pdf["chrom"].astype(str)

    pdf = pdf.dropna(subset=["chrom", "start", "p_wald"])
    pdf = pdf[pdf["p_wald"] > 0]  # avoid log10 issues

    # Order chromosomes naturally
    chroms = sorted(pdf["chrom"].unique(), key=_natural_chrom_key)
    chrom_index = {c: i for i, c in enumerate(chroms)}
    pdf["chrom_numeric"] = pdf["chrom"].map(chrom_index)

    pdf = pdf.sort_values(["chrom_numeric", "start"])
    pdf["minus_log10_p"] = -np.log10(pdf["p_wald"])

    # Cumulative positions
    chrom_max = pdf.groupby("chrom")["start"].max()
    offsets = (chrom_max.cumsum() - chrom_max).to_dict()
    pdf["cum_pos"] = pdf.apply(lambda r: r["start"] + offsets.get(r["chrom"], 0), axis=1)

    # Plot
    plt.figure(figsize=figsize)

    # Alternating colors without hardcoding a long list
    # (matplotlib default cycle would also work; this is stable)
    colors = ["#4C72B0", "#55A868"]

    for chrom in chroms:
        sub = pdf[pdf["chrom"] == chrom]
        idx = chrom_index[chrom]
        plt.scatter(sub["cum_pos"], sub["minus_log10_p"], s=point_size, color=colors[idx % 2])

    y_thr = -np.log10(float(p_threshold))
    plt.axhline(y_thr, color="red", linestyle="--", linewidth=1.3)

    ticks = [pdf[pdf["chrom"] == c]["cum_pos"].median() for c in chroms]
    plt.xticks(ticks, chroms, rotation=45, ha="right")

    plt.xlabel("Chromosome")
    plt.ylabel("-log10(p)")
    plt.title(f"Manhattan Plot — {trait}")
    plt.tight_layout()
    plt.show()

    return pdf  # returns the processed dataframe (handy for debugging)
