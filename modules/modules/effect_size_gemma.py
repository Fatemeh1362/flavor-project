
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import functions as F, Window
from scipy.stats import spearmanr, linregress


def save_clean_allelic_png(
    df,
    trait,
    chrom,
    start_min,
    start_max,
    taglo_ids,
    out_dir
):
    """
    Allelic effect plot (tetraploid-capped dosage 0–4)
    - dosage on x-axis
    - n above each box
    - taglo IDs shown in title
    - fold change (4 vs 0) shown in title
    """



    # -----------------------------
    # SAFETY: enforce 0–4
    # -----------------------------
    df = df.copy()
    df["dosage"] = df["dosage"].clip(0, 4)

    # -----------------------------
    # counts
    # -----------------------------
    counts = df.groupby("dosage")["trait_value"].count()

    # -----------------------------
    # fold change (4 vs 0)
    # -----------------------------
    if 0 in counts.index and 4 in counts.index:
        fc = (
            df.loc[df["dosage"] == 4, "trait_value"].mean()
            / df.loc[df["dosage"] == 0, "trait_value"].mean()
        )
    else:
        fc = np.nan

    # -----------------------------
    # plot
    # -----------------------------
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # sns.violinplot(
    #     data=df,
    #     x="dosage",
    #     y="trait_value",
    #     inner=None,
    #     cut=0,
    #     ax=ax
    # )

    sns.boxplot(
        data=df,
        x="dosage",
        y="trait_value",
        width=0.4,
        showfliers=False,
        boxprops={"facecolor": "white"},
        ax=ax
    )

    # -----------------------------
    # n ABOVE each box
    # -----------------------------
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    for i, d in enumerate(sorted(counts.index)):
        y_top = df.loc[df["dosage"] == d, "trait_value"].max()
        ax.text(
            i,
            y_top + 0.05 * yrange,
            f"n={counts[d]}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_ylim(ymin, ymax + 0.15 * yrange)

    # -----------------------------
    # title with taglo IDs + FC
    # -----------------------------
    taglo_txt = ",".join(map(str, taglo_ids[:4]))
    if len(taglo_ids) > 4:
        taglo_txt += " …"

    fc_txt = f"FC(4/0)={fc:.2f}" if not np.isnan(fc) else "FC(4/0)=NA"

    ax.set_title(
        f"{trait} | {chrom}:{start_min}-{start_max}\n"
        f"Taglo IDs: {taglo_txt} | {fc_txt}",
        fontsize=11
    )

    ax.set_xlabel("Allelic dosage (0–4)")
    ax.set_ylabel(trait)

    # -----------------------------
    # save
    # -----------------------------
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{trait}_{chrom}_{start_min}_{start_max}.png".replace(" ", "_")
    path = os.path.join(out_dir, fname)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return path, fc



    


def build_dosage_df_from_taglo(
    spark,
    taglo_table,
    taglo_ids,
    pheno_df,
    trait,
):
    """
    Build tetraploid-capped allelic dosage dataframe (0–4 ONLY).

    Steps:
    1) Sum alleles across taglo_id list (raw burden)
    2) CAP to tetraploid scale [0, 4]
    """

    # -----------------------------
    # genotype → raw summed dosage
    # -----------------------------
    gt = (
        spark.table(taglo_table)
        .filter(F.col("taglo_id").isin(taglo_ids))
        .select(
            F.col("variety").cast("string").alias("Variety"),
            F.col("value").cast("int").alias("allele")
        )
        .groupBy("Variety")
        .agg(F.sum("allele").alias("dosage_raw"))
        .toPandas()
    )

    # -----------------------------
    # phenotype + merge
    # -----------------------------
    df = (
        pheno_df[["Variety", trait]]
        .dropna()
        .rename(columns={trait: "trait_value"})
        .merge(gt, on="Variety", how="left")
    )

    # missing genotype → 0
    df["dosage_raw"] = df["dosage_raw"].fillna(0).astype(int)

    # -----------------------------
    #  HARD CAP to 0–4 (tetraploid)
    # -----------------------------
    df["dosage"] = df["dosage_raw"].clip(lower=0, upper=4)

    # -----------------------------
    # HARD SAFETY CHECK
    # -----------------------------
    assert df["dosage"].max() <= 4, "❌ Dosage > 4 detected after capping!"

    return df


def get_cluster_taglos(
    spark,
    gwas_table,
    trait,
    chrom,
    start_min,
    start_max,
    p_thresh=1e-6,
):
    sdf = (
        spark.table(gwas_table)
        .filter(F.col("trait") == trait)
        .filter(F.col("chrom") == chrom)
        .filter(F.col("start").between(start_min, start_max))
        .filter(F.col("p_wald") <= p_thresh)
        .select(
            F.explode(
                F.array("taglo_id1","taglo_id2","taglo_id3","taglo_id4")
            ).alias("taglo")
        )
        .filter(F.col("taglo").isNotNull())
        .filter(F.col("taglo") > 0)
        .select(F.col("taglo").cast("int"))
        .distinct()
    )

    return [r.taglo for r in sdf.collect()]


    
def build_gwas_clusters(
    spark,
    gwas_table,
    p_thresh=1e-6,
    window_bp=500_000,
):
    """
    Build genomic clusters (loci) from GWAS results.
    Each cluster = contiguous significant SNPs for one trait & chromosome.
    """

    sdf = (
        spark.table(gwas_table)
        .filter(F.col("p_wald") <= p_thresh)
        .select("trait", "chrom", "start")
        #  CRITICAL FIX: cast start to numeric
        .withColumn("start", F.col("start").cast("long"))
        .dropna(subset=["start"])
    )

    # window: per trait + chromosome
    w = Window.partitionBy("trait", "chrom").orderBy("start")

    sdf = (
        sdf
        .withColumn("prev_start", F.lag("start").over(w))
        .withColumn(
            "new_cluster",
            F.when(
                (F.col("prev_start").isNull()) |
                (F.col("start") - F.col("prev_start") > window_bp),
                1
            ).otherwise(0)
        )
        .withColumn(
            "cluster_id",
            F.sum("new_cluster").over(w)
        )
    )

    # summarize clusters
    clusters = (
        sdf
        .groupBy("trait", "chrom", "cluster_id")
        .agg(
            F.min("start").alias("start_min"),
            F.max("start").alias("start_max"),
            F.count("*").alias("n_snps")
        )
        .orderBy("trait", "chrom", "start_min")
        .toPandas()
    )

    return clusters






def compute_dosage_linearity(df):

    if df is None or df.empty:
        return None

    agg = (
        df.groupby("dosage")["trait_value"]
        .agg(count="count", median="median")
        .reset_index()
        .sort_values("dosage")
    )

    if agg.shape[0] < 2:
        return None

    x = agg["dosage"].values
    y = agg["median"].values

    # Spearman
    rho, p_spear = spearmanr(x, y)

    # Linear regression
    slope, intercept, r, p_lin, _ = linregress(x, y)

    # shape
    diffs = np.diff(y)
    if np.all(np.abs(diffs) < 0.1):
        shape = "flat"
    elif np.all(diffs > 0) or np.all(diffs < 0):
        shape = "monotonic"
    else:
        shape = "non_monotonic"

    return {
        "status": "ok",
        "n_groups": int(len(agg)),
        "dosages_used": agg["dosage"].tolist(),

        "spearman_rho": float(rho),
        "spearman_p": float(p_spear),

        "linear_slope": float(slope),
        "linear_r2": float(r**2),
        "linear_p": float(p_lin),

        "delta_extreme": float(y[-1] - y[0]),
        "dosage_shape": shape
    }


