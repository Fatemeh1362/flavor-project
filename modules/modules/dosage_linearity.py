# =========================
# imports
# =========================
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from scipy.stats import spearmanr, linregress


# =========================
# 1. Build dosage dataframe
# =========================
def build_dosage_df_from_taglo(
    spark,
    taglo_table,
    taglo_ids,
    pheno_df,
    trait,
):
    """
    Build tetraploid-capped allelic dosage dataframe (0–4).
    """

    if not taglo_ids:
        return None

    # genotype → raw summed dosage
    gt = (
        spark.table(taglo_table)
        .filter(F.col("taglo_id").isin([int(t) for t in taglo_ids]))
        .select(
            F.col("variety").cast("string").alias("Variety"),
            F.col("value").cast("int").alias("allele")
        )
        .groupBy("Variety")
        .agg(F.sum("allele").alias("dosage_raw"))
        .toPandas()
    )

    # phenotype + merge
    df = (
        pheno_df[["Variety", trait]]
        .dropna()
        .rename(columns={trait: "trait_value"})
        .merge(gt, on="Variety", how="left")
    )

    df["dosage_raw"] = df["dosage_raw"].fillna(0).astype(int)

    # tetraploid cap
    df["dosage"] = df["dosage_raw"].clip(lower=0, upper=4)

    assert df["dosage"].max() <= 4, "Dosage > 4 detected!"

    return df


# =========================
# 2. Dosage linearity
# =========================
def compute_dosage_linearity(df):
    """
    Analyze relationship between dosage and trait_value.
    """

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

    # monotonicity
    rho, p_spear = spearmanr(x, y)

    # linear regression
    slope, intercept, r, p_lin, _ = linregress(x, y)

    # qualitative shape
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


# =========================
# 3. Extract taglos from GWAS cluster
# =========================
def get_cluster_taglos(
    spark,
    gwas_table,
    trait,
    chrom,
    start_min,
    start_max,
    p_thresh=1e-6
):
    """
    Return unique taglo_ids associated with a GWAS locus window.
    """

    sdf = (
        spark.table(gwas_table)
        .filter(F.col("trait") == trait)
        .filter(F.col("chrom") == chrom)
        .filter(F.col("start").between(start_min, start_max))
        .filter(F.col("p_wald") <= p_thresh)
        .select(
            F.explode(
                F.array(
                    "taglo_id1",
                    "taglo_id2",
                    "taglo_id3",
                    "taglo_id4"
                )
            ).alias("taglo_id")
        )
        .filter(F.col("taglo_id").isNotNull())
        .filter(F.col("taglo_id") > 0)
        .select(F.col("taglo_id").cast("int"))
        .distinct()
    )

    return [r.taglo_id for r in sdf.collect()]


# =========================
# 4. Build GWAS clusters
# =========================
def build_clusters(spark, gwas_table, p_thresh=1e-6):
    """
    Build locus clusters from GWAS table.
    """

    clusters = (
        spark.table(gwas_table)
        .filter(F.col("p_wald") <= p_thresh)
        .select("trait", "chrom", "start")
        .groupBy("trait", "chrom")
        .agg(
            F.min("start").alias("start_min"),
            F.max("start").alias("start_max")
        )
        .toPandas()
    )

    return clusters


# =========================
# 5. Full pipeline
# =========================
def run_dosage_pipeline(
    spark,
    gwas_table,
    taglo_table,
    pheno_df,
    output_csv,
    p_thresh=1e-6
):
    """
    End-to-end pipeline:
    GWAS → clusters → taglos → dosage → linearity
    """

    clusters = build_clusters(spark, gwas_table, p_thresh)

    results = []

    for r in clusters.itertuples():
        taglo_ids = get_cluster_taglos(
            spark,
            gwas_table,
            r.trait,
            r.chrom,
            int(r.start_min),
            int(r.start_max),
            p_thresh
        )

        if not taglo_ids:
            continue

        df = build_dosage_df_from_taglo(
            spark,
            taglo_table,
            taglo_ids,
            pheno_df,
            r.trait
        )

        beh = compute_dosage_linearity(df)

        if beh is None:
            continue

        results.append({
            "trait": r.trait,
            "chrom": r.chrom,
            "start_min": int(r.start_min),
            "start_max": int(r.start_max),
            "n_taglos": len(taglo_ids),
            "taglo_ids": sorted(taglo_ids),
            **beh
        })

    result_df = pd.DataFrame(results)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df