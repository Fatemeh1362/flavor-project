
"""
Author: Fatemeh Monfared
Module: PLSR.py
Title: Global-QTL PLSR from GWAS (config-driven, batch-safe)

What this module does
---------------------
This module builds a compact, interpretable “global QTL” representation from GWAS results and then
runs a PLSR-based ranking to identify which global QTL blocks are most informative for each trait.

It is designed for Databricks / Spark workflows:
- heavy lifting happens in Spark
- only the final trait × QTL matrix is collected to pandas for PLSR
- plotting uses the non-interactive matplotlib backend (“Agg”) so it can run in batch jobs

Workflow overview
-----------------
1) Read GWAS rows from a Spark table (trait, chrom, start, p_wald, beta, taglo_id1..taglo_id4).
2) Create “global QTL blocks” per chromosome by grouping nearby marker positions using a distance threshold.
3) For each (trait × global-QTL block), pick one representative marker:
   - lowest p_wald (tie-breaker: larger absolute beta)
   - store a representative taglo combination (rep_combo) for traceability
4) Convert each (trait × QTL block) into a signed association score that keeps effect direction.
5) Keep only the strongest QTL blocks (to limit matrix width and keep things fast).
6) Pivot to a signed matrix: rows = traits, columns = QTL_id, values = signed score (missing → 0).
7) Run PLSRegression in a one-vs-rest way for each trait to extract the top contributing QTL blocks.
8) Save outputs:
   - CSV summary of top PLS loadings per trait
   - per-trait bar plots of loadings (PNG)"""





# ============================================================
# GLOBAL-QTL PLSR-from-GWAS (CONFIG-DRIVEN, BATCH-SAFE)
# ============================================================

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# ---- matplotlib: SAFE for Databricks batch ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression


# ============================================================
# 0) Small helpers
# ============================================================

def _cfg(config: Optional[dict], path: list, default=None):
    cur = config
    for p in path:
        if cur is None or p not in cur:
            return default
        cur = cur[p]
    return cur


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(name))


# ============================================================
# 1) Traits list (CONFIG-AWARE)
# ============================================================

def get_traits(
    spark,
    config: dict,
    trait_col: str = "trait",
) -> list[str]:

    gwas_table = _cfg(config, ["data", "gwas_table"])
    if gwas_table is None:
        raise ValueError("config['data']['gwas_table'] is missing")

    return (
        spark.table(gwas_table)
        .select(trait_col)
        .distinct()
        .toPandas()[trait_col]
        .astype(str)
        .tolist()
    )


# ============================================================
# 2) GLOBAL QTL blocks
# ============================================================

def build_global_qtl_blocks(
    spark,
    traits: Iterable[str],
    gwas_table: str,
    distance_threshold: int,
    drop_ch00: bool = True,
) -> Tuple[SparkDataFrame, SparkDataFrame]:

    base = (
        spark.table(gwas_table)
        .filter(F.col("trait").isin(list(traits)))
        .select("chrom", F.col("start").cast("long").alias("start"))
        .dropna(subset=["chrom", "start"])
        .dropDuplicates(["chrom", "start"])
    )

    if drop_ch00:
        base = base.filter(~F.col("chrom").like("%ch00"))

    w = Window.partitionBy("chrom").orderBy("start")

    marker_gqtl = (
        base
        .withColumn("prev_start", F.lag("start").over(w))
        .withColumn(
            "new_block",
            F.when(
                F.col("prev_start").isNull()
                | ((F.col("start") - F.col("prev_start")) > distance_threshold),
                1,
            ).otherwise(0),
        )
        .withColumn("gqtl_id", F.sum("new_block").over(w))
        .select("chrom", "start", "gqtl_id")
    )

    gqtl_bounds = (
        marker_gqtl.groupBy("chrom", "gqtl_id")
        .agg(
            F.min("start").alias("start_min"),
            F.max("start").alias("start_max"),
        )
        .withColumn("span", F.col("start_max") - F.col("start_min"))
    )

    return marker_gqtl, gqtl_bounds


# ============================================================
# 3) Per-trait GLOBAL-QTL meta
# ============================================================

def build_qtl_meta_global(
    spark,
    traits: Iterable[str],
    marker_gqtl: SparkDataFrame,
    gqtl_bounds: SparkDataFrame,
    gwas_table: str,
) -> SparkDataFrame:

    gwas = (
        spark.table(gwas_table)
        .filter(F.col("trait").isin(list(traits)))
        .select(
            "trait",
            "chrom",
            F.col("start").cast("long").alias("start"),
            F.col("p_wald").cast("double"),
            F.col("beta").cast("double"),
            *[F.col(f"taglo_id{i}").cast("long") for i in range(1, 5)],
        )
        .dropna(subset=["trait", "chrom", "start", "p_wald", "beta"])
    )

    gwas_combo = (
        gwas.withColumn(
            "combo_arr",
            F.expr(
                """
                array_sort(array_distinct(
                    filter(array(taglo_id1,taglo_id2,taglo_id3,taglo_id4),
                           x -> x is not null and x > 0)
                ))
                """
            ),
        )
        .withColumn("rep_combo", F.concat_ws(",", F.col("combo_arr")))
        .select("trait", "chrom", "start", "p_wald", "beta", "rep_combo")
    )

    gwas_combo_g = gwas_combo.join(marker_gqtl, ["chrom", "start"], "inner")

    rep_w = Window.partitionBy("trait", "chrom", "gqtl_id").orderBy(
        F.col("p_wald").asc(), F.abs(F.col("beta")).desc()
    )

    rep_marker = (
        gwas_combo_g.withColumn("rn", F.row_number().over(rep_w))
        .filter(F.col("rn") == 1)
        .select(
            "trait",
            "chrom",
            "gqtl_id",
            F.col("p_wald").alias("min_p"),
            F.col("beta").alias("rep_beta"),
            "rep_combo",
        )
    )

    meta = (
        rep_marker.join(gqtl_bounds, ["chrom", "gqtl_id"])
        .withColumn(
            "QTL_id",
            F.concat_ws("_", "chrom", F.concat(F.lit("GQTL"), F.col("gqtl_id"))),
        )
        .withColumn(
            "signed_score",
            F.signum("rep_beta")
            * (
                -F.log10(
                    F.when(F.col("min_p") <= 0, 1e-300).otherwise(F.col("min_p"))
                )
            ),
        )
    )

    return meta


# ============================================================
# 4) Top-QTL selection
# ============================================================

def select_top_qtls(meta_sdf: SparkDataFrame, max_qtls: int) -> SparkDataFrame:
    return (
        meta_sdf.withColumn("abs_score", F.abs("signed_score"))
        .groupBy("QTL_id", "chrom", "gqtl_id", "start_min", "start_max", "span")
        .agg(F.max("abs_score").alias("max_abs_score"))
        .orderBy(F.desc("max_abs_score"))
        .limit(int(max_qtls))
    )


# ============================================================
# 5) Signed matrix
# ============================================================

def build_signed_matrix_from_meta(
    meta_sdf: SparkDataFrame, keep_qtls_sdf: Optional[SparkDataFrame]
) -> SparkDataFrame:

    if keep_qtls_sdf is not None:
        meta_sdf = meta_sdf.join(
            keep_qtls_sdf.select("QTL_id").distinct(), "QTL_id"
        )

    return (
        meta_sdf.groupBy("trait")
        .pivot("QTL_id")
        .agg(F.first("signed_score"))
        .fillna(0.0)
    )


# ============================================================
# 6) PLSR (one-vs-rest)
# ============================================================

def plsr_all_traits(
    qtl_pd: pd.DataFrame,
    n_components: int,
    top_n: int,
) -> pd.DataFrame:

    X = StandardScaler().fit_transform(qtl_pd.values)
    traits = qtl_pd.index.tolist()
    features = qtl_pd.columns.tolist()

    rows = []

    for t in traits:
        y = (np.array(traits) == t).astype(int).reshape(-1, 1)
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X, y)

        load = pls.x_loadings_[:, 0]

        df = (
            pd.DataFrame({"QTL_id": features, "loading": load})
            .assign(abs=lambda d: d.loading.abs())
            .sort_values("abs", ascending=False)
            .head(top_n)
        )
        df["trait"] = t
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


# ============================================================
# 7) Save plots (FAST DASHBOARD MODE)
# ============================================================

def save_plsr_plot(df: pd.DataFrame, trait: str, out_dir: str):
    sub = df[df["trait"] == trait]
    if sub.empty:
        return

    plt.figure(figsize=(8, max(2, 0.35 * len(sub))))
    colors = ["tab:red" if x > 0 else "tab:blue" for x in sub["loading"]]

    plt.barh(sub["QTL_id"], sub["loading"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("PLS1 loading")
    plt.title(f"GLOBAL-QTL PLSR — {trait}")
    plt.tight_layout()

    outpath = os.path.join(out_dir, f"{_safe_filename(trait)}.png")
    plt.savefig(outpath, dpi=200)
    plt.close()


# ============================================================
# 8) MAIN PIPELINE (CONFIG-DRIVEN, BATCH)
# ============================================================

def run_global_qtl_plsr_pipeline(
    spark,
    config: dict,
    drop_ch00: bool = True,
) -> dict:

    gwas_table = _cfg(config, ["data", "gwas_table"])
    distance = _cfg(config, ["params", "distance_threshold"], 250_000)
    max_qtls = _cfg(config, ["params", "max_qtls_for_pivot"], 2000)
    n_comp = _cfg(config, ["params", "n_components"], 2)
    top_n = _cfg(config, ["params", "top_n"], 20)

    out_dir = _cfg(
        config,
        ["paths", "plsr_out_dir"],
        "/Volumes/bmqg/default_bronze/fatemeh/final_project/plsr",
    )

    os.makedirs(out_dir, exist_ok=True)

    traits = get_traits(spark, config)

    marker_gqtl, gqtl_bounds = build_global_qtl_blocks(
        spark, traits, gwas_table, distance, drop_ch00
    )

    meta_sdf = build_qtl_meta_global(
        spark, traits, marker_gqtl, gqtl_bounds, gwas_table
    )

    keep_qtls = select_top_qtls(meta_sdf, max_qtls)

    signed_sdf = build_signed_matrix_from_meta(meta_sdf, keep_qtls)
    qtl_pd = signed_sdf.toPandas().set_index("trait")

    all_top = plsr_all_traits(qtl_pd, n_comp, top_n)

    # ---- SAVE ----
    all_top.to_csv(os.path.join(out_dir, "plsr_top_loadings.csv"), index=False)

    for t in traits:
        save_plsr_plot(all_top, t, out_dir)

    return {
        "traits": traits,
        "out_dir": out_dir,
        "qtl_pd_signed": qtl_pd,
        "all_top": all_top,
    }


import os
from pyspark.sql import functions as F

import os
from pyspark.sql import functions as F

import os

def run_fast_plsr_pipeline(spark, config, drop_ch00=True, make_plots=False):
    gwas_table = config["data"]["gwas_table"]

    params = config.get("params", config.get("parameters", {}))
    distance = params.get("distance_threshold", 250_000)
    max_qtls = params.get("max_qtls_for_pivot", 300)
    n_comp = params.get("n_components", 2)
    top_n = params.get("top_n", 10)

    out_dir = config["paths"]["plsr_out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[run_fast_plsr_pipeline] Using GWAS table: {gwas_table}")

    gwas_df = spark.table(gwas_table).cache()
    gwas_df.count()  # materialize cache

    traits = (
        gwas_df.select("trait")
        .distinct()
        .toPandas()["trait"]
        .astype(str)
        .tolist()
    )

    marker_gqtl, gqtl_bounds = build_global_qtl_blocks(
        spark, traits, gwas_table, distance, drop_ch00
    )

    marker_gqtl = marker_gqtl.cache()
    gqtl_bounds = gqtl_bounds.cache()
    marker_gqtl.count()
    gqtl_bounds.count()

    meta_sdf = build_qtl_meta_global(
        spark, traits, marker_gqtl, gqtl_bounds, gwas_table
    ).cache()
    meta_sdf.count()

    keep_qtls = select_top_qtls(meta_sdf, max_qtls).cache()
    keep_qtls.count()

    signed_sdf = build_signed_matrix_from_meta(meta_sdf, keep_qtls).cache()
    signed_sdf.count()

    print("Signed matrix rows:", signed_sdf.count())
    print("Signed matrix columns:", len(signed_sdf.columns))

    qtl_pd = signed_sdf.toPandas().set_index("trait")
    all_top = plsr_all_traits(qtl_pd, n_comp, top_n)

    out_csv = os.path.join(out_dir, "plsr_top_loadings_fast.csv")
    all_top.to_csv(out_csv, index=False)

    if make_plots:
        for t in qtl_pd.index.tolist():
            save_plsr_plot(all_top, t, out_dir)

    return {
        "qtl_pd_signed": qtl_pd,
        "all_top": all_top,
        "out_csv": out_csv,
    }