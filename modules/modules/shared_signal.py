""""
Author: Fatemeh Monfared

This module identifies significant genomic regions from GWAS data by grouping SNPs into genomic windows,
selecting the most significant variant (leader SNP), and evaluating taglotype support. It also computes
consecutive taglotype runs to assess signal consistency and ranks the top regions per trait based on 
statistical strength and structural support.
"""




"""
Author: Fatemeh Monfared

This module identifies significant genomic regions from GWAS data by grouping SNPs into genomic windows,
selecting the most significant variant (leader SNP), and evaluating taglotype support. It also computes
consecutive taglotype runs to assess signal consistency and ranks the top regions per trait based on 
statistical strength and structural support.
"""

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ---------------------------
# Helper: consecutive run
# ---------------------------
def longest_consecutive_run(ids):
    ids = sorted(set([int(x) for x in ids if x is not None]))

    if not ids:
        return 0, []

    best = cur = [ids[0]]

    for i in range(1, len(ids)):
        if ids[i] == ids[i-1] + 1:
            cur.append(ids[i])
        else:
            if len(cur) > len(best):
                best = cur
            cur = [ids[i]]

    if len(cur) > len(best):
        best = cur

    return len(best), best


def safe_consecutive(x):
    if x is None:
        return 0
    try:
        return longest_consecutive_run(list(x))[0]
    except:
        return 0


# ---------------------------
# MAIN PIPELINE (FIXED)
# ---------------------------
def run_pipeline(
    spark,   # ✅ FIX: اضافه شد
    gwas_table,
    taglo_table,
    pheno_df,
    phenotype_col,
    p_thresh=1e-6,
    window_bp=500_000,
    min_uniq_pos=3,
    top_regions_per_trait=5
):

    # ---------------------------
    # 1. Load GWAS
    # ---------------------------
    hits = (
        spark.table(gwas_table)
        .select(
            F.col("trait").cast("string"),
            F.col("chrom").cast("string"),
            F.col("start").cast("long"),
            F.col("p_wald").cast("double"),
            "taglo_id1","taglo_id2","taglo_id3","taglo_id4"
        )
        .dropna(subset=["trait","chrom","start","p_wald"])
        .filter(F.col("p_wald") > 0)
        .filter(F.col("p_wald") <= p_thresh)
        .withColumn("window", (F.col("start") / window_bp).cast("long") * window_bp)
        .withColumn("nlp", -F.log10("p_wald"))
    )

    # ---------------------------
    # 2. Unique positions
    # ---------------------------
    pos_counts = (
        hits
        .select("trait","chrom","window","start")
        .distinct()
        .groupBy("trait","chrom","window")
        .agg(F.count("*").alias("n_unique_pos"))
    )

    # ---------------------------
    # 3. Leader SNP
    # ---------------------------
    w = Window.partitionBy("trait","chrom","window").orderBy(F.asc("p_wald"))

    leaders = (
        hits
        .withColumn("rn", F.row_number().over(w))
        .filter("rn = 1")
        .select(
            "trait","chrom","window",
            F.col("start").alias("leader_start"),
            F.col("p_wald").alias("leader_p"),
            F.col("nlp").alias("leader_nlp"),
            "taglo_id1","taglo_id2","taglo_id3","taglo_id4"
        )
    )

    # ---------------------------
    # 4. Taglo support
    # ---------------------------
    taglo_support = (
        hits
        .select(
            "trait","chrom","window",
            F.explode(F.array(
                "taglo_id1","taglo_id2","taglo_id3","taglo_id4"
            )).alias("taglo_id")
        )
        .filter(F.col("taglo_id").isNotNull())
        .filter(F.col("taglo_id") != "0")
        .withColumn("taglo_id", F.col("taglo_id").cast("int"))
        .groupBy("trait","chrom","window")
        .agg(
            F.countDistinct("taglo_id").alias("n_taglo"),
            F.collect_set("taglo_id").alias("taglo_ids")
        )
    )

    # ---------------------------
    # 5. Combine
    # ---------------------------
    regions = (
        pos_counts
        .join(leaders, ["trait","chrom","window"])
        .join(taglo_support, ["trait","chrom","window"], "left")
        .filter(F.col("n_unique_pos") >= min_uniq_pos)
        .orderBy(F.desc("leader_nlp"), F.desc("n_unique_pos"))
    )

    # ---------------------------
    # 6. To pandas
    # ---------------------------
    regions_pdf = regions.toPandas()

    # ---------------------------
    # 7. Consecutive run
    # ---------------------------
    regions_pdf["consecutive_run_len"] = regions_pdf["taglo_ids"].apply(safe_consecutive)

    regions_pdf["best_consecutive_run"] = regions_pdf["taglo_ids"].apply(
        lambda x: longest_consecutive_run(list(x))[1] if x is not None else []
    )

    # ---------------------------
    # 8. Ranking
    # ---------------------------
    regions_pdf["rank_in_trait"] = (
        regions_pdf
        .sort_values(
            ["trait","leader_nlp","n_unique_pos","consecutive_run_len"],
            ascending=[True, False, False, False]
        )
        .groupby("trait")
        .cumcount() + 1
    )

    top_regions = regions_pdf[
        regions_pdf["rank_in_trait"] <= top_regions_per_trait
    ].copy()

    return top_regions

def get_shared_signals(
    spark,
    gwas_table,
    p_thresh=1e-6
):

    shared = (
        spark.table(gwas_table)
        .select(
            "trait",
            "chrom",
            "start",
            "p_wald",
            "taglo_id1",
            "taglo_id2",
            "taglo_id3",
            "taglo_id4"
        )
        .filter(F.col("p_wald") > 0)
        .filter(F.col("p_wald") <= p_thresh)
        
        # explode taglo
        .withColumn(
            "taglo_id",
            F.explode(F.array(
                "taglo_id1","taglo_id2","taglo_id3","taglo_id4"
            ))
        )
        .filter(F.col("taglo_id").isNotNull())
        .filter(F.col("taglo_id") != "0")

        .withColumn("taglo_id", F.col("taglo_id").cast("int"))
        .withColumn("nlp", -F.log10("p_wald"))
    )

    return shared.toPandas()