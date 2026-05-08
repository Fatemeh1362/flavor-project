"""
Author: Fatemeh Monfared
Module: manhattan_module.py
Title: Config-driven GWAS utilities + Manhattan plotting 


# Config-driven utilities for:
# - loading phenotypes
# - listing traits from GWAS table
# - fetching a lightweight GWAS subset for one trait
# - plotting Manhattan plots
#
# Works in Databricks (Spark) and is thesis-grade & reproducible."""


"""
Author: Fatemeh Monfared
Module: manhattan_module.py
Title: Config-driven GWAS utilities + Manhattan plotting
"""




import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql import DataFrame as SparkDataFrame


# ============================================================
# Config
# ============================================================

_PROJECT_CONFIG = None


def load_project_config(config_path: str) -> dict:
    """Load and cache YAML config."""
    global _PROJECT_CONFIG
    if _PROJECT_CONFIG is None:
        with open(config_path, "r") as f:
            _PROJECT_CONFIG = yaml.safe_load(f)
    return _PROJECT_CONFIG


# ============================================================
# Phenotype helpers (optional)
# ============================================================

def load_phenotypes_csv(pheno_path: str) -> pd.DataFrame:
    """
    Load phenotype matrix (CSV) with 'Variety' column as index.
    Converts trait columns to numeric.
    """
    df = pd.read_csv(pheno_path)

    if "Variety" not in df.columns:
        raise ValueError("Column 'Variety' not found in phenotype file.")

    df["Variety"] = df["Variety"].astype(str).str.strip()
    df = df.set_index("Variety")
    df = df[~df.index.duplicated(keep="first")]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_phenotypes_from_config(config_path: str) -> pd.DataFrame:
    """Load phenotype matrix path from config and read CSV."""
    cfg = load_project_config(config_path)
    pheno_path = cfg["paths"]["aroma_matrix"]
    return load_phenotypes_csv(pheno_path)


# ============================================================
# Trait listing / GWAS loading
# ============================================================

def get_traits(
    spark,
    gwas_table: str = None,
    config_path: str = None,
    trait_col_candidates=("trait", "phenotype"),
) -> list:
    """
    Get distinct trait names from GWAS table (recommended source for Manhattan).
    """
    if gwas_table is None:
        if config_path is None:
            raise ValueError("Provide either gwas_table or config_path.")
        cfg = load_project_config(config_path)
        gwas_table = cfg["data"]["gwas_table"]

    df0 = spark.table(gwas_table)

    trait_col = None
    for c in trait_col_candidates:
        if c in df0.columns:
            trait_col = c
            break
    if trait_col is None:
        raise ValueError(f"No trait column found. Tried: {trait_col_candidates}")

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
    source_table: str = None,
    config_path: str = None,
    trait_col_candidates=("trait", "phenotype"),
    chrom_col: str = "chrom",
    pos_col: str = "start",
    p_col: str = "p_wald",
    beta_col: str = None,
    extra_cols: list = None,
    p_sig: float = 1e-6,
    random_fraction: float = 0.02,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Load a lightweight GWAS subset for one trait:
      - all significant rows (p < p_sig)
      - random sample of NON-significant rows as background

    Output normalized columns:
      chrom, start, p_wald, [beta], [extra_cols...]

    Notes:
    - Avoids duplicate significant rows in background.
    - Removes invalid p-values (<= 0).
    """
    if source_table is None:
        if config_path is None:
            raise ValueError("Provide either source_table or config_path.")
        cfg = load_project_config(config_path)
        source_table = cfg["data"]["gwas_table"]

    df0: SparkDataFrame = spark.table(source_table)

    # trait col detect
    trait_col = None
    for c in trait_col_candidates:
        if c in df0.columns:
            trait_col = c
            break
    if trait_col is None:
        raise ValueError(f"No trait column found. Tried: {trait_col_candidates}")

    # validate required columns
    required = [chrom_col, pos_col, p_col]
    missing = [c for c in required if c not in df0.columns]
    if missing:
        raise ValueError(f"GWAS table missing required columns: {missing}")

    select_exprs = [
        F.col(chrom_col).cast("string").alias("chrom"),
        F.col(pos_col).cast("long").alias("start"),
        F.col(p_col).cast("double").alias("p_wald"),
    ]

    dedup_cols = ["chrom", "start", "p_wald"]

    # optional beta
    has_beta = False
    if beta_col is not None:
        if beta_col not in df0.columns:
            raise ValueError(f"beta_col='{beta_col}' not found in GWAS table.")
        select_exprs.append(F.col(beta_col).cast("double").alias("beta"))
        dedup_cols.append("beta")
        has_beta = True

    # optional extras (e.g. taglo_id or taglo_id1..taglo_id4)
    extra_cols = extra_cols or []
    missing_extra = [c for c in extra_cols if c not in df0.columns]
    if missing_extra:
        raise ValueError(f"GWAS table missing extra columns: {missing_extra}")

    for c in extra_cols:
        select_exprs.append(F.col(c))
        if c not in dedup_cols:
            dedup_cols.append(c)

    df = (
        df0.filter(F.col(trait_col) == F.lit(str(trait)))
           .select(*select_exprs)
           .dropna(subset=["chrom", "start", "p_wald"])
           .filter(F.col("p_wald") > 0)
    )

    # all significant
    df_sig = df.filter(F.col("p_wald") < F.lit(float(p_sig)))

    # background only from NON-significant rows
    df_bg = df.filter(F.col("p_wald") >= F.lit(float(p_sig)))
    df_rand = df_bg.sample(withReplacement=False, fraction=float(random_fraction), seed=int(seed))

    out = (
        df_sig.unionByName(df_rand)
              .dropDuplicates(dedup_cols)
              .toPandas()
    )

    return out


# ============================================================
# Internal helpers
# ============================================================

def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(s))


def _natural_chrom_key(chrom: str):
    """
    Robust chromosome sorting:
    handles ch1, chr1, 1, ST4.03ch01 ...
    """
    s = str(chrom).strip()
    sl = s.lower()

    m = re.search(r"(?:chr|ch)\s*0*(\d+)", sl)
    if m:
        prefix = re.sub(r"(?:chr|ch)\s*0*\d+", "ch", sl)
        return (0, prefix, int(m.group(1)))

    m2 = re.search(r"0*(\d+)\s*$", sl)
    if m2:
        prefix = re.sub(r"0*\d+\s*$", "", sl)
        return (1, prefix, int(m2.group(1)))

    return (2, sl, 10**9)


def _prepare_manhattan_dataframe(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and constructs cumulative genomic positions.
    Expected columns: chrom, start, p_wald
    Optional columns (preserved): beta, taglo_id, taglo_id1..4, etc.
    """
    if pdf is None or len(pdf) == 0:
        return pd.DataFrame()

    df = pdf.copy()

    # Flatten possible list-like values from Spark conversions
    for c in df.columns:
        df[c] = df[c].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

    # Required fields
    required_cols = {"chrom", "start", "p_wald"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Input dataframe must contain: chrom, start, p_wald")

    # Type coercion for required fields
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["p_wald"] = pd.to_numeric(df["p_wald"], errors="coerce")

    # Optional numeric columns
    if "beta" in df.columns:
        df["beta"] = pd.to_numeric(df["beta"], errors="coerce")

    # Drop invalid required rows
    df = df.dropna(subset=["chrom", "start", "p_wald"]).copy()
    df = df[df["p_wald"] > 0].copy()

    if df.empty:
        return df

    df["start"] = df["start"].astype(np.int64)
    df["minus_log10_p"] = -np.log10(df["p_wald"].clip(lower=1e-300))

    # chromosome ordering
    chroms = sorted(df["chrom"].unique(), key=_natural_chrom_key)
    chrom_index = {c: i for i, c in enumerate(chroms)}
    df["chrom_numeric"] = df["chrom"].map(chrom_index)

    # sort by chromosome + position
    df = df.sort_values(["chrom_numeric", "start"]).copy()

    # cumulative offsets
    chrom_max = df.groupby("chrom")["start"].max()
    offsets = (chrom_max.cumsum() - chrom_max).to_dict()
    df["cum_pos"] = df["start"] + df["chrom"].map(offsets).astype(np.int64)

    return df


def _pick_lead_taglo_from_row(row, taglo_cols):
    """
    Pick a lead taglo from a row using the first non-null / non-empty value
    among taglo_cols.
    """
    for c in taglo_cols:
        if c in row.index:
            v = row[c]
            if pd.notna(v):
                s = str(v).strip()
                if s and s.lower() != "nan":
                    return s
    return None


# ============================================================
# Plotting functions
# ============================================================

def plot_manhattan(
    pdf: pd.DataFrame,
    trait: str,
    p_threshold: float = 1e-6,
    point_size: int = 7,
    figsize=(18, 6),
    save_path: str = None,
    show: bool = True,
):
    """
    Standard Manhattan plot.
    Input columns: chrom, start, p_wald
    """
    df = _prepare_manhattan_dataframe(pdf)
    if df.empty:
        return df

    chroms = sorted(df["chrom"].unique(), key=_natural_chrom_key)
    colors = ["#4C72B0", "#55A868"]
    chrom_index = {c: i for i, c in enumerate(chroms)}

    plt.figure(figsize=figsize)

    for chrom in chroms:
        sub = df[df["chrom"] == chrom]
        idx = chrom_index[chrom]
        plt.scatter(
            sub["cum_pos"],
            sub["minus_log10_p"],
            s=point_size,
            color=colors[idx % 2],
            alpha=0.85,
            linewidths=0,
        )

    # highlight significant points
    sig = df[df["p_wald"] < float(p_threshold)]
    if not sig.empty:
        plt.scatter(
            sig["cum_pos"],
            sig["minus_log10_p"],
            s=max(point_size + 6, 12),
            alpha=0.95,
            linewidths=0,
            label="p < threshold"
        )

    y_thr = -np.log10(float(p_threshold))
    plt.axhline(y_thr, color="red", linestyle="--", linewidth=1.2)

    ticks = [df[df["chrom"] == c]["cum_pos"].median() for c in chroms]
    plt.xticks(ticks, chroms, rotation=45, ha="right")

    plt.xlabel("Chromosome")
    plt.ylabel("-log10(p)")
    plt.title(f"Manhattan Plot - {trait}")
    if not sig.empty:
        plt.legend()
    plt.tight_layout()

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return df


def plot_manhattan_signed_beta(
    pdf: pd.DataFrame,
    trait: str,
    p_threshold: float = 1e-6,
    figsize=(18, 6),
    background_point_size: int = 5,
    pos_point_size: int = 18,
    neg_point_size: int = 30,
    save_path: str = None,
    show: bool = True,
):
    """
    Beta-signed Manhattan plot.
    Input columns: chrom, start, p_wald, beta
    """
    if "beta" not in pdf.columns:
        raise ValueError("plot_manhattan_signed_beta requires column 'beta'.")

    df = _prepare_manhattan_dataframe(pdf)
    if df.empty:
        return df

    # Drop rows where beta failed numeric conversion
    df = df.dropna(subset=["beta"]).copy()
    if df.empty:
        return df

    sig = df[df["p_wald"] < float(p_threshold)]
    pos = sig[sig["beta"] > 0]
    neg = sig[sig["beta"] < 0]

    plt.figure(figsize=figsize)

    # background
    plt.scatter(
        df["cum_pos"], df["minus_log10_p"],
        s=background_point_size,
        alpha=0.15,
        zorder=1,
        label="background"
    )

    # positive beta
    if not pos.empty:
        plt.scatter(
            pos["cum_pos"], pos["minus_log10_p"],
            s=pos_point_size,
            alpha=0.90,
            zorder=2,
            label="beta > 0"
        )

    # negative beta
    if not neg.empty:
        plt.scatter(
            neg["cum_pos"], neg["minus_log10_p"],
            s=neg_point_size,
            alpha=1.00,
            zorder=3,
            label="beta < 0"
        )

    y_thr = -np.log10(float(p_threshold))
    plt.axhline(y_thr, color="red", linestyle="--", linewidth=1.2)

    chroms = sorted(df["chrom"].unique(), key=_natural_chrom_key)
    ticks = [df[df["chrom"] == c]["cum_pos"].median() for c in chroms]
    plt.xticks(ticks, chroms, rotation=45, ha="right")

    plt.xlabel("Chromosome")
    plt.ylabel("-log10(p)")
    plt.title(f"Manhattan Plot (beta-signed) - {trait}")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return df


def plot_manhattan_tagloid(
    pdf,
    trait,
    taglo_col="taglo_id",
    highlight_taglo_ids=None,
    p_threshold=1e-6,
    figsize=(18, 6),
    background_point_size=5,
    highlight_point_size=24,
    annotate=True,
    annotate_top_only=True,
    annotate_points_df=None,
    save_path=None,
    show=True,
):
    if taglo_col not in pdf.columns:
        raise ValueError(f"plot_manhattan_tagloid requires column '{taglo_col}'.")

    df = _prepare_manhattan_dataframe(pdf)
    if df.empty:
        return df

    df[taglo_col] = df[taglo_col].astype(str).str.strip()
    df = df[~df[taglo_col].str.lower().isin(["nan", "none", "null", ""])].copy()
    if df.empty:
        return df

    plt.figure(figsize=figsize)

    # background
    plt.scatter(
        df["cum_pos"], df["minus_log10_p"],
        s=background_point_size, alpha=0.15, zorder=1, label="background"
    )

    # threshold
    y_thr = -np.log10(float(p_threshold))
    plt.axhline(y_thr, color="red", linestyle="--", linewidth=1.2)

    # highlight selected taglo IDs
    hi = pd.DataFrame()
    if highlight_taglo_ids:
        target_set = {str(x).strip() for x in highlight_taglo_ids if x is not None}
        hi = df[df[taglo_col].isin(target_set)].copy()

        if not hi.empty:
            plt.scatter(
                hi["cum_pos"], hi["minus_log10_p"],
                s=highlight_point_size, alpha=0.95, zorder=3, label="highlight taglo_id"
            )

    # ---- FIXED ANNOTATION LOGIC ----
    labels_df = pd.DataFrame()

    # Preferred: annotate exact lead SNP row(s) by matching on chrom/start/p_wald in df
    if annotate and annotate_points_df is not None and len(annotate_points_df) > 0:
        ann = annotate_points_df.copy()

        # normalize types to match df
        ann["chrom"] = ann["chrom"].astype(str)
        ann["start"] = pd.to_numeric(ann["start"], errors="coerce")
        ann["p_wald"] = pd.to_numeric(ann["p_wald"], errors="coerce")
        if taglo_col in ann.columns:
            ann[taglo_col] = ann[taglo_col].astype(str).str.strip()

        ann = ann.dropna(subset=["chrom", "start", "p_wald"]).copy()
        if not ann.empty and taglo_col in ann.columns:
            ann["start"] = ann["start"].astype("int64")

            # match exact plotted points
            labels_df = df.merge(
                ann[["chrom", "start", "p_wald", taglo_col]].drop_duplicates(),
                on=["chrom", "start", "p_wald", taglo_col],
                how="inner"
            )

    # Fallback: annotate from highlighted points
    if labels_df.empty and annotate and not hi.empty:
        if annotate_top_only:
            labels_df = (
                hi.sort_values("minus_log10_p", ascending=False)
                  .groupby(taglo_col, as_index=False)
                  .first()
            )
        else:
            labels_df = hi.copy()

    # Draw labels on actual plotted coordinates
    if annotate and not labels_df.empty:
        for _, r in labels_df.iterrows():
            plt.annotate(
                str(r[taglo_col]),
                xy=(r["cum_pos"], r["minus_log10_p"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=1.0,
                zorder=5
            )

    chroms = sorted(df["chrom"].unique(), key=_natural_chrom_key)
    ticks = [df[df["chrom"] == c]["cum_pos"].median() for c in chroms]
    plt.xticks(ticks, chroms, rotation=45, ha="right")

    plt.xlabel("Chromosome")
    plt.ylabel("-log10(p)")
    plt.title(f"Manhattan Plot (taglo_id) - {trait}")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return df

# ============================================================
# Batch runners
# ============================================================

def run_manhattan_for_traits(
    spark,
    traits: list,
    out_dir: str,
    source_table: str = None,
    config_path: str = None,
    p_threshold: float = 1e-6,
    random_fraction: float = 0.02,
    seed: int = 123,
    trait_col_candidates=("trait", "phenotype"),
    chrom_col: str = "chrom",
    pos_col: str = "start",
    p_col: str = "p_wald",
    beta_col: str = None,
    signed_beta: bool = False,
    show: bool = False,
):
    """
    Batch run standard or beta-signed Manhattan plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    n_done, n_empty, n_err = 0, 0, 0

    for i, trait in enumerate(traits, 1):
        try:
            pdf = load_trait_gwas_subset(
                spark=spark,
                trait=trait,
                source_table=source_table,
                config_path=config_path,
                trait_col_candidates=trait_col_candidates,
                chrom_col=chrom_col,
                pos_col=pos_col,
                p_col=p_col,
                beta_col=beta_col if signed_beta else None,
                p_sig=p_threshold,
                random_fraction=random_fraction,
                seed=seed,
            )

            if pdf.empty:
                n_empty += 1
                continue

            save_path = os.path.join(out_dir, f"{i:02d}_{_safe_name(trait)}.png")

            if signed_beta:
                plot_manhattan_signed_beta(
                    pdf=pdf,
                    trait=trait,
                    p_threshold=p_threshold,
                    save_path=save_path,
                    show=show,
                )
            else:
                plot_manhattan(
                    pdf=pdf,
                    trait=trait,
                    p_threshold=p_threshold,
                    save_path=save_path,
                    show=show,
                )

            n_done += 1

            if i % 20 == 0:
                print(f"[{i}/{len(traits)}] plotted so far: {n_done}")

        except Exception as e:
            n_err += 1
            print(f"[ERROR] trait={trait}: {e}")

    print("\nDone.")
    print(f"Plotted: {n_done}/{len(traits)}")
    print(f"Empty:   {n_empty}")
    print(f"Errors:  {n_err}")
    print(f"Saved to: {out_dir}")


def run_manhattan_lead_taglo_for_traits(
    spark,
    traits,
    out_dir,
    source_table=None,
    config_path=None,
    p_threshold=1e-6,
    random_fraction=0.02,
    seed=123,
    trait_col_candidates=("trait", "phenotype"),
    chrom_col="chrom",
    pos_col="start",
    p_col="p_wald",
    taglo_cols=None,
    show=False,
):
    """
    Batch Manhattan plots highlighting the lead taglo per trait.

    Supports either:
    - a single taglo column: ["taglo_id"]
    - multiple taglo columns: ["taglo_id1", "taglo_id2", "taglo_id3", "taglo_id4"]

    Lead taglo is selected from the most significant SNP row (smallest p-value),
    taking the first non-null taglo in taglo_cols.
    """
    os.makedirs(out_dir, exist_ok=True)

    if taglo_cols is None:
        taglo_cols = ["taglo_id"]

    n_done, n_empty, n_err = 0, 0, 0

    for i, trait in enumerate(traits, 1):
        try:
            pdf = load_trait_gwas_subset(
                spark=spark,
                trait=trait,
                source_table=source_table,
                config_path=config_path,
                trait_col_candidates=trait_col_candidates,
                chrom_col=chrom_col,
                pos_col=pos_col,
                p_col=p_col,
                beta_col=None,
                extra_cols=taglo_cols,
                p_sig=p_threshold,
                random_fraction=random_fraction,
                seed=seed,
            )

            if pdf.empty:
                n_empty += 1
                continue

            # p-value parsing
            pvals = pd.to_numeric(pdf["p_wald"], errors="coerce")
            if pvals.isna().all():
                n_empty += 1
                continue

            # lead SNP row = smallest p-value
            lead_idx = pvals.idxmin()
            lead_row_series = pdf.loc[lead_idx]        # Series (for picking lead taglo)
            lead_row_df = pdf.loc[[lead_idx]].copy()   # 1-row DataFrame (for exact annotation)

            lead_taglo = _pick_lead_taglo_from_row(lead_row_series, taglo_cols)

            if lead_taglo is None:
                n_empty += 1
                continue

            # build plotting dataframe / plotting taglo column
            plot_pdf = pdf.copy()

            if len(taglo_cols) == 1:
                plot_taglo_col = taglo_cols[0]
            else:
                # create a helper single taglo column from first non-null taglo across taglo_cols
                plot_taglo_col = "_lead_taglo_tmp"
                plot_pdf[plot_taglo_col] = plot_pdf.apply(
                    lambda r: _pick_lead_taglo_from_row(r, taglo_cols), axis=1
                )
                lead_row_df[plot_taglo_col] = lead_row_df.apply(
                    lambda r: _pick_lead_taglo_from_row(r, taglo_cols), axis=1
                )

            save_path = os.path.join(out_dir, f"{i:02d}_{_safe_name(trait)}.png")

            # IMPORTANT: call the separate plot function (do not define it here)
            plot_manhattan_tagloid(
                pdf=plot_pdf,
                trait=trait,
                taglo_col=plot_taglo_col,
                highlight_taglo_ids=[str(lead_taglo)],
                p_threshold=p_threshold,
                annotate=True,
                annotate_top_only=True,
                annotate_points_df=lead_row_df,   # exact lead point gets label
                save_path=save_path,
                show=show,
            )

            n_done += 1

            if i % 20 == 0:
                print(f"[{i}/{len(traits)}] plotted so far: {n_done}")

        except Exception as e:
            n_err += 1
            print(f"[ERROR] trait={trait}: {e}")

    print("\nDone.")
    print(f"Plotted: {n_done}/{len(traits)}")
    print(f"Empty:   {n_empty}")
    print(f"Errors:  {n_err}")
    print(f"Saved to: {out_dir}")