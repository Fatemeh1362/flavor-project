"""
Author: Fatemeh Monfared
Module: compound_analysis.py
This module provides reusable utilities for compound-level GC–MS analysis that a company team can plug into notebooks or pipelines. It combines NIST/compound CSV outputs (samples + blanks), performs contamination filtering (blank/env/qc vs sample within RT tolerance), and then matches identified compounds to corrected peak tables (RT ± m/z tolerances) . Linking with sensory data , sensory predction and  generates summary plots and intensity matrices for downstream statistics and reporting."""

import os
import glob
import pandas as pd
from difflib import get_close_matches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import re
from difflib import get_close_matches
from collections import OrderedDict, defaultdict
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial import distance
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
import threading
import socket
import base64
from io import BytesIO


def combine_compound_data(sample_files, base_dir, output_path, blank_files=None):
    """
    Concatenate ONLY sample + blank GC-MS compound CSV files into one DataFrame.
    Adds FileType (sample/blank) + FileName + FilePath.
    """

    def load_file(file_path, file_type):
        try:
            df = pd.read_csv(file_path)
            df["FileType"] = file_type
            df["FileName"] = os.path.basename(file_path)
            df["FilePath"] = file_path
            return df
        except Exception as e:
            print(f"Error reading {file_type} file: {file_path}\n{e}")
            return pd.DataFrame()

    all_data = []

    for f in (sample_files or []):
        all_data.append(load_file(f, "sample"))

    for f in (blank_files or []):
        all_data.append(load_file(f, "blank"))

    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined file saved to: {output_path}")
    if "FileType" in combined_df.columns:
        print(combined_df["FileType"].value_counts(dropna=False))
    print(combined_df.head())

    return combined_df





def plot_filetype_counts(df, column="FileType", title="Number of Files per Type"):
    """
    Plot the count of different file types in a GC–MS dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the file type information.
    column : str, optional
        Name of the column containing file type labels (default: "FileType").
    title : str, optional
        Title for the plot.
    """
    if column not in df.columns:
        raise ValueError(f" Column '{column}' not found in DataFrame")

    # Count file types
    file_type_counts = df[column].value_counts()

    # Bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(
        file_type_counts.index,
        file_type_counts.values,
        color=["#4C72B0", "#FFA500", "#55A868", "#C44E52"]
    )
    plt.title(title, fontsize=12)
    plt.xlabel("File Type")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(" File type count plot generated successfully.")
    print(file_type_counts)



def filter_real_compounds(compound_df, rt_tolerance=0.4):
    """
    Removes sample compounds that also appear in blank, env, or QC files
    within a given RT (retention time) tolerance.

    Parameters
    ----------
    compound_df : DataFrame
        Combined compound data containing 'Component RT', 'Compound Name', and 'FileType'.
    rt_tolerance : float
        Allowed RT difference (in minutes) to consider a match (default = 0.3).

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only sample compounds not matching contaminants.
    """

    # Separate by file type
    sample_df = compound_df[compound_df["FileType"] == "sample"].copy()
    contam_df = compound_df[compound_df["FileType"].isin(["blank", "env", "qc"])].copy()

    print(f"Sample compounds before filtering: {len(sample_df)}")

    # Ensure numeric RT
    sample_df["Component RT"] = pd.to_numeric(sample_df["Component RT"], errors="coerce")
    contam_df["Component RT"] = pd.to_numeric(contam_df["Component RT"], errors="coerce")

    # Sort to speed up comparisons
    sample_df = sample_df.sort_values("Component RT")
    contam_df = contam_df.sort_values("Component RT")

    # Identify contaminated peaks
    contaminated = []
    for _, c in contam_df.iterrows():
        rt = c["Component RT"]
        name = c["Compound Name"]
        # find sample peaks within RT tolerance and with same compound name
        matches = sample_df[
            (np.abs(sample_df["Component RT"] - rt) <= rt_tolerance)
            & (sample_df["Compound Name"].str.lower() == name.lower())
        ].index
        contaminated.extend(matches)

    contaminated = list(set(contaminated))

    # Remove contaminated ones
    filtered_df = sample_df.drop(index=contaminated)
    print(f"Sample compounds after filtering: {len(filtered_df)}")
    print(f"Removed {len(contaminated)} sample peaks (ΔRT ≤ {rt_tolerance})")

    return filtered_df



def plot_rt_distribution(compound_df, real_compounds, bins=40):
    """
    Plot retention time (RT) distribution before and after contamination filtering.

    Parameters
    ----------
    compound_df : pandas.DataFrame
        Original dataset containing all sample entries.
    real_compounds : pandas.DataFrame
        Filtered dataset after removing contaminations.
    bins : int, optional
        Number of histogram bins (default: 40).
    """
    plt.figure(figsize=(6, 4))
    plt.hist(
        compound_df[compound_df["FileType"] == "sample"]["Component RT"],
        bins=bins, alpha=0.5, label="Before filtering", color="#4C72B0"
    )
    plt.hist(
        real_compounds["Component RT"],
        bins=bins, alpha=0.7, label="After filtering", color="#55A868"
    )
    plt.xlabel("Retention Time (min)")
    plt.ylabel("Count")
    plt.title("RT Distribution Before and After Filtering")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(" RT distribution plot generated successfully.")


def plot_filtering_effect(compound_df, real_compounds, rt_tolerance=0.5):
    """
    Plot the number of sample compounds before and after filtering.

    Parameters
    ----------
    compound_df : pandas.DataFrame
        Original dataset containing all samples.
    real_compounds : pandas.DataFrame
        Filtered dataset after removing contaminants.
    rt_tolerance : float, optional
        Retention time tolerance used for filtering (default: 0.5).
    """
    before = compound_df[compound_df["FileType"] == "sample"].shape[0]
    after = real_compounds.shape[0]

    plt.figure(figsize=(5, 4))
    plt.bar(
        ["Before filtering", "After filtering"],
        [before, after],
        color=["#4C72B0", "#55A868"]
    )
    plt.ylabel("Number of sample compound entries")
    plt.title(f"Effect of Contamination Filtering (RT ±{rt_tolerance} min)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f" Filtering effect plot generated successfully. (Before: {before}, After: {after})")




def match_compounds_to_peaks(
    compound_df,
    samples_filtered_path,
    output_path,
    rt_tolerance=0.4
):
    """
    Match NIST-identified compounds with GC–MS peaks based on retention time (RT).
    """

    # Load filtered samples
    samples_filtered = pd.read_csv(samples_filtered_path)

    # Handle missing 'Filename' column
    if "Filename" in samples_filtered.columns:
        samples_filtered = samples_filtered[
            ~samples_filtered['Filename'].str.contains("QC|ENV|BLANK", case=False, na=False)
        ].copy()
    else:
        print(" 'Filename' column not found — skipping QC/blank filtering.")

    print(f"Samples kept after filtering: {samples_filtered['Variety'].nunique()} unique varieties")

    # Clean compound filenames
    compound_df["Variety"] = compound_df["FileName"].str.replace(".csv", "", regex=False).str.strip()

    # Helper: find closest peak
    def find_closest_peak(rt, peaks_rt, tol=rt_tolerance):
        diffs = np.abs(peaks_rt - rt)
        if diffs.empty:
            return None
        min_diff = diffs.min()
        return diffs.idxmin() if min_diff <= tol else None

    matches, unmatched = [], []

    for _, row in compound_df.iterrows():
        variety = row["Variety"]
        sample_data = samples_filtered[samples_filtered["Variety"] == variety]
        if sample_data.empty:
            unmatched.append({**row, "Reason": "No rows for this variety"})
            continue

        compound_rt = row["Component RT"]
        peak_idx = find_closest_peak(compound_rt, sample_data["tR_best"], tol=rt_tolerance)

        if peak_idx is None:
            unmatched.append({**row, "Reason": "No matching RT"})
            continue

        peak_row = sample_data.loc[peak_idx]
        matches.append({
            "Compound Name": row["Compound Name"],
            "FileName": row["FileName"],
            "Variety": variety,
            "Component RT": compound_rt,
            "Peak": peak_row["Peak"],
            "tR_best": peak_row["tR_best"],
            "m/z": peak_row["m/z"],
            "Intensity": peak_row["Intensity_corrected"],
        })

    compound_intensity_df = pd.DataFrame(matches)
    unmatched_df = pd.DataFrame(unmatched)

    compound_intensity_df.to_csv(output_path, index=False)
    print(f"\n Matched {len(compound_intensity_df)} compounds — saved to: {output_path}")

    return compound_intensity_df, unmatched_df




# =========================================================
# Potato aroma lists
# =========================================================
POTATO_AROMAS_CORE = [
    "Hexanal","Heptanal","Octanal","Nonanal","Decanal",
    "Pentanal","Propanal",
    "2-Methylbutanal","3-Methylbutanal","Butanal, 3-methyl-",
    "4-Heptenal","2-Pentenal","2-Hexenal","2-Octenal","2-Nonenal","2-Heptenal",
    "2-Trans-nonenal","2-Trans-octenal",
    "2,4-Heptadienal","2,4-Octadienal",
    "(E, E)-2,4-Nonadienal","(E, Z)-2,4-Decadienal","(E, E)-2,4-Decadienal","2,4-Nonadienal",
    "Benzaldehyde","2-methyl-Benzaldehyde","4-methyl-Benzaldehyde",
    "Benzeneacetaldehyde","Phenylacetaldehyde",

    "1-Heptanol","1-Nonanol","1-Octen-3-ol","1-Pentanol","3-Nonen-1-ol",
    "Hexanol","Phenylethyl alcohol","1-Phenylethanol",
    "2-Methyl-1-propanol","3-Methyl-1-butanol","Ethanol","Benzyl alcohol",

    "2-Heptanone","2-Nonanone","3-Heptanone","3-Octanone","3-Octen-2-one","1-Penten-3-one",
    "2,3-Butanedione","2,3-Pentanedione","Acetophenone","4-Methyl-2-pentanone",
    "(E, E)-3,5-Octadien-2-one","3,5-Octadien-2-one",

    "Acetic acid","Butanoic acid","Hexanoic acid","Isobutyric acid","Isovaleric acid",
    "Propanoic acid","Octanoic acid","2-Octenoic acid","Decanoic acid","nonanoic acid",

    "Methanethiol","Methional",
    "Dimethyl sulfide","Dimethyl disulfide","Dimethyl trisulfide","Dimethyl tetrasulfide",
    "Carbon disulfide","Hydrogen sulfide",
    "2-Methyl-3-furanthiol","3-Methylthiopropanal","3-(Methylthio)propanal",
    "2-Acetylthiazole","2-Methyl-3-thiazoline","2-Methylthiazole","Allyl methyl sulfide",

    "Ethyl acetate","Isoamyl acetate","Ethyl butanoate","Ethyl hexanoate","Methyl butanoate",
    "Ethyl 2-methylbutanoate","Ethyl propanoate","Methyl propanoate","Ethyl pentanoate",

    "2-Acetylpyrazine","2-Ethyl-3,5-dimethylpyrazine",
    "2-Isobutyl-3-methoxypyrazine","2-Isopropyl-3-methoxypyrazine",
    "2-Ethyl-3-methylpyrazine","Trimethylpyrazine","Tetramethylpyrazine",
    "2,5-Dimethylpyrazine","2,6-Dimethylpyrazine","2-Methylpyrazine",

    "Furfural","5-Methylfurfural","3-Furaldehyde",
    "methyl-Cyclopentane","n-Hexane",
    "2-n-Butyl furan","cis-2-(2-Pentenyl) furan",
    "2-pentyl-Furan","Furan, 2-pentyl-","2-Pentylfuran","2-(2-propenyl)-Furan",
    "1,2-dimethoxy-Benzene","2-methoxy-Phenol","Methyl salicylate","Mequinol",

    "Copaene","trans-beta-Ionone","Linalool","alpha-Terpineol",".alpha.-Terpineol",
    "Dimethyl phthalate",
    "Undecanal","Dodecanal",
    "(E)-2-Hexenal","(E)-2-Heptenal","(E)-2-Decenal",
    "2-Octenal, (E)-","2-Nonenal, (E)-",
    "2(3H)-Furanone, dihydro-5-pentyl-",
]

POTATO_AROMAS_OPTIONAL = [
    "Phenol",
    "Benzoic acid",
    "Benzoic acid, methyl ester",

    "Eucalyptol",
    "Limonene",
    "3-Carene",

    "Tridecanal",
    "Tetradecanal",
    "2-Undecenal",
    "2-Tridecenal",

    "6-Methyl-5-hepten-2-one",
    "2-Methylpropanal",
    "1-Octyn-3-ol",
    "3-Methylfuran",
    "2-Ethylfuran",
    "2-Acetyl-5-methylfuran",

    "Acetic acid, ethenyl ester",
    "Acetic acid, methyl ester",
    "Butanoic acid, butyl ester",
    "Hexanoic acid, methyl ester",
    "Octanoic acid, methyl ester",
    "Dodecanoic acid",
    "Dodecanoic acid, 1-methylethyl ester",

    "1-Hexanol, 2-ethyl-",
    "Styrene",
    "n-Hexadecanoic acid",
    "Ethanol, 2-phenoxy-",
    "Tridecane",
    "Tetradecane",
    "2-Propanone, 1-methoxy-",
    "2-Decanone",
    "3-Heptanone, 4-methyl-",
    "Methyl 8-oxooctanoate",
    "Nonanoic acid, 9-oxo-, methyl ester",
]

POTATO_AROMAS = list(OrderedDict.fromkeys(POTATO_AROMAS_CORE + POTATO_AROMAS_OPTIONAL))


ALIASES = {
    "Phenylacetaldehyde": ["benzeneacetaldehyde"],
    "Phenylethyl alcohol": ["2phenylethanol", "phenethyl alcohol", "phenylethanol", "2-phenylethanol"],
    "alpha-Terpineol": [".alpha.-terpineol", "alphaterpineol", "alpha terpineol"],
    "Furan, 2-pentyl-": ["2pentylfuran", "2-pentylfuran", "2-pentyl furan", "furan, 2-pentyl"],

    "Dimethyl disulfide": ["disulfide, dimethyl"],
    "Dimethyl sulfide": ["sulfide, dimethyl"],
    "Dimethyl trisulfide": ["trisulfide, dimethyl"],

    "Butanal, 3-methyl-": ["3-methylbutanal", "butanal,3-methyl-", "3-methyl butanal"],
    "2-Octenal, (E)-": ["(e)-2-octenal", "2-octenal (e)", "2-octenal, (e)-"],
    "2-Nonenal, (E)-": ["(e)-2-nonenal", "2-nonenal (e)", "2-nonenal, (e)-"],

    "2-Methylbutanal": ["butanal, 2-methyl-", "2-methyl butanal", "2-methylbutyraldehyde"],
    "3-Methylbutanal": ["butanal, 3-methyl-", "isovaleraldehyde", "3-methyl butanal"],

    "6-Methyl-5-hepten-2-one": ["5-hepten-2-one, 6-methyl-", "sulcatone"],
    "2-Methylpropanal": ["propanal, 2-methyl-", "isobutyraldehyde"],

    "2-Undecenal": ["(e)-2-undecenal", "2-undecenal (e)", "2-undecenal, (e)-"],
    "2-Tridecenal": ["(e)-2-tridecenal", "2-tridecenal (e)", "2-tridecenal, (e)-"],

    "3-Methylfuran": ["furan, 3-methyl-"],
    "2-Ethylfuran": ["furan, 2-ethyl-"],

    "Acetic acid, ethenyl ester": ["vinyl acetate"],
    "Acetic acid, methyl ester": ["methyl acetate"],
    "Butanoic acid, butyl ester": ["butyl butanoate"],
    "Hexanoic acid, methyl ester": ["methyl hexanoate"],
    "Octanoic acid, methyl ester": ["methyl octanoate"],
    "Dodecanoic acid, 1-methylethyl ester": ["isopropyl dodecanoate"],

    "1-Hexanol, 2-ethyl-": ["2-ethyl-1-hexanol", "2-ethylhexanol"],
    "n-Hexadecanoic acid": ["palmitic acid", "hexadecanoic acid"],
    "Ethanol, 2-phenoxy-": ["phenoxyethanol", "2-phenoxyethanol"],

    "Benzoic acid, methyl ester": ["methyl benzoate"],
    "Eucalyptol": ["1,8-cineole", "cineole"],
    "Limonene": ["d-limonene", "dl-limonene"],
    "3-Carene": ["delta-3-carene", "δ-3-carene"],

    "2-Propanone, 1-methoxy-": ["1-methoxy-2-propanone", "propylene glycol monomethyl ether"],
}


def _strip_stereo(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\(\s*[EeZz]\s*,\s*[EeZz]\s*\)", "", s)
    s = re.sub(r"\(\s*[EeZz]\s*\)", "", s)
    s = re.sub(r"\b(cis|trans)\b\s*-?", "", s, flags=re.IGNORECASE)
    return s

def _normalize(s: str) -> str:
    s = _strip_stereo(s).lower().strip()
    return re.sub(r"[^a-z0-9]+", "", s)

def _build_name_mapper(aroma_list):
    by_norm = defaultdict(list)
    for x in aroma_list:
        by_norm[_normalize(x)].append(x)

    def pick_canonical(names):
        prefs = ["(E, E)", "(E, Z)", "(E)-", "alpha-", ".alpha.-", ", (E)-"]
        for p in prefs:
            for n in names:
                if p in n:
                    return n
        return sorted(names, key=lambda s: (len(s), s))[0]

    canon_norm = {k: pick_canonical(v) for k, v in by_norm.items()}

    syn_to_canon = {}
    for norm_key, names in by_norm.items():
        canon = canon_norm[norm_key]
        for n in names:
            syn_to_canon[_normalize(n)] = canon

    for canon, alts in ALIASES.items():
        for a in alts:
            syn_to_canon[_normalize(a)] = canon

    canon_keys = list(canon_norm.keys())

    def map_to_canonical(name: str, cutoff: float = 0.86):
        n = _normalize(name)
        if n in syn_to_canon:
            return syn_to_canon[n], "exact/alias"

        for ck in canon_keys:
            if len(ck) >= 6 and ck in n:
                return canon_norm[ck], "substring"

        hit = get_close_matches(n, canon_keys, n=1, cutoff=cutoff)
        if hit:
            return canon_norm[hit[0]], "fuzzy"

        return None, "no"

    return map_to_canonical


def match_potato_aromas_strict_from_notebook(
    combined_df,
    samples_filtered_path,
    out_dir="/Volumes/bmqg/default_bronze/fatemeh/aroma_flavor_project/Results",
    paths=None,
    rt_tol=0.4,
    mz_tol=0.5,
):
    """
    COMPLETE NOTEBOOK LOGIC AS A FUNCTION (module-friendly)

    Parameters
    ----------
    combined_df : pd.DataFrame
        Your combined dataframe with at least:
        - Compound Name (or Name/Compound)
        - Component RT (or RT/ComponentRT)
        - Base Peak MZ (or MZ/m/z)
        - optional FileType (will filter to FileType == 'sample' if present)
    samples_filtered_path : str
        CSV path for peaks input.
        Works with:
          - LONG: columns include Peak,tR_best,m/z + (Intensity_corrected or Intensity/Area)
                  + (Variety or Filename or Sample or SampleCol)
          - WIDE: columns Peak,tR_best,m/z + many variety columns
    out_dir : str
        Output directory (must be writable)
    paths : dict-like or None
        If provided, will use:
          - paths.get("rt_tol", rt_tol)
          - paths.get("mz_tol", mz_tol)
    rt_tol, mz_tol : float
        Default tolerances if paths not provided (or missing keys)

    Returns
    -------
    dict with:
      - matched_per_variety (pd.DataFrame)
      - aroma_matrix (pd.DataFrame)
      - ref (pd.DataFrame)
      - paths (dict of output file paths)
    """


    # ---------------------------------------------------------
    # 1) Output directory
    # ---------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2) Potato aroma list (unchanged)
    # ---------------------------------------------------------
    POTATO_AROMAS_CORE = [
        "Hexanal","Heptanal","Octanal","Nonanal","Decanal",
        "Pentanal","Propanal",
        "2-Methylbutanal","3-Methylbutanal","Butanal, 3-methyl-",
        "4-Heptenal","2-Pentenal","2-Hexenal","2-Octenal","2-Nonenal","2-Heptenal",
        "2-Trans-nonenal","2-Trans-octenal",
        "2,4-Heptadienal","2,4-Octadienal",
        "(E, E)-2,4-Nonadienal","(E, Z)-2,4-Decadienal","(E, E)-2,4-Decadienal","2,4-Nonadienal",
        "Benzaldehyde","2-methyl-Benzaldehyde","4-methyl-Benzaldehyde",
        "Benzeneacetaldehyde","Phenylacetaldehyde",

        "1-Heptanol","1-Nonanol","1-Octen-3-ol","1-Pentanol","3-Nonen-1-ol",
        "Hexanol","Phenylethyl alcohol","1-Phenylethanol",
        "2-Methyl-1-propanol","3-Methyl-1-butanol","Ethanol","Benzyl alcohol",

        "2-Heptanone","2-Nonanone","3-Heptanone","3-Octanone","3-Octen-2-one","1-Penten-3-one",
        "2,3-Butanedione","2,3-Pentanedione","Acetophenone","4-Methyl-2-pentanone",
        "(E, E)-3,5-Octadien-2-one","3,5-Octadien-2-one",

        "Acetic acid","Butanoic acid","Hexanoic acid","Isobutyric acid","Isovaleric acid",
        "Propanoic acid","Octanoic acid","2-Octenoic acid","Decanoic acid","nonanoic acid",

        "Methanethiol","Methional",
        "Dimethyl sulfide","Dimethyl disulfide","Dimethyl trisulfide","Dimethyl tetrasulfide",
        "Carbon disulfide","Hydrogen sulfide",
        "2-Methyl-3-furanthiol","3-Methylthiopropanal","3-(Methylthio)propanal",
        "2-Acetylthiazole","2-Methyl-3-thiazoline","2-Methylthiazole","Allyl methyl sulfide",

        "Ethyl acetate","Isoamyl acetate","Ethyl butanoate","Ethyl hexanoate","Methyl butanoate",
        "Ethyl 2-methylbutanoate","Ethyl propanoate","Methyl propanoate","Ethyl pentanoate",

        "2-Acetylpyrazine","2-Ethyl-3,5-dimethylpyrazine",
        "2-Isobutyl-3-methoxypyrazine","2-Isopropyl-3-methoxypyrazine",
        "2-Ethyl-3-methylpyrazine","Trimethylpyrazine","Tetramethylpyrazine",
        "2,5-Dimethylpyrazine","2,6-Dimethylpyrazine","2-Methylpyrazine",

        "Furfural","5-Methylfurfural","3-Furaldehyde",
        "methyl-Cyclopentane","n-Hexane",
        "2-n-Butyl furan","cis-2-(2-Pentenyl) furan",
        "2-pentyl-Furan","Furan, 2-pentyl-","2-Pentylfuran","2-(2-propenyl)-Furan",
        "1,2-dimethoxy-Benzene","2-methoxy-Phenol","Methyl salicylate","Mequinol",

        "Copaene","trans-beta-Ionone","Linalool","alpha-Terpineol",".alpha.-Terpineol",
        "Dimethyl phthalate",
        "Undecanal","Dodecanal",
        "(E)-2-Hexenal","(E)-2-Heptenal","(E)-2-Decenal",
        "2-Octenal, (E)-","2-Nonenal, (E)-",
        "2(3H)-Furanone, dihydro-5-pentyl-",
    ]

    POTATO_AROMAS_OPTIONAL = [
        "Phenol",
        "Benzoic acid",
        "Benzoic acid, methyl ester",

        "Eucalyptol",
        "Limonene",
        "3-Carene",

        "Tridecanal",
        "Tetradecanal",
        "2-Undecenal",
        "2-Tridecenal",

        "6-Methyl-5-hepten-2-one",
        "2-Methylpropanal",
        "1-Octyn-3-ol",
        "3-Methylfuran",
        "2-Ethylfuran",
        "2-Acetyl-5-methylfuran",

        "Acetic acid, ethenyl ester",
        "Acetic acid, methyl ester",
        "Butanoic acid, butyl ester",
        "Hexanoic acid, methyl ester",
        "Octanoic acid, methyl ester",
        "Dodecanoic acid",
        "Dodecanoic acid, 1-methylethyl ester",

        "1-Hexanol, 2-ethyl-",
        "Styrene",
        "n-Hexadecanoic acid",
        "Ethanol, 2-phenoxy-",
        "Tridecane",
        "Tetradecane",
    
        "2-Propanone, 1-methoxy-",
        "2-Decanone",
        "3-Heptanone, 4-methyl-",
        "Methyl 8-oxooctanoate",
        "Nonanoic acid, 9-oxo-, methyl ester",
    ]

    potato_aromas = list(OrderedDict.fromkeys(POTATO_AROMAS_CORE + POTATO_AROMAS_OPTIONAL))

    ALIASES = {
        "Phenylacetaldehyde": ["benzeneacetaldehyde"],
        "Phenylethyl alcohol": ["2phenylethanol", "phenethyl alcohol", "phenylethanol", "2-phenylethanol"],
        "alpha-Terpineol": [".alpha.-terpineol", "alphaterpineol", "alpha terpineol"],
        "Furan, 2-pentyl-": ["2pentylfuran", "2-pentylfuran", "2-pentyl furan", "furan, 2-pentyl"],

        "Dimethyl disulfide": ["disulfide, dimethyl"],
        "Dimethyl sulfide": ["sulfide, dimethyl"],
        "Dimethyl trisulfide": ["trisulfide, dimethyl"],

        "Butanal, 3-methyl-": ["3-methylbutanal", "butanal,3-methyl-", "3-methyl butanal"],
        "2-Octenal, (E)-": ["(e)-2-octenal", "2-octenal (e)", "2-octenal, (e)-"],
        "2-Nonenal, (E)-": ["(e)-2-nonenal", "2-nonenal (e)", "2-nonenal, (e)-"],

        "2-Methylbutanal": ["butanal, 2-methyl-", "2-methyl butanal", "2-methylbutyraldehyde"],
        "3-Methylbutanal": ["butanal, 3-methyl-", "isovaleraldehyde", "3-methyl butanal"],

        "6-Methyl-5-hepten-2-one": ["5-hepten-2-one, 6-methyl-", "sulcatone"],
        "2-Methylpropanal": ["propanal, 2-methyl-", "isobutyraldehyde"],

        "2-Undecenal": ["(e)-2-undecenal", "2-undecenal (e)", "2-undecenal (e)-", "2-undecenal, (e)-"],
        "2-Tridecenal": ["(e)-2-tridecenal", "2-tridecenal (e)", "2-tridecenal (e)-", "2-tridecenal, (e)-"],

        "3-Methylfuran": ["furan, 3-methyl-"],
        "2-Ethylfuran": ["furan, 2-ethyl-"],

        "Acetic acid, ethenyl ester": ["vinyl acetate"],
        "Acetic acid, methyl ester": ["methyl acetate"],
        "Butanoic acid, butyl ester": ["butyl butanoate"],
        "Hexanoic acid, methyl ester": ["methyl hexanoate"],
        "Octanoic acid, methyl ester": ["methyl octanoate"],
        "Dodecanoic acid, 1-methylethyl ester": ["isopropyl dodecanoate"],

        "1-Hexanol, 2-ethyl-": ["2-ethyl-1-hexanol", "2-ethylhexanol"],
        "n-Hexadecanoic acid": ["palmitic acid", "hexadecanoic acid"],
        "Ethanol, 2-phenoxy-": ["phenoxyethanol", "2-phenoxyethanol"],

        "Benzoic acid, methyl ester": ["methyl benzoate"],
        "Eucalyptol": ["1,8-cineole", "cineole"],
        "Limonene": ["d-limonene", "dl-limonene"],
        "3-Carene": ["delta-3-carene", "δ-3-carene"],

        "2-Propanone, 1-methoxy-": ["1-methoxy-2-propanone", "propylene glycol monomethyl ether"],
    }

    # ---------------------------------------------------------
    # 3) Helpers (unchanged)
    # ---------------------------------------------------------
    def strip_stereo(s: str) -> str:
        s = "" if pd.isna(s) else str(s)
        s = re.sub(r"\(\s*[EeZz]\s*,\s*[EeZz]\s*\)", "", s)
        s = re.sub(r"\(\s*[EeZz]\s*\)", "", s)
        s = re.sub(r"\b(cis|trans)\b\s*-?", "", s, flags=re.IGNORECASE)
        return s

    def normalize(s: str) -> str:
        s = strip_stereo(s).lower().strip()
        return re.sub(r"[^a-z0-9]+", "", s)

    by_norm = defaultdict(list)
    for x in potato_aromas:
        by_norm[normalize(x)].append(x)

    def pick_canonical(names):
        prefs = ["(E, E)", "(E, Z)", "(E)-", "alpha-", ".alpha.-", ", (E)-"]
        for p in prefs:
            for n in names:
                if p in n:
                    return n
        return sorted(names, key=lambda s: (len(s), s))[0]

    canon_norm = {k: pick_canonical(v) for k, v in by_norm.items()}

    syn_to_canon = {}
    for norm_key, names in by_norm.items():
        canon = canon_norm[norm_key]
        for n in names:
            syn_to_canon[normalize(n)] = canon

    for canon, alts in ALIASES.items():
        for a in alts:
            syn_to_canon[normalize(a)] = canon

    canon_keys = list(canon_norm.keys())

    def map_to_canonical(name: str, cutoff: float = 0.86):
        n = normalize(name)
        if n in syn_to_canon:
            return syn_to_canon[n], "exact/alias"

        for ck in canon_keys:
            if len(ck) >= 6 and ck in n:
                return canon_norm[ck], "substring"

        hit = get_close_matches(n, canon_keys, n=1, cutoff=cutoff)
        if hit:
            return canon_norm[hit[0]], "fuzzy"
        return None, "no"

    # ---------------------------------------------------------
    # 4) Load peaks (LONG + WIDE)  (unchanged behavior)
    # ---------------------------------------------------------
    peaks_raw = pd.read_csv(samples_filtered_path, sep=None, engine="python").copy()
    peaks_raw.columns = [str(c).strip() for c in peaks_raw.columns]

    rt_col = "tR_best"
    mz_col = "m/z"
    for c in ["Peak", rt_col, mz_col]:
        if c not in peaks_raw.columns:
            raise KeyError(f"peaks must contain '{c}'. Available: {peaks_raw.columns.tolist()}")

    long_int_candidates = ["Intensity_corrected", "Intensity", "Area", "peak_Intensity"]
    id_candidates = ["Variety", "Filename", "Sample", "SampleCol"]

    is_long = any(c in peaks_raw.columns for c in long_int_candidates) and any(
        c in peaks_raw.columns for c in id_candidates
    )

    if is_long:
        INT_COL = next(c for c in long_int_candidates if c in peaks_raw.columns)

        if "Variety" in peaks_raw.columns:
            ID_COL = "Variety"
        elif "Filename" in peaks_raw.columns:
            ID_COL = "Filename"
        elif "Sample" in peaks_raw.columns:
            ID_COL = "Sample"
        else:
            ID_COL = "SampleCol"

        peaks = peaks_raw.copy()
        peaks["Variety"] = peaks[ID_COL].astype(str).str.strip()

        if ID_COL == "Filename":
            peaks["Variety"] = peaks["Variety"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

    else:
        id_vars = ["Peak", rt_col, mz_col]
        value_vars = [c for c in peaks_raw.columns if c not in id_vars]
        if not value_vars:
            raise ValueError(
                "WIDE peaks table has no variety intensity columns. "
                "Your file might be LONG but missing Intensity_corrected."
            )

        peaks = peaks_raw.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="Variety",
            value_name="peak_Intensity",
        )
        INT_COL = "peak_Intensity"

    peaks[rt_col] = pd.to_numeric(peaks[rt_col], errors="coerce")
    peaks[mz_col] = pd.to_numeric(peaks[mz_col], errors="coerce")
    peaks[INT_COL] = pd.to_numeric(peaks[INT_COL], errors="coerce")

    peaks = peaks.dropna(subset=["Variety", rt_col, mz_col, INT_COL]).copy()
    peaks = peaks[peaks[INT_COL] > 0].copy()
    peaks["_mz_round"] = peaks[mz_col].round(0)

    # ---------------------------------------------------------
    # 5) Prepare compound reference from combined_df
    # ---------------------------------------------------------
    cdf = combined_df.copy()
    cdf.columns = [str(c).strip() for c in cdf.columns]

    compound_rt_col = next((c for c in ["Component RT", "RT", "ComponentRT"] if c in cdf.columns), None)
    compound_mz_col = next((c for c in ["Base Peak MZ", "BasePeakMZ", "MZ", "m/z"] if c in cdf.columns), None)
    name_col = next((c for c in ["Compound Name", "Compound", "Name"] if c in cdf.columns), None)

    if compound_rt_col is None or compound_mz_col is None or name_col is None:
        raise KeyError(
            "combined_df must contain Name + RT + MZ columns.\n"
            f"Detected name={name_col}, rt={compound_rt_col}, mz={compound_mz_col}\n"
            f"Available columns: {cdf.columns.tolist()}"
        )

    if "FileType" in cdf.columns:
        cdf = cdf[cdf["FileType"].eq("sample")].copy()

    cdf[name_col] = cdf[name_col].astype(str)
    cdf[compound_rt_col] = pd.to_numeric(cdf[compound_rt_col], errors="coerce")
    cdf[compound_mz_col] = pd.to_numeric(cdf[compound_mz_col], errors="coerce")
    cdf = cdf.dropna(subset=[name_col, compound_rt_col, compound_mz_col]).copy()

    mapped = cdf[name_col].map(map_to_canonical)
    cdf["Compound Name"] = [m[0] for m in mapped]
    cdf["_match_type"] = [m[1] for m in mapped]

    unmapped = cdf[cdf["Compound Name"].isna()].copy()
    unmapped_path = os.path.join(out_dir, "compound_df_unmapped_names.csv")
    if not unmapped.empty:
        unmapped[[name_col]].drop_duplicates().to_csv(unmapped_path, index=False)

    cdf = cdf[cdf["Compound Name"].notna()].copy()

    ref = (cdf.groupby("Compound Name", as_index=False)
           .agg(rt_ref=(compound_rt_col, "median"),
                mz_ref=(compound_mz_col, "median"),
                n_hits=("Compound Name", "size")))

    ref_path = os.path.join(out_dir, "/Volumes/bmqg/default_bronze/fatemeh/final_project/csv_outputs/aroma_reference_rt_mz.csv")
    ref.to_csv(ref_path, index=False)

    missing_list = [x for x in potato_aromas if normalize(x) not in set(ref["Compound Name"].map(normalize))]
    missing_ref_path = os.path.join(out_dir, "missing_aromas_from_reference.csv")
    pd.DataFrame({"Compound Name": missing_list}).to_csv(missing_ref_path, index=False)

    # ---------------------------------------------------------
    # 6) Strict matching per Variety (RT + MZ)
    # ---------------------------------------------------------
    if paths is not None:
        rt_tol_use = float(paths.get("rt_tol", rt_tol))
        mz_tol_use = float(paths.get("mz_tol", mz_tol))
    else:
        rt_tol_use = float(rt_tol)
        mz_tol_use = float(mz_tol)

    rows = []
    for variety, pv in peaks.groupby("Variety", sort=False):
        pv = pv.copy()

        for _, r in ref.iterrows():
            cname = r["Compound Name"]
            rt_ref = float(r["rt_ref"])
            mz_ref = float(r["mz_ref"])
            mz_round_ref = round(mz_ref)

            cand = pv[np.abs(pv["_mz_round"] - mz_round_ref) <= 1].copy()
            if cand.empty:
                continue

            cand["_rt_diff"] = (cand[rt_col] - rt_ref).abs()
            cand["_mz_diff"] = (cand[mz_col] - mz_ref).abs()

            strict = cand[
                (cand["_rt_diff"] <= rt_tol_use) &
                (cand["_mz_diff"] <= mz_tol_use)
            ].copy()

            if strict.empty:
                continue

            strict["_score"] = (
                strict["_rt_diff"] / max(rt_tol_use, 1e-9) +
                strict["_mz_diff"] / max(mz_tol_use, 1e-9)
            )

            best = strict.sort_values("_score").iloc[0]

            rows.append({
                "Variety": variety,
                "Compound Name": cname,
                "tR_best": float(best[rt_col]),
                "m/z": float(best[mz_col]),
                "peak_Intensity": float(best[INT_COL]),
                "_rt_diff": float(best["_rt_diff"]),
                "_mz_diff": float(best["_mz_diff"]),
            })

    matched_per_variety = pd.DataFrame(rows)
    matched_out = os.path.join(out_dir, "matched_per_variety_STRICT_ONLY.csv")
    matched_per_variety.to_csv(matched_out, index=False)

    # ---------------------------------------------------------
    # 7) Aroma intensity matrix (Variety × Compound)
    # ---------------------------------------------------------
    if matched_per_variety.empty:
        aroma_matrix = pd.DataFrame()
    else:
        aroma_matrix = (matched_per_variety
                        .pivot_table(index="Variety",
                                     columns="Compound Name",
                                     values="peak_Intensity",
                                     aggfunc="max",
                                     fill_value=0)
                        .reset_index())

    matrix_out = os.path.join(out_dir, "aroma_intensity_matrix_STRICT.csv")
    aroma_matrix.to_csv(matrix_out, index=False)

    return {
        "matched_per_variety": matched_per_variety,
        "aroma_matrix": aroma_matrix,
        "ref": ref,
        "paths": {
            "out_dir": out_dir,
            "ref_path": ref_path,
            "missing_ref_path": missing_ref_path,
            "unmapped_path": unmapped_path if os.path.exists(unmapped_path) else None,
            "matched_out": matched_out,
            "matrix_out": matrix_out,
        }}





def assign_aroma_descriptions_and_families(aroma_sample_df):
    """
    Assign aroma descriptions and chemical families to each compound in the DataFrame.

    Parameters
    ----------
    aroma_sample_df : pd.DataFrame
        Must contain a column named 'Compound Name'.

    Returns
    -------
    aroma_sample_df : pd.DataFrame
        Updated DataFrame with added 'Aroma_Description' and 'Chemical_Family' columns.
    aroma_descriptions : dict
        Mapping of compounds to sensory aroma notes.
    chemical_family_map : dict
        Mapping of compounds to chemical family classification.
    """
   

    if "Compound Name" not in aroma_sample_df.columns:
        raise KeyError("Input DataFrame must include a 'Compound Name' column.")


    # -----------------------------
    aroma_descriptions = {
        # Aldehydes
        "Propanal": "Sharp, pungent, green",
        "Butanal, 3-methyl-": "Malty, chocolate, cocoa-like (Maillard reaction product)",
        "Pentanal": "Fatty, green, pungent",
        "2-Methylbutanal": "Malty, chocolate, roasted (amino acid-derived)",
        "3-Methylbutanal": "Malty, chocolate, roasted (Leu-derived Strecker aldehyde)",
        "Hexanal": "Green, grassy, fresh-cut potato",
        "Heptanal": "Fatty, citrus-like, slightly green",
        "Octanal": "Citrus, fruity, green",
        "Nonanal": "Waxy, citrus, floral (lipid oxidation product)",
        "Decanal": "Citrus, fatty, waxy, potato-like",
        "2-Trans-nonenal": "Cardboard, fatty, old potato note",
        "2-Trans-octenal": "Green, fatty, cucumber-like",
        "4-Heptenal": "Fatty, oily, green",
        "2,4-Decadienal": "Fatty, fried, potato-chip-like",
        "Phenylacetaldehyde": "Honey-like, floral, sweet potato note",
        "Benzaldehyde": "Almond, cherry, sweet",

        # Alcohols
        "1-Heptanol": "Floral, herbal, fatty",
        "1-Nonanol": "Waxy, floral, oily",
        "1-Octen-3-ol": "Mushroom, earthy, raw potato",
        "1-Phenylethanol": "Floral, rose, sweet",
        "Hexanol": "Green, grassy, herbaceous",
        "2-Methyl-1-propanol": "Fusel, solvent-like",
        "3-Methyl-1-butanol": "Whiskey-like, malty, fusel",
        "Ethanol": "Alcoholic, sweet",

        # Ketones
        "2-Heptanone": "Fruity, blue cheese, creamy",
        "2-Nonanone": "Fatty, floral, soapy",
        "3-Heptanone": "Cheesy, earthy, fruity",
        "3-Octanone": "Mushroom, earthy, fatty",
        "2,3-Butanedione": "Buttery, creamy, cooked potato",
        "2,3-Pentanedione": "Creamy, buttery, sweet",
        "Acetophenone": "Floral, sweet, almond-like",
        "2-Propanone, 1-methoxy-": "Solvent, ether-like",
        "4-Methyl-2-pentanone": "Solvent, fruity, pungent",
        "2-Undecanone": "Fatty, waxy, green (potato tuber-like note)",

        # Sulfur compounds
        "Methanethiol": "Sulfurous, cabbage, cooked potato",
        "Methional": "Cooked potato, meaty, sulfurous (key potato note)",
        "Dimethyl disulfide": "Garlic, onion, sulfurous",
        "Dimethyl trisulfide": "Cabbage, onion, sulfurous",
        "Dimethyl tetrasulfide": "Strongly sulfurous, boiled vegetable",
        "Carbon disulfide": "Sulfurous, chemical",
        "Hydrogen sulfide": "Rotten egg, sulfurous",
        "2-Methyl-3-furanthiol": "Meaty, roasted, savory",
        "3-Methylthiopropanal": "Cooked potato, malty, onion-like",
        "2-Acetylthiazole": "Roasted, popcorn-like",
        "2-Methylthiazole": "Nutty, roasted",
        "2-Methyl-3-thiazoline": "Meaty, roasted",
        "3-(Methylthio)propanal": "Boiled potato, onion-like",
        "Allyl methyl sulfide": "Garlic, onion-like",

        # Acids
        "Acetic acid": "Vinegar-like, sour",
        "Butanoic acid": "Rancid, cheesy, sweaty",
        "Hexanoic acid": "Fatty, sweaty, rancid",
        "Isobutyric acid": "Cheesy, rancid, sour",
        "Isovaleric acid": "Sweaty, cheesy, foot-like",
        "Octanoic acid": "Fatty, soapy, rancid",
        "Propanoic acid": "Sour, pungent",

        # Esters
        "Ethyl acetate": "Fruity, solvent-like",
        "Isoamyl acetate": "Banana, fruity, sweet",
        "Ethyl butanoate": "Pineapple, fruity, sweet",
        "Ethyl hexanoate": "Apple, pineapple, sweet",
        "Methyl butanoate": "Apple, fruity, sweet",
        "Ethyl 2-methylbutanoate": "Apple, sweet, fruity",
        "Ethyl propanoate": "Fruity, rum-like",
        "Methyl propanoate": "Fruity, sweet",
        "Ethyl pentanoate": "Fruity, sweet",

        # Pyrazines
        "2-Acetylpyrazine": "Nutty, roasted, popcorn-like",
        "2-Ethyl-3,5-dimethylpyrazine": "Nutty, earthy, roasted",
        "2-Isobutyl-3-methoxypyrazine": "Green bell pepper, earthy",
        "2-Isopropyl-3-methoxypyrazine": "Green, earthy, potato peel-like",
        "2-Ethyl-3-methylpyrazine": "Nutty, roasted, earthy",
        "Trimethylpyrazine": "Roasted, cocoa, nutty",
        "Tetramethylpyrazine": "Roasted, nutty, cocoa",
        "2,5-Dimethylpyrazine": "Roasted, nutty",
        "2,6-Dimethylpyrazine": "Roasted, earthy",
        "2-Methylpyrazine": "Roasted, nutty, chocolate-like",

        # Aromatics/Furans
        "Benzeneacetaldehyde": "Honey, floral, sweet",
        "Benzyl alcohol": "Floral, sweet, mild",
        "Phenylethyl alcohol": "Rose-like, floral, sweet",
        "Furfural": "Sweet, almond, caramel",
        "5-Methylfurfural": "Caramel, baked, sweet"
    }

    # -----------------------------
    # 2) Normalization helpers (handles (E)-, cis/trans, punctuation)
    # -----------------------------
    def _strip_stereo(x: str) -> str:
        x = "" if pd.isna(x) else str(x)
        x = re.sub(r"\(\s*[EeZz]\s*,\s*[EeZz]\s*\)", "", x)
        x = re.sub(r"\(\s*[EeZz]\s*\)", "", x)
        x = re.sub(r"\b(cis|trans)\b\s*-?", "", x, flags=re.IGNORECASE)
        return x

    def _norm(x: str) -> str:
        x = _strip_stereo(x).lower().strip()
        return re.sub(r"[^a-z0-9]+", "", x)

    # normalized lookup for descriptions
    aroma_desc_norm = {_norm(k): v for k, v in aroma_descriptions.items()}

    # -----------------------------
    # 3) Chemical family rules (covers ALL your potato aroma list)
    # -----------------------------
    # Order matters: first match wins.
    FAMILY_RULES = [
        # Sulfur family (strong signals)
        ("Sulfur Compound", [
            "methanethiol", "methional", "dimethylsulfide", "dimethyldisulfide",
            "dimethyltrisulfide", "dimethyltetrasulfide", "carbondisulfide",
            "hydrogensulfide", "furanthiol", "methylthio", "thiazole", "thiazoline",
            "allylmethylsulfide"
        ]),

        # Acids
        ("Acid", [
            "acid"
        ]),

        # Esters (common suffixes)
        ("Ester", [
            "acetate", "butanoate", "propanoate", "pentanoate", "hexanoate", "benzoate",
            "salicylate", "ester"
        ]),

        # Aldehydes (suffixes + known tokens)
        ("Aldehyde", [
            "al", "aldehyde", "furaldehyde", "hexenal", "heptenal", "octenal", "nonenal", "decenal",
            "undecenal", "tridecenal", "pentanal", "hexanal", "heptanal", "octanal", "nonanal", "decanal",
            "undecanal", "dodecanal", "tridecanal", "tetradecanal"
        ]),

        # Alcohols
        ("Alcohol", [
            "ol", "alcohol", "ethanol", "hexanol", "heptanol", "nonanol", "phenylethanol", "benzylalcohol"
        ]),

        # Ketones
        ("Ketone", [
            "one", "butanedione", "pentanedione", "acetophenone", "propanone"
        ]),

        # Pyrazines
        ("Pyrazine", [
            "pyrazine", "methoxypyrazine"
        ]),

        # Furans / aromatics / phenolics / terpenes / hydrocarbons
        ("Aromatic/Furan", [
            "furfural", "furan", "benzene", "phenol", "styrene", "methoxyphenol", "dimethoxybenzene",
            "linalool", "terpineol", "eucalyptol", "limonene", "carene", "ionone",
            "copaene", "hexane", "cyclopentane", "tridecane", "tetradecane", "hexadecane",
            "phthalate"
        ]),
    ]

    def classify_family(compound_name: str) -> str:
        n = _norm(compound_name)

        # Special-case: some "al" endings could be alcohols; keep it conservative.
        # Use explicit known aldehyde tokens above; otherwise fall through.
        for fam, keys in FAMILY_RULES:
            for k in keys:
                if k in n:
                    # A tiny guard: "ol" appears in many; don't misclassify aldehydes as alcohol
                    if fam == "Alcohol" and ("al" in n and n.endswith("al")):
                        continue
                    return fam
        return "Other"

    # -----------------------------
    # 4) Default description per family (so Unknown is rare)
    # -----------------------------
    DEFAULT_DESC_BY_FAMILY = {
        "Aldehyde": "Green/fatty notes; lipid-oxidation related",
        "Alcohol": "Green/floral notes; fresh/vegetal nuances",
        "Ketone": "Buttery/creamy/mushroom-like notes",
        "Sulfur Compound": "Sulfurous cooked-potato / onion-cabbage notes",
        "Acid": "Sour/rancid/cheesy notes",
        "Ester": "Fruity/sweet notes",
        "Pyrazine": "Roasted/nutty/earthy notes",
        "Aromatic/Furan": "Sweet/toasty/spicy/floral notes (aromatic/furan/terpene/hydrocarbon)",
        "Other": "General aroma compound (no curated descriptor yet)",
    }

    # -----------------------------
    # 5) Apply mappings
    # -----------------------------
    comp_series = aroma_sample_df["Compound Name"].astype(str)

    aroma_sample_df["Chemical_Family"] = comp_series.map(classify_family)

    def get_description(name: str) -> str:
        n = _norm(name)
        if n in aroma_desc_norm:
            return aroma_desc_norm[n]
        fam = classify_family(name)
        return DEFAULT_DESC_BY_FAMILY.get(fam, "General aroma compound")

    aroma_sample_df["Aroma_Description"] = comp_series.map(get_description)

    # also return explicit maps for your export / reproducibility
    chemical_family_map = {c: classify_family(c) for c in comp_series.unique()}
    # build a full description map (curated + defaults)
    full_desc_map = {c: get_description(c) for c in comp_series.unique()}

    return aroma_sample_df, full_desc_map, chemical_family_map



def plot_aroma_family_distribution(aroma_df):
    """
    Plot the distribution of identified aroma compound families.

    Parameters
    ----------
    aroma_df : pd.DataFrame
        Must contain 'Compound Name' and 'Chemical_Family' columns.

    Returns
    -------
    family_counts : pd.Series
        Number of unique compounds per chemical family.
    """
    if "Compound Name" not in aroma_df.columns or "Chemical_Family" not in aroma_df.columns:
        raise KeyError("DataFrame must include 'Compound Name' and 'Chemical_Family' columns.")

    # Count unique compounds per family
    family_counts = (
        aroma_df[["Compound Name", "Chemical_Family"]]
        .drop_duplicates()
        .groupby("Chemical_Family")
        .size()
        .sort_values(ascending=False)
    )

    # --- Color palette for chemical families ---
    family_palette = {
        "Aldehyde": "#FDD835",          # yellow
        "Alcohol": "#81C784",           # green
        "Ketone": "#64B5F6",            # blue
        "Ester": "#FFB74D",             # orange
        "Pyrazine": "#A1887F",          # brown
        "Sulfur Compound": "#E57373",   # red
        "Acid": "#4DB6AC",              # teal
        "Aromatic/Furan": "#BA68C8",    # purple
        "Other": "#B0BEC5",             # gray
        "Unknown": "#E0E0E0"            # light gray
    }

    # Apply color for families present
    colors = [family_palette.get(fam, "#B0BEC5") for fam in family_counts.index]

    # --- Plot ---
    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=family_counts.values,
        y=family_counts.index,
        palette=colors
    )
    plt.title("Distribution of Identified Aroma Compound Families", fontsize=14, weight="bold")
    plt.xlabel("Number of Unique Compounds", fontsize=12)
    plt.ylabel("Chemical Family", fontsize=12)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return family_counts


def classify_and_plot_families(aroma_sample_df, aroma_descriptions):
    """
    Classify aroma compounds into chemical families and plot their distribution.
    
    Parameters
    ----------
    aroma_sample_df : pd.DataFrame
        DataFrame containing at least a 'Compound Name' column.
    aroma_descriptions : dict
        Dictionary of compound descriptions (used to ensure consistent mapping).
    
    Returns
    -------
    pd.DataFrame
        Updated DataFrame with a new column 'Chemical_Family'.
    """
    chemical_family_map = {}

    for compound in aroma_descriptions.keys():
        if compound in [
            "Propanal", "Butanal, 3-methyl-", "Pentanal", "2-Methylbutanal", "3-Methylbutanal",
            "Hexanal", "Heptanal", "Octanal", "Nonanal", "Decanal", "2-Trans-nonenal",
            "2-Trans-octenal", "4-Heptenal", "2,4-Decadienal", "Phenylacetaldehyde", "Benzaldehyde"
        ]:
            family = "Aldehyde"
        elif compound in [
            "1-Heptanol", "1-Nonanol", "1-Octen-3-ol", "1-Phenylethanol", "Hexanol",
            "2-Methyl-1-propanol", "3-Methyl-1-butanol", "Ethanol"
        ]:
            family = "Alcohol"
        elif compound in [
            "2-Heptanone", "2-Nonanone", "3-Heptanone", "3-Octanone", "2,3-Butanedione",
            "2,3-Pentanedione", "Acetophenone", "2-Propanone, 1-methoxy-", "4-Methyl-2-pentanone",
            "2-Undecanone"
        ]:
            family = "Ketone"
        elif compound in [
            "Methanethiol", "Methional", "Dimethyl disulfide", "Dimethyl trisulfide",
            "Dimethyl tetrasulfide", "Carbon disulfide", "Hydrogen sulfide",
            "2-Methyl-3-furanthiol", "3-Methylthiopropanal", "2-Acetylthiazole",
            "2-Methylthiazole", "2-Methyl-3-thiazoline", "3-(Methylthio)propanal", "Allyl methyl sulfide"
        ]:
            family = "Sulfur Compound"
        elif compound in [
            "Acetic acid", "Butanoic acid", "Hexanoic acid", "Isobutyric acid",
            "Isovaleric acid", "Octanoic acid", "Propanoic acid"
        ]:
            family = "Acid"
        elif compound in [
            "Ethyl acetate", "Isoamyl acetate", "Ethyl butanoate", "Ethyl hexanoate",
            "Methyl butanoate", "Ethyl 2-methylbutanoate", "Ethyl propanoate",
            "Methyl propanoate", "Ethyl pentanoate"
        ]:
            family = "Ester"
        elif compound in [
            "2-Acetylpyrazine", "2-Ethyl-3,5-dimethylpyrazine", "2-Isobutyl-3-methoxypyrazine",
            "2-Isopropyl-3-methoxypyrazine", "2-Ethyl-3-methylpyrazine", "Trimethylpyrazine",
            "Tetramethylpyrazine", "2,5-Dimethylpyrazine", "2,6-Dimethylpyrazine", "2-Methylpyrazine"
        ]:
            family = "Pyrazine"
        elif compound in [
            "Benzeneacetaldehyde", "Benzyl alcohol", "Phenylethyl alcohol", "Furfural", "5-Methylfurfural"
        ]:
            family = "Aromatic/Furan"
        else:
            family = "Other"

        chemical_family_map[compound] = family

    # Map family to DataFrame
    aroma_sample_df["Chemical_Family"] = aroma_sample_df["Compound Name"].map(chemical_family_map)

    # Count unique families
    family_counts = (
        aroma_sample_df[["Compound Name", "Chemical_Family"]]
        .drop_duplicates()["Chemical_Family"]
        .value_counts()
    )

    # --- Visualization ---
    plt.figure(figsize=(8, 5))
    family_counts.plot(kind="bar", color="#CBB680")
    plt.title("Distribution of Identified Aroma Compound Families")
    plt.ylabel("Number of Unique Compounds")
    plt.xlabel("Chemical Family")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nChemical Family Counts:\n", family_counts)
    return aroma_sample_df




def plot_aroma_family_distribution(aroma_df):
    """
    Plot the distribution of identified aroma compound families.

    Parameters
    ----------
    aroma_df : pd.DataFrame
        Must contain 'Compound Name' and 'Chemical_Family' columns.

    Returns
    -------
    family_counts : pd.Series
        Number of unique compounds per chemical family.
    """
    if "Compound Name" not in aroma_df.columns or "Chemical_Family" not in aroma_df.columns:
        raise KeyError("DataFrame must include 'Compound Name' and 'Chemical_Family' columns.")

    # Count unique compounds per family
    family_counts = (
        aroma_df[["Compound Name", "Chemical_Family"]]
        .drop_duplicates()
        .groupby("Chemical_Family")
        .size()
        .sort_values(ascending=False)
    )

    # --- Color palette for chemical families ---
    family_palette = {
        "Aldehyde": "#FDD835",          # yellow
        "Alcohol": "#81C784",           # green
        "Ketone": "#64B5F6",            # blue
        "Ester": "#FFB74D",             # orange
        "Pyrazine": "#A1887F",          # brown
        "Sulfur Compound": "#E57373",   # red
        "Acid": "#4DB6AC",              # teal
        "Aromatic/Furan": "#BA68C8",    # purple
        "Other": "#B0BEC5",             # gray
        "Unknown": "#E0E0E0"            # light gray
    }

    # Apply color for families present
    colors = [family_palette.get(fam, "#B0BEC5") for fam in family_counts.index]

    # --- Plot ---
    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=family_counts.values,
        y=family_counts.index,
        palette=colors
    )
    plt.title("Distribution of Identified Aroma Compound Families", fontsize=14, weight="bold")
    plt.xlabel("Number of Unique Compounds", fontsize=12)
    plt.ylabel("Chemical Family", fontsize=12)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return family_counts




def summarize_and_visualize_aroma_compounds_v2(
    aroma_sample_df,
    aroma_json_path="/Volumes/bmqg/default_bronze/fatemeh/final_project/csv_outputs/aroma_descriptions.json",
    family_json_path="/Volumes/bmqg/default_bronze/fatemeh/final_project/csv_outputs/chemical_family_map.json",
    output_path=None,
    abundance_col=None,      # "peak_Intensity_corrected" 
    within_variety_agg="median",  # "median" "max"
):
    df = aroma_sample_df.copy()

    # -------- detect columns ----------
    name_col = next((c for c in ["Compound Name","Compound","Name","Best Hit","Hit Name","Library Name"] if c in df.columns), None)
    if name_col is None:
        raise KeyError("No compound name column found (e.g. 'Compound Name').")

    rt_col = next((c for c in ["Component RT","tR_best","tR","RT","rt","RetentionTime","retention_time"] if c in df.columns), None)
    if rt_col is None:
        raise KeyError("No RT column found (e.g. 'tR_best' or 'Component RT').")

    mz_col = next((c for c in ["m/z","mz","MZ","Base Peak MZ","Base Peak m/z"] if c in df.columns), None)
    if mz_col is None:
        raise KeyError("No m/z column found (e.g. 'm/z').")

    sample_col = next((c for c in ["Variety","Sample","FileName"] if c in df.columns), None)
    if sample_col is None:
        raise KeyError("No sample identifier found. Need one of: Variety / Sample / FileName")

    # -------- load mappings ----------
    with open(aroma_json_path, "r", encoding="utf-8") as f:
        AROMA_DESCRIPTIONS = json.load(f)
    with open(family_json_path, "r", encoding="utf-8") as f:
        CHEMICAL_FAMILY_MAP = json.load(f)

    # -------- pick abundance column ----------
    if abundance_col is None:
        pref = [
            "peak_Intensity_corrected","peak_intensity_corrected",
            "Intensity_corrected","Intensity","Area","PeakHeight","Height",
            "peak_Intensity","peak_intensity"
        ]
        abundance_col = next((c for c in pref if c in df.columns), None)

    if abundance_col is None:
        # fallback: any col containing keywords
        cands = [c for c in df.columns if any(k in c.lower() for k in ["intensity","area","height","abundance"])]
        abundance_col = cands[0] if cands else None

    if abundance_col is None:
        raise KeyError(f"No abundance column found. Columns are:\n{df.columns.tolist()}")

    print(f"Using sample='{sample_col}', name='{name_col}', RT='{rt_col}', m/z='{mz_col}', abundance='{abundance_col}'")

    df[abundance_col] = pd.to_numeric(df[abundance_col], errors="coerce")

    if within_variety_agg == "max":
        per_variety = (df.groupby([sample_col, name_col], as_index=False)
                         .agg(Abundance=(abundance_col, "max")))
    else:
        per_variety = (df.groupby([sample_col, name_col], as_index=False)
                         .agg(Abundance=(abundance_col, "median")))

    # -------- 2) add metadata ----------
    per_variety["Chemical_Family"] = per_variety[name_col].map(CHEMICAL_FAMILY_MAP).fillna("Unknown")
    per_variety["Aroma_Description"] = per_variety[name_col].map(AROMA_DESCRIPTIONS).fillna("Unknown")

    precursor_map = {
        "Nonanal": "Fatty acids",
        "3-Heptanone": "Fatty acids",
        "2-Nonanone": "Fatty acids",
        "Butanal, 3-methyl-": "BCAA (Leucine)",
        "Isoamyl acetate": "BCAA (Leu/Ile)",
        "Hexanal": "Fatty acids",
        "Octanal": "Fatty acids",
        "Pentanal": "Fatty acids",
        "Acetic acid": "Sugar/Fatty acids",
        "Propanal": "Fatty acids",
        "Tetramethylpyrazine": "Sugar + Amino acids",
        "Ethanol": "Carbohydrate fermentation",
        "Decanal": "Fatty acids",
        "Hexanoic acid": "Fatty acids",
        "Octanoic acid": "Fatty acids",
        "3-Octanone": "Fatty acids",
        "Methyl butanoate": "Fatty acids / Esterification",
        "Phenylacetaldehyde": "Phenylalanine"
    }
    per_variety["Precursor/Pathway"] = per_variety[name_col].map(precursor_map).fillna("Unknown")

    # -------- 3) stats across varieties ----------
    summary_df = (per_variety
        .groupby([name_col, "Chemical_Family", "Precursor/Pathway", "Aroma_Description"], dropna=False)
        .agg(
            N_Varieties=(sample_col, "nunique"),
            Median_Abundance=("Abundance", "median"),
            Mean_Abundance=("Abundance", "mean"),
            Std_Abundance=("Abundance", "std"),
        )
        .reset_index()
    )

    summary_df["Std_Abundance"] = summary_df["Std_Abundance"].fillna(0)
    summary_df["Coefficient of Variation (%)"] = (
        (summary_df["Std_Abundance"] / summary_df["Mean_Abundance"].replace(0, np.nan)) * 100
    ).replace([np.inf, -np.inf], np.nan).round(1)

    # rename for final table
    summary_df = summary_df.rename(columns={
        name_col: "Compound",
        "Chemical_Family": "Functional Group",
        "Precursor/Pathway": "Precursor/Pathway",
        "Aroma_Description": "Aroma Description",
        "Median_Abundance": "Median (relative intensity)",
        "Mean_Abundance": "Mean",
        "Std_Abundance": "Std"
    }).sort_values("Median (relative intensity)", ascending=False)

    if output_path:
        summary_df.to_excel(output_path, index=False)
        print(f" Summary saved to: {output_path}")

    to_show = summary_df.head(top_n_plot).copy()

    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(to_show))))
    ax.axis("off")

    header_color = "#D35400"
    row_colors = ["#FDF2E9", "#FBEEE6"]

    show_cols = ["Compound", "Functional Group", "Precursor/Pathway", "Aroma Description", "Coefficient of Variation (%)", "N_Varieties"]
    table = ax.table(
        cellText=to_show[show_cols].values,
        colLabels=["Compound","Functional Group","Precursor/Pathway","Aroma Description","CV (%)","N varieties"],
        cellLoc="center",
        loc="center",
        colColours=[header_color]*len(show_cols)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    for i in range(len(to_show)):
        color = row_colors[i % 2]
        for j in range(len(show_cols)):
            table[(i+1, j)].set_facecolor(color)

    for _, cell in table.get_celld().items():
        cell.set_edgecolor("white")

    plt.tight_layout()
    plt.show()

    return summary_df



def plot_outlier_vs_normal_aromas(aroma_matrix, outliers, chemical_family_map=None, output_dir="aroma_cluster_results"):
  
    os.makedirs(output_dir, exist_ok=True)

    df = aroma_matrix.copy()


    if (("Variety" not in df.columns) and (df.index.name in ["Variety", "variety"])) or (df.index.dtype == object):
        df = df.reset_index()


    df.columns = [str(c).strip() for c in df.columns]


    variety_col = None
    for c in df.columns:
        if str(c).strip().lower() == "variety":
            variety_col = c
            break
    if variety_col is None:
        variety_col = df.columns[0]   # fallback

    df = df.rename(columns={variety_col: "Variety"})
    df["Variety"] = df["Variety"].astype(str).fillna("").str.strip()
    df["Variety_up"] = df["Variety"].str.upper()

    out_set = {str(o).upper().strip() for o in outliers}


    df = df[df["Variety_up"].ne("") & df["Variety_up"].ne("NAN")].copy()

    df["group"] = np.where(df["Variety_up"].isin(out_set), "Outlier", "Normal")

    features = df.drop(columns=["Variety", "Variety_up", "group"], errors="ignore").select_dtypes(include="number")
    if features.shape[1] == 0:
        raise ValueError("No numeric aroma columns found. Provide an aroma MATRIX (Variety + numeric compounds).")

    mean_profiles = df.groupby("group")[features.columns].mean().T

    if "Outlier" not in mean_profiles.columns or "Normal" not in mean_profiles.columns:
        raise ValueError(
            f"Groups present: {list(mean_profiles.columns)}. "
            f"Check outlier names vs your Variety values."
        )

    mean_profiles["Difference"] = mean_profiles["Outlier"] - mean_profiles["Normal"]
    mean_profiles = mean_profiles.reset_index().rename(columns={"index": "Compound Name"})
    mean_profiles["compound_norm"] = mean_profiles["Compound Name"].astype(str).str.strip().str.lower()

 
    if isinstance(chemical_family_map, dict):
        chem_map_norm = {str(k).strip().lower(): v for k, v in chemical_family_map.items()}
        mean_profiles["Chemical Family"] = mean_profiles["compound_norm"].map(chem_map_norm).fillna("Other")
    else:
        mean_profiles["Chemical Family"] = "Other"


    top_up = mean_profiles.nlargest(5, "Difference")
    top_down = mean_profiles.nsmallest(5, "Difference")
    top_diff = pd.concat([top_up, top_down]).sort_values("Difference", ascending=True)


    family_colors = {
        "Alcohol": "#81C784",
        "Aldehyde": "#FBC02D",
        "Ketone": "#4DB6AC",
        "Ester": "#BA68C8",
        "Acid": "#E57373",
        "Furan": "#90A4AE",
        "Aromatic": "#F48FB1",
        "Aromatic/Furan": "#F48FB1",
        "Sulfur": "#8D6E63",
        "Sulfur Compound": "#8D6E63",
        "Pyrazine": "#6D4C41",
        "Other": "#C0C0C0"
    }
    for fam in top_diff["Chemical Family"].unique():
        if fam not in family_colors:
            family_colors[fam] = "#BDBDBD"

    # Plot
    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    sns.barplot(
        data=top_diff,
        y="Compound Name",
        x="Difference",
        hue="Chemical Family",
        palette=family_colors,
        dodge=False,
        edgecolor="none"
    )
    plt.axvline(0, color="gray", linestyle="--", lw=1)
    plt.title("Key Aroma Compounds by Chemical Family (Outliers vs Normal)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Mean Intensity Difference (Outliers – Normal)", fontsize=12)
    plt.ylabel("Compound Name", fontsize=12)
    plt.legend(title="Chemical Family", frameon=True, fontsize=10, title_fontsize=11)
    plt.tight_layout()

    out_png = os.path.join(output_dir, "outlier_vs_normal_aromas.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", out_png)
    return top_diff, mean_profiles




def create_aroma_intensity_matrix(aroma_sample_df):
    """
    Creates an aroma compound intensity matrix (samples × compounds).

    Parameters
    ----------
    aroma_sample_df : pd.DataFrame
        DataFrame containing aroma compound data with columns:
        ['FileName', 'Variety', 'Compound Name', 'Intensity'].

    Returns
    -------
    aroma_matrix : pd.DataFrame
        Pivoted DataFrame where each row represents a variety/sample
        and each column represents an aroma compound's mean intensity.
    """

    # --- Pivot to create matrix ---
    aroma_pivot = aroma_sample_df.pivot_table(
        index=["FileName", "Variety"],
        columns="Compound Name",
        values="Intensity",
        aggfunc="mean"
    ).reset_index()

    # --- Clean up column names ---
    aroma_pivot.columns.name = None

    # --- Drop filename column (optional, if you only want variety + compounds) ---
    aroma_matrix = aroma_pivot.drop(columns=["FileName"], errors="ignore")

    # --- Summary ---
    print(f"Aroma matrix created with shape: {aroma_matrix.shape}")
    print("\nPreview:")
    print(aroma_matrix.head())

    return aroma_matrix





def pca_kmeans_aroma(aroma_matrix, output_dir="aroma_pca_kmeans"):
    """
    PCA + K-Means clustering + Outlier detection + Confidence Ellipses
    Handles both 'Variety' and 'variety' column names automatically.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare data ---
    df = aroma_matrix.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "variety" in df.columns:
        varieties = df["variety"].astype(str).values
    else:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "variety"}, inplace=True)
        varieties = df["variety"].astype(str).values

    X = df.drop(columns=["variety"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # --- PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance (2 PCs): {sum(pca.explained_variance_ratio_)*100:.2f}%")

    # --- Outlier detection (Mahalanobis) ---
    mean_vec = np.mean(X_pca, axis=0)
    cov_matrix = np.cov(X_pca, rowvar=False)
    inv_cov = np.linalg.inv(cov_matrix)
    mahal = [distance.mahalanobis(x, mean_vec, inv_cov) for x in X_pca]

    Q1, Q3 = np.percentile(mahal, [25, 75])
    IQR = Q3 - Q1
    threshold = Q3 + 2 * IQR
    outlier_mask = np.array(mahal) > threshold

    # --- Determine best k for K-Means ---
    sil_scores = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, labels))

    best_k = K_range[np.argmax(sil_scores)]
    print(f"Optimal number of clusters (based on silhouette): {best_k}")

    # --- Final K-Means ---
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    centers = kmeans.cluster_centers_

    # --- Create results table ---
    results = pd.DataFrame({
        "Variety": varieties,
        "Cluster": cluster_labels,
        "Mahalanobis": mahal,
        "Outlier": outlier_mask
    })
    results.to_csv(os.path.join(output_dir, "/Volumes/bmqg/default_bronze/fatemeh/final_project/csv_outputs/aroma_pca_kmeans_results.csv"), index=False)

    # --- Helper: draw confidence ellipse ---
        # --- Helper: draw confidence ellipse ---
    def draw_ellipse(position, covariance, ax=None, color='gray'):
        if ax is None:
            ax = plt.gca()
        if covariance.shape == (2, 2):
            eigvals, eigvecs = np.linalg.eigh(covariance)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)
        else:
            width, height, angle = 0, 0, 0
        for nsig in range(1, 4):  # 1σ, 2σ, 3σ
            ellipse = Ellipse(position, nsig * width, nsig * height,
                              angle=angle, facecolor=color, edgecolor='none', alpha=0.15)
            ax.add_patch(ellipse)


    # --- Plot PCA with K-Means clusters, centroids, and ellipses ---
    plt.figure(figsize=(10, 7))
    plt.grid(False)
    colors = plt.cm.Set2(np.linspace(0, 1, best_k))

    for i, color in enumerate(colors):
        cluster_points = X_pca[cluster_labels == i]
        cov = np.cov(cluster_points, rowvar=False)
        draw_ellipse(centers[i], cov, color=color)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=color, s=70, alpha=0.8, edgecolor="k", label=f"Cluster {i+1}")

    # --- Plot centroids ---
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=180,
                marker="X", edgecolor="white", linewidth=1.5, label="Centroids")

    # --- Highlight outliers ---
    plt.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
                color="red", edgecolor="black", s=130, label="Outliers", zorder=5)
    for i, name in enumerate(varieties):
        if outlier_mask[i]:
            plt.text(X_pca[i, 0]+0.3, X_pca[i, 1], name,
                     fontsize=8, color='darkred', weight='bold')

    # --- Aesthetics ---
    plt.axhline(0, color='gray', lw=0.8, alpha=0.6)
    plt.axvline(0, color='gray', lw=0.8, alpha=0.6)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12, weight='bold')
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12, weight='bold')
    plt.title("PCA + K-Means Clustering of Aroma Varieties\n(Centroids, Ellipses, and Outliers Highlighted)",
              fontsize=14, weight='bold', pad=15)
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, "/Volumes/bmqg/default_bronze/fatemeh/final_project/csv_outputs/pca_kmeans_clusters_ellipses.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nResults saved in: {output_dir}")
    return results, X_pca, cluster_labels




import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse


def analyze_aroma_clusters_gmm(aroma_matrix, output_dir="aroma_cluster_results"):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.manifold import TSNE
    from matplotlib.patches import Ellipse

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # --- Prepare data
    # =========================================================
    df = aroma_matrix.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "variety" in df.columns:
        varieties = df["variety"].astype(str).values
    else:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "variety"}, inplace=True)
        varieties = df["variety"].astype(str).values

    # =========================================================
    # --- CLEAN FEATURE MATRIX (🔥 مهم‌ترین قسمت)
    # =========================================================
    X = (
        df.drop(columns=["variety", "dataset", "cluster"], errors="ignore")
        .select_dtypes(include=[np.number])
    )

    # fix NaN / inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    # log transform (CRITICAL)
    X = np.log1p(X)

    # remove low variance
    X = X.loc[:, X.std() > 1e-6]

    # =========================================================
    # --- Scaling
    # =========================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================================================
    # --- PCA
    # =========================================================
    pca_full = PCA().fit(X_scaled)
    explained = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = np.argmax(explained >= 0.9) + 1

    print(f"PCA components: {n_comp} ({explained[n_comp-1]*100:.1f}% variance)")

    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    # =========================================================
    # --- GMM model selection
    # =========================================================
    k_range = range(2, 11)
    bics = []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='tied', random_state=42)
        gmm.fit(X_pca)
        bics.append(gmm.bic(X_pca))

    best_k = k_range[np.argmin(bics)]
    print(f"Best k (BIC): {best_k}")

    # =========================================================
    # --- Fit GMM
    # =========================================================
    gmm = GaussianMixture(n_components=best_k, covariance_type='tied', random_state=42)
    cluster_labels = gmm.fit_predict(X_pca)

    # =========================================================
    # --- Metrics
    # =========================================================
    print("\nClustering metrics:")
    print(f" Silhouette: {silhouette_score(X_pca, cluster_labels):.3f}")
    print(f" DB Index:   {davies_bouldin_score(X_pca, cluster_labels):.3f}")
    print(f" CH Score:   {calinski_harabasz_score(X_pca, cluster_labels):.1f}")

    # =========================================================
    # --- Outliers
    # =========================================================
    log_likelihood = gmm.score_samples(X_pca)
    threshold = np.percentile(log_likelihood, 5)
    outlier_mask = log_likelihood < threshold

    # =========================================================
    # --- Explain outliers
    # =========================================================
    X_df = pd.DataFrame(X_scaled, columns=X.columns, index=varieties)

    explanations = {}

    for i, is_out in enumerate(outlier_mask):
        if not is_out:
            continue

        variety = varieties[i]
        cluster = cluster_labels[i]

        cluster_mean = X_df.iloc[cluster_labels == cluster].mean()
        sample = X_df.iloc[i]

        diff = sample - cluster_mean
        z_scores = diff / (X_df.std() + 1e-6)

        top = diff.abs().sort_values(ascending=False).head(5)

        explanations[variety] = {
            "cluster": int(cluster),
            "top_features": list(top.index),
            "raw_diff": list(diff[top.index].values),
            "z_scores": list(z_scores[top.index].values)
        }

    explain_df = pd.DataFrame.from_dict(explanations, orient="index")
    explain_df.to_csv(os.path.join(output_dir, "outlier_explanations.csv"))

    print(f"\nOutliers ({sum(outlier_mask)}): {list(explain_df.index)}")

    # =========================================================
    # --- cluster_means (🔥 درست و مهم)
    # =========================================================
    cluster_means = pd.DataFrame(X, columns=X.columns)
    cluster_means["Cluster"] = cluster_labels
    cluster_means = cluster_means.groupby("Cluster").mean()

    # =========================================================
    # --- results table
    # =========================================================
    results = pd.DataFrame({
        "Variety": varieties,
        "Cluster": cluster_labels,
        "LogLikelihood": log_likelihood,
        "is_outlier": outlier_mask
    })

    print(f"\nAll results saved in: {output_dir}")

    return results, cluster_means, explain_df


def summarize_aroma_intensities(aroma_matrix, top_n=10):
    """
    Summarizes and visualizes aroma compound intensities across potato varieties.

    Parameters
    ----------
    aroma_matrix : pd.DataFrame
        DataFrame containing aroma compound intensity data (with 'Variety' column).
    top_n : int, optional
        Number of top compounds to display based on average intensity (default = 10).

    Returns
    -------
    desc_stats : pd.DataFrame
        DataFrame containing descriptive statistics (mean, std, min, max) for each compound.
    top_compounds : pd.DataFrame
        Top N aroma compounds ranked by mean intensity.
    """
    # ---  Keep only numeric columns (exclude 'Variety' etc.) ---
    numeric_data = aroma_matrix.select_dtypes(include=['number'])

    # ---  Compute descriptive statistics ---
    desc_stats = numeric_data.describe().T  # Transpose for compounds as rows
    # --- 3 Identify top compounds by mean intensity ---
    top_compounds = desc_stats.sort_values(by='mean', ascending=False).head(top_n)
    top_compounds_plot = top_compounds[['mean', 'std']]
    # ---  Print summary ---
    print(f" Total compounds analyzed: {numeric_data.shape[1]}")
    print(f" Top {top_n} aroma compounds by average intensity:\n")
    print(top_compounds_plot.round(2))
    # ---  Visualization: bar plot with error bars ---
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_compounds_plot.index,
        top_compounds_plot['mean'],
        xerr=top_compounds_plot['std'],
        color='skyblue',
        edgecolor='black'
    )
    plt.xlabel("Mean Intensity (±1 SD)")
    plt.title(f"Top {top_n} Aroma Compounds in Potato Varieties by Mean Intensity")
    plt.gca().invert_yaxis()  # highest at top
    plt.tight_layout()
    plt.show()

    return desc_stats, top_compounds




def plot_top_aroma_heatmap(aroma_matrix, top_n=5, mode="per_variety",
                           id_col_candidates=("Variety", "variety", "Sample", "sample")):
    """
    Robust heatmap for top aroma compounds.
    Accepts Variety either as a column OR index, and also accepts lowercased 'variety'.
    """
 

    aroma_df = aroma_matrix.copy()

    # --- 1) Ensure we have a proper ID column (Variety) ---
    # If index looks like varieties, move it to a column
    if not isinstance(aroma_df.index, pd.RangeIndex):
        if aroma_df.index.name is not None:
            aroma_df = aroma_df.reset_index()
        else:
            # unnamed index -> reset and name it
            aroma_df = aroma_df.reset_index().rename(columns={"index": "Variety"})

    # Find existing id column
    id_col = next((c for c in id_col_candidates if c in aroma_df.columns), None)

    # If still not found, assume first column is the id
    if id_col is None:
        id_col = aroma_df.columns[0]

    # Normalize to 'Variety'
    if id_col != "Variety":
        aroma_df = aroma_df.rename(columns={id_col: "Variety"})

    # Make Variety clean strings
    aroma_df["Variety"] = aroma_df["Variety"].astype(str).str.strip()
    aroma_df = aroma_df[aroma_df["Variety"].ne("") & aroma_df["Variety"].ne("nan")].copy()

    # --- 2) Keep numeric aroma columns only ---
    feat = aroma_df.drop(columns=["Variety"], errors="ignore")
    feat = feat.apply(pd.to_numeric, errors="coerce")  # force numeric where possible
    aroma_df = pd.concat([aroma_df[["Variety"]], feat], axis=1)

    # Drop columns that are entirely NaN
    aroma_df = aroma_df.dropna(axis=1, how="all")

    # --- 3) Melt to long format ---
    aroma_long = aroma_df.melt(id_vars="Variety", var_name="Compound", value_name="Intensity")
    aroma_long["Intensity"] = pd.to_numeric(aroma_long["Intensity"], errors="coerce").fillna(0)

    # Optional: remove non-positive if you want only detected
    # aroma_long = aroma_long[aroma_long["Intensity"] > 0].copy()

    # --- 4) Select top aromas ---
    if mode == "per_variety":
        top_aromas = (
            aroma_long.groupby("Variety", group_keys=False)
            .apply(lambda g: g.nlargest(top_n, "Intensity"))
            .reset_index(drop=True)
        )
        title_suffix = f"Top {top_n} Compounds per Variety"

    elif mode == "overall":
        top_compounds = (
            aroma_long.groupby("Compound")["Intensity"].mean()
            .nlargest(top_n)
            .index
        )
        top_aromas = aroma_long[aroma_long["Compound"].isin(top_compounds)].copy()
        title_suffix = f"Top {top_n} Overall Compounds Across Varieties"

    else:
        raise ValueError("mode must be either 'per_variety' or 'overall'")

    # --- 5) Pivot for heatmap ---
    heatmap_data = top_aromas.pivot_table(
        index="Variety",
        columns="Compound",
        values="Intensity",
        aggfunc="mean",
        fill_value=0
    )

    # --- 6) Plot ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        cbar_kws={'label': 'Intensity'},
        linewidths=0.2
    )
    plt.title(f"Aroma Intensity Heatmap – {title_suffix}")
    plt.ylabel("Potato Variety")
    plt.xlabel("Aroma Compounds")
    plt.tight_layout()
    plt.show()

    return heatmap_data


def launch_aroma_dashboard_barplot_dbx(
    aroma_matrix,
    port=7000,
    top_n=50,
    host="0.0.0.0",
    default_view="compound",   # "compound" or "variety"
):
    """
    Databricks-friendly Flask dashboard (NO reloader):
      - View=compound: select compound -> barplot Top N varieties
      - View=variety : select variety  -> barplot Top N compounds
    Input:
      aroma_matrix: pd.DataFrame with 'Variety' column + compound columns
    """



    # ----------------------------
    # Validate / standardize input
    # ----------------------------
    if not isinstance(aroma_matrix, pd.DataFrame):
        aroma_matrix = pd.DataFrame(aroma_matrix)

    if "Variety" not in aroma_matrix.columns:
        # allow Variety as index
        aroma_matrix = aroma_matrix.reset_index().rename(columns={"index": "Variety"})

    aroma_matrix["Variety"] = aroma_matrix["Variety"].astype(str).str.strip()
    aroma_matrix = aroma_matrix.dropna(subset=["Variety"]).copy()

    compounds = [c for c in aroma_matrix.columns if c != "Variety"]
    if not compounds:
        raise ValueError("aroma_matrix must have 'Variety' + compound columns")

    varieties = aroma_matrix["Variety"].tolist()
    n_var = len(varieties)
    n_cmp = len(compounds)

    # ----------------------------
    # Find a free port
    # ----------------------------
    def _is_port_free(p):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, p))
            return True
        except OSError:
            return False
        finally:
            s.close()

    if not _is_port_free(port):
        for p in range(port + 1, port + 300):
            if _is_port_free(p):
                port = p
                break

    # ----------------------------
    # Helpers: plot -> base64 PNG
    # ----------------------------
    def _plot_barh_base64(labels, values, title, xlabel="Intensity"):
        fig = plt.figure(figsize=(12, 7))
        plt.barh(labels[::-1], values[::-1])
        plt.xlabel(xlabel)
        plt.title(title)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _top_table_html(df, max_rows=15):
        # small table under plot
        head = df.head(max_rows).copy()
        rows = []
        for _, r in head.iterrows():
            rows.append(
                f"<tr><td style='padding:6px;border-bottom:1px solid #eee'>{r.iloc[0]}</td>"
                f"<td style='padding:6px;border-bottom:1px solid #eee;text-align:right'>{r.iloc[1]:,.3f}</td></tr>"
            )
        return (
            "<table style='border-collapse:collapse;width:60%;margin-top:10px'>"
            "<tr><th style='text-align:left;padding:6px;border-bottom:2px solid #ddd'>Name</th>"
            "<th style='text-align:right;padding:6px;border-bottom:2px solid #ddd'>Intensity</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    # ----------------------------
    # Flask app
    # ----------------------------
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def home():
        view = request.args.get("view", default_view).strip().lower()
        if view not in ("compound", "variety"):
            view = "compound"

        compound = request.args.get("compound", compounds[0])
        if compound not in compounds:
            compound = compounds[0]

        variety = request.args.get("variety", varieties[0] if varieties else "")
        if variety not in set(varieties) and varieties:
            variety = varieties[0]

        try:
            top = int(request.args.get("top", top_n))
        except Exception:
            top = top_n
        top = max(5, min(top, 200))

        # Build content depending on view
        img_b64 = ""
        subtitle = ""
        table_html = ""

        if view == "compound":
            df = aroma_matrix[["Variety", compound]].copy()
            df[compound] = pd.to_numeric(df[compound], errors="coerce").fillna(0.0)
            df = df[df[compound] > 0].sort_values(compound, ascending=False).head(top)

            subtitle = f"Top {len(df)} varieties for compound: <b>{compound}</b>"
            img_b64 = _plot_barh_base64(
                labels=df["Variety"].tolist(),
                values=df[compound].astype(float).tolist(),
                title=f"Top {len(df)} varieties for {compound}",
            )
            table_html = _top_table_html(df[["Variety", compound]].rename(columns={"Variety": "Name"}))

        else:  # view == "variety"
            row = aroma_matrix[aroma_matrix["Variety"] == variety]
            if row.empty:
                subtitle = f"No row found for variety: <b>{variety}</b>"
            else:
                s = row.iloc[0][compounds].copy()
                s = pd.to_numeric(s, errors="coerce").fillna(0.0)
                s = s[s > 0].sort_values(ascending=False).head(top)

                df = pd.DataFrame({"Compound": s.index, "Intensity": s.values})
                subtitle = f"Top {len(df)} compounds for variety: <b>{variety}</b>"
                img_b64 = _plot_barh_base64(
                    labels=df["Compound"].tolist(),
                    values=df["Intensity"].astype(float).tolist(),
                    title=f"Top {len(df)} compounds in {variety}",
                )
                table_html = _top_table_html(df)

        # Dropdown options
        compound_opts = "\n".join(
            [f"<option {'selected' if c==compound else ''}>{c}</option>" for c in compounds]
        )
        variety_opts = "\n".join(
            [f"<option {'selected' if v==variety else ''}>{v}</option>" for v in varieties]
        )

        # Radio buttons
        comp_checked = "checked" if view == "compound" else ""
        var_checked = "checked" if view == "variety" else ""

        html = f"""
        <html>
        <head><title>Potato Aroma Dashboard</title></head>
        <body style="font-family:Arial;margin:25px">
          <h1>Potato Aroma Dashboard (Barplot)</h1>
          <div style="padding:12px;border:1px solid #ddd;border-radius:10px;margin-bottom:20px">
            <div><b>Rows:</b> {n_var} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Compounds:</b> {n_cmp}</div>

            <form method="get" action="/" style="margin-top:12px">
              <div style="margin-bottom:10px">
                <label><input type="radio" name="view" value="compound" {comp_checked}> View by Compound</label>
                &nbsp;&nbsp;
                <label><input type="radio" name="view" value="variety" {var_checked}> View by Variety</label>
              </div>

              <div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap">
                <div>
                  <label><b>Compound:</b></label><br/>
                  <select name="compound" style="min-width:360px">{compound_opts}</select>
                </div>

                <div>
                  <label><b>Variety:</b></label><br/>
                  <select name="variety" style="min-width:260px">{variety_opts}</select>
                </div>

                <div>
                  <label><b>Top N:</b></label><br/>
                  <input type="number" name="top" value="{top}" min="5" max="200" style="width:90px"/>
                </div>

                <div style="margin-top:18px">
                  <button type="submit">Show</button>
                </div>
              </div>
            </form>
          </div>

          <h2>{subtitle}</h2>

          <div style="margin-top:10px">
            <img src="data:image/png;base64,{img_b64}" style="max-width:100%;border:1px solid #eee;border-radius:8px"/>
          </div>

          {table_html}

          <p style="margin-top:25px;color:#888;font-size:12px">
            Tip: switch “View by Compound / View by Variety” to invert the plot.
          </p>
        </body>
        </html>
        """
        return html

    # ----------------------------
    # Start server in background thread (no reloader!)
    # ----------------------------
    def _run():
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # ----------------------------
    # Best-effort Databricks proxy URL
    # ----------------------------
    proxy_url = None
    try:
        # Usually allowed in Databricks notebooks
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        org_id = ctx.workspaceId().get()
        cluster_id = ctx.clusterId().get()
        proxy_url = f"/driver-proxy/o/{org_id}/{cluster_id}/{port}/"
    except Exception:
        proxy_url = f"(proxy url depends on your workspace/cluster) -> /driver-proxy/o/<orgId>/<clusterId>/{port}/"

    print(f" Dashboard running on port {port}")
    print(f" Open in Databricks: {proxy_url}")

    return {"port": port, "proxy_url": proxy_url}




def compute_feature_importance_from_pca(aroma_matrix, n_components=2):
    """
    Computes feature importance of aroma compounds based on PCA loadings.

    Parameters
    ----------
    aroma_matrix : pd.DataFrame
        Aroma intensity matrix with 'Variety' column and compound intensity columns.
    n_components : int, optional
        Number of PCA components to include when computing feature importance (default = 2).

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame showing compounds ranked by absolute PCA loading (importance).
    """
    # ---  Prepare Data ---
    X = aroma_matrix.drop(columns=["Variety"], errors="ignore").fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # ---  PCA ---
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    loadings = pd.DataFrame(
        np.abs(pca.components_.T),
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # ---  Compute overall importance (sum of abs loadings across PCs) ---
    loadings["Total_Importance"] = loadings.sum(axis=1)
    importance_df = loadings.sort_values(by="Total_Importance", ascending=False)

    # ---  Visualization ---
    top_n = 15
    plt.figure(figsize=(10,6))
    plt.barh(importance_df.head(top_n).index[::-1],
              importance_df["Total_Importance"].head(top_n)[::-1],
              color='goldenrod')
    plt.xlabel("Total PCA Loading (|contribution|)")
    plt.title(f"Top {top_n} Aroma Compounds Contributing to PCA Variance")
    plt.tight_layout()
    plt.show()

    print(f" Top {top_n} most contributing compounds:")
    print(importance_df.head(top_n))

    return importance_df




sensory_cols = [
    "Metallic Flavour","Bitter Flavour","Earthy Flavour","Sour Flavour",
    "Fresh Flavour","Sweet Flavour","Root/ Vegetable Flavour",
    "Farmyard (grass/hay) flavour","Bitter Aftertaste","Sour Aftertaste","Sweet Aftertaste", 'Earthy Aroma',
    'Farmyard', 'Sweet', 'Starchy', 'Damp', 'Buttery', 'Fresh', 'Potato Starch','Metalic'
]
def summarize_sensory_traits(sensory_df, flavor_columns):
    """
    Cleans and summarizes sensory (flavour and aroma) data per potato variety.

    Parameters
    ----------
    sensory_df : pd.DataFrame
        Raw sensory data containing columns for flavour intensity and a 'VARIETY' column.
    flavor_columns : list
        List of sensory-related column names to include in the summary.

    Returns
    -------
    sensory_summary : pd.DataFrame
        DataFrame containing the mean sensory trait values for each variety.
    """

    #  Clean variety names 
    sensory_df = sensory_df.copy()
    sensory_df["VARIETY"] = sensory_df["VARIETY"].astype(str).str.strip()

    # --- Step 2: Remove any unnamed or empty columns 
    sensory_df = sensory_df.drop(
        columns=[c for c in sensory_df.columns if str(c).startswith("Unnamed")],
        errors="ignore"
    )

    # Keep only numeric sensory columns + variety ---
    sensory_traits = sensory_df[sensory_cols].apply(pd.to_numeric, errors="coerce")
    sensory_traits = sensory_traits.assign(VARIETY=sensory_df["VARIETY"])

    #  Group by variety and average traits 
    sensory_summary = (
        sensory_traits
        .groupby("VARIETY", as_index=False)
        .mean(numeric_only=True)
    )

    # Print summary info 
    print(f" Rows (unique varieties): {sensory_summary['VARIETY'].nunique()}")
    print(f" Traits averaged: {len(flavor_columns)}")
    print("\n Example of summarized sensory data:")
    print(sensory_summary.head())

    return sensory_summary

def plot_sensory_vs_flavour(df, base_trait="POTATO_FLAVOUR"):
    """
    Creates scatter + regression plots between potato flavour
    and all other sensory traits (based on PCA sensory features).
    """
    # --- Standardize column names ---
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()

    if base_trait not in df.columns:
        raise KeyError(f"'{base_trait}' not found in dataframe columns!")

    # --- Sensory traits (from your PCA biplot) ---
    sensory_traits = [
        "SWEET FLAVOUR", "BITTER FLAVOUR", "EARTHY FLAVOUR", 
        "SOUR FLAVOUR", "FRESH FLAVOUR", "SWEET AFTERTASTE", "BITTER AFTERTASTE",
        "SOUR AFTERTASTE", "FARMYARD (GRASS/HAY) FLAVOUR", "ROOT/ VEGETABLE FLAVOUR",
        "METALLIC FLAVOUR", "DAMP", "BUTTERY", "POTATO STARCH", "EARTHY AROMA"
    ]
    sensory_traits = [t for t in sensory_traits if t in df.columns]

    # --- Color mapping similar to your reference image ---
    colors = {
        "SWEET FLAVOUR": "royalblue",
        "TEXTURE": "red",
        "SOUR FLAVOUR": "green",
        "BITTER FLAVOUR": "orange",
        "SWEET AFTERTASTE": "purple",
        "BUTTERY": "crimson",
        "POTATO STARCH": "darkred",
        "EARTHY FLAVOUR": "brown",
        "METALLIC FLAVOUR": "darkslategray"
    }

    plt.figure(figsize=(10, 6))

    for trait in sensory_traits:
        sns.regplot(
            data=df,
            x=base_trait,
            y=trait,
            scatter_kws={"alpha": 0.7, "s": 45},
            line_kws={"lw": 2},
            color=colors.get(trait, None),
            label=trait.title()
        )

    plt.xlabel("Potato Flavour", fontsize=12, weight="bold")
    plt.ylabel("Sensory Score", fontsize=12)
    plt.title("Relationship between Potato Flavour and Sensory Traits", fontsize=14, weight="bold")
    plt.legend(title="Sensory Traits", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.grid(False)
    plt.show()



def merge_aroma_with_sensory(aroma_df, sensory_summary, how="inner", preview_rows=5):
    """
    Merge aroma matrix (wide format) with sensory data by variety.
    Robust to:
      - Variety being index OR column
      - 'variety' / 'Variety' / first-column being the ID
      - extra reset_index columns like 'level_0' or 'index'
      - missing display() outside notebooks
    """



    aroma_df = aroma_df.copy()
    sensory_summary = sensory_summary.copy()

    # --- helper: ensure Variety exists as a column ---
    def ensure_variety_column(df, df_name="df"):
        # if index is meaningful, bring it back as a column
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()

        # drop common junk cols
        df = df.drop(columns=[c for c in ["level_0", "index"] if c in df.columns], errors="ignore")

        # find a variety-like column
        candidates = [c for c in df.columns if str(c).strip().lower() == "variety"]
        if candidates:
            vcol = candidates[0]
            if vcol != "Variety":
                df = df.rename(columns={vcol: "Variety"})
            return df

        # if not found, assume first column is variety ID
        if len(df.columns) == 0:
            raise KeyError(f"{df_name}: DataFrame has no columns to infer Variety.")
        df = df.rename(columns={df.columns[0]: "Variety"})
        return df

    aroma_df = ensure_variety_column(aroma_df, "aroma_df")
    sensory_summary = ensure_variety_column(sensory_summary, "sensory_summary")

    # --- clean names ---
    aroma_df["Variety"] = aroma_df["Variety"].astype(str).str.strip().str.upper()
    sensory_summary["Variety"] = sensory_summary["Variety"].astype(str).str.strip().str.upper()

    # --- report overlaps ---
    a_set = set(aroma_df["Variety"].unique())
    s_set = set(sensory_summary["Variety"].unique())
    common = sorted(a_set.intersection(s_set))

    print(f"Varieties in Aroma Data   : {len(a_set)}")
    print(f"Varieties in Sensory Data : {len(s_set)}")
    print(f"Common varieties          : {len(common)}")

    # --- merge ---
    merged = pd.merge(aroma_df, sensory_summary, on="Variety", how=how)

    print(f"Merged varieties          : {merged['Variety'].nunique()}")
    print(f"Merged shape              : {merged.shape}")

    # --- preview (safe display) ---
    if preview_rows and preview_rows > 0:
        try:
            from IPython.display import display
            display(merged.head(preview_rows))
        except Exception:
            print(merged.head(preview_rows))

    return merged, common







def plot_aroma_sensory_correlation(
    merged_aroma_sensory,
    sensory_cols,
    aroma_descriptions=None,
    threshold=0.1,
    min_compounds=15,
    save_path=None
):
    """
    Correlation heatmap between aroma compound intensities and sensory traits.

    - Robust to:
      * aroma_descriptions being None / "None" / not a dict
      * sensory column name case/spacing differences
      * duplicate / constant columns
    """


    df = merged_aroma_sensory.copy()

    # ---- 1) Standardize column names ----
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    # drop constant columns early (keeps only columns with >1 unique value)
    df = df.loc[:, df.nunique(dropna=False) > 1]

    # ---- 2) Normalize sensory column names (user list) ----
    sensory_cols_norm = [str(c).strip().lower() for c in sensory_cols]

    # ---- 3) Build a mapping: original -> normalized ----
    col_norm = {c: c.strip().lower() for c in df.columns}

    # ---- 4) Keep numeric columns only (for correlation) ----
    df_num = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)

    # drop columns with too many NaNs (keep if at least half rows are non-NaN, min 3)
    df_num = df_num.dropna(axis=1, thresh=max(3, len(df_num) // 2))

    # IMPORTANT: do NOT drop all rows if you have few NaNs; instead fill remaining NaNs
    # (correlation is sensitive to row deletion)
    df_num = df_num.fillna(0)

    # ---- 5) Identify sensory vs aroma columns ----
    sensory_cols_found = [c for c in df_num.columns if col_norm.get(c, "").lower() in sensory_cols_norm]
    compound_cols = [c for c in df_num.columns if c not in sensory_cols_found]

    print(f"Found {len(compound_cols)} aroma compounds and {len(sensory_cols_found)} sensory traits.")
    if not sensory_cols_found or not compound_cols:
        raise ValueError("No valid sensory or compound columns found for correlation analysis.")

    # ---- 6) Correlation block (compounds x sensory) ----
    corr_matrix = df_num[compound_cols + sensory_cols_found].corr()
    corr_block = corr_matrix.loc[compound_cols, sensory_cols_found].fillna(0)

    # ---- 7) Filter by threshold ----
    strong_corr = corr_block[(corr_block.abs() > threshold).any(axis=1)]

    # ensure at least min_compounds
    if strong_corr.shape[0] < min_compounds:
        top_compounds = (
            corr_block.abs().max(axis=1)
            .sort_values(ascending=False)
            .head(min_compounds)
            .index
        )
        strong_corr = corr_block.loc[top_compounds]

    print(f"Displaying {strong_corr.shape[0]} compounds (threshold={threshold}).")

    # ---- 8) Optional: add descriptions (ONLY if dict) ----
    if isinstance(aroma_descriptions, dict) and len(aroma_descriptions) > 0:
        def norm_name(x: str) -> str:
            x = str(x).strip().lower()
            x = x.replace("-", "").replace(",", "").replace(" ", "")
            return x

        # make normalized lookup once
        desc_lookup = {norm_name(k): v for k, v in aroma_descriptions.items()}

        annotated_index = []
        for compound in strong_corr.index:
            key = norm_name(compound)
            desc = desc_lookup.get(key, "No description")
            annotated_index.append(f"{compound}\n({desc})")
        strong_corr.index = annotated_index

    # ---- 9) Plot ----
    plt.figure(figsize=(16, 9))
    sns.heatmap(
        strong_corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation (r)"}
    )
    plt.title("Correlation Between Aroma Compounds and Sensory Attributes", fontsize=16, pad=15)
    plt.ylabel("Aroma Compounds")
    plt.xlabel("Sensory Attributes")
    plt.tight_layout()
    plt.show()

    # ---- 10) Save ----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        strong_corr.to_csv(save_path)
        print(f"Correlation matrix saved to: {save_path}")

    return strong_corr







def perform_knn_sensory_projection(
    aroma_matrix,
    sensory_summary,
    k=3,
    n_components=3,
    save_projection_path=None,
    save_loadings_path=None,
    scaling_factor=5
):
    """
    KNN impute sensory for non-panel using aroma,
    PCA on sensory (panel), project all varieties into PCA space.
    Works if Variety is a column OR index (and handles duplicate Variety columns).
    """

    def _ensure_single_variety_col(df: pd.DataFrame, label="data"):
        df = df.copy()

        # clean column names
        df.columns = pd.Index(df.columns).map(lambda x: str(x).strip())

        # find variety columns (case-insensitive)
        var_cols = [c for c in df.columns if str(c).strip().lower() == "variety"]

        if len(var_cols) >= 1:
            # keep the first, rename to exactly "Variety"
            first = var_cols[0]
            if first != "Variety":
                df = df.rename(columns={first: "Variety"})
            # drop other duplicate variety columns
            for c in var_cols[1:]:
                df = df.drop(columns=c, errors="ignore")

            # drop any duplicated columns (after rename)
            df = df.loc[:, ~df.columns.duplicated()]
            return df

        # if no Variety column, try index
        idx_name = "" if df.index.name is None else str(df.index.name).strip().lower()
        if idx_name == "variety" or (df.index.dtype == object and df.index.nunique() == len(df)):
            df = df.reset_index()
            df.columns = pd.Index(df.columns).map(lambda x: str(x).strip())
            # first column is the old index
            df = df.rename(columns={df.columns[0]: "Variety"})
            return df

        raise KeyError(f"No Variety column or suitable index found in {label}.")

    # -----------------------------
    # 0) Ensure Variety column (and remove duplicate Variety columns)
    # -----------------------------
    aroma = _ensure_single_variety_col(aroma_matrix, label="aroma_matrix")
    sens  = _ensure_single_variety_col(sensory_summary, label="sensory_summary")

    # Normalize Variety values (now guaranteed Series)
    aroma["Variety"] = aroma["Variety"].astype(str).fillna("").str.strip().str.upper()
    sens["Variety"]  = sens["Variety"].astype(str).fillna("").str.strip().str.upper()

    aroma = aroma.drop_duplicates(subset=["Variety"]).reset_index(drop=True)
    sens  = sens.drop_duplicates(subset=["Variety"]).reset_index(drop=True)

    # -----------------------------
    # 1) Identify columns
    # -----------------------------
    sensory_cols = [c for c in sens.columns if c != "Variety"]
    aroma_cols   = [c for c in aroma.columns if c != "Variety"]

    # numeric aroma only
    for c in aroma_cols:
        aroma[c] = pd.to_numeric(aroma[c], errors="coerce")
    aroma[aroma_cols] = aroma[aroma_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # numeric sensory
    for c in sensory_cols:
        sens[c] = pd.to_numeric(sens[c], errors="coerce")

    # -----------------------------
    # 2) Merge + split into panel / non-panel
    # -----------------------------
    merged = aroma.merge(sens, on="Variety", how="left")

    merged["_has_sensory"] = merged[sensory_cols].notna().any(axis=1)
    panel_df    = merged[merged["_has_sensory"]].reset_index(drop=True)
    nonpanel_df = merged[~merged["_has_sensory"]].reset_index(drop=True)

    print(f"Varieties total (aroma): {merged['Variety'].nunique()}")
    print(f"Panel varieties (sensory present): {panel_df['Variety'].nunique()}")
    print(f"Non-panel varieties (sensory missing): {nonpanel_df['Variety'].nunique()}")

    if panel_df.shape[0] < max(3, k):
        raise ValueError("Not enough panel varieties to run KNN + PCA reliably.")

    # -----------------------------
    # 3) KNN imputation in aroma space
    # -----------------------------
    X_panel    = panel_df[aroma_cols].values
    X_nonpanel = nonpanel_df[aroma_cols].values
    Y_panel    = panel_df[sensory_cols].values  # may have NaNs

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_panel)
    _, indices = nn.kneighbors(X_nonpanel)

    Y_nonpanel_pred = np.array([np.nanmean(Y_panel[idx], axis=0) for idx in indices])

    col_means = np.nanmean(Y_panel, axis=0)

    # fill remaining NaNs in predictions
    m = np.isnan(Y_nonpanel_pred)
    if m.any():
        Y_nonpanel_pred[m] = np.take(col_means, np.where(m)[1])

    # fill NaNs in panel before PCA
    Y_panel_filled = Y_panel.copy()
    m2 = np.isnan(Y_panel_filled)
    if m2.any():
        Y_panel_filled[m2] = np.take(col_means, np.where(m2)[1])

    # -----------------------------
    # 4) PCA on sensory (panel) + project non-panel
    # -----------------------------
    scaler = StandardScaler()
    Y_panel_scaled    = scaler.fit_transform(Y_panel_filled)
    Y_nonpanel_scaled = scaler.transform(Y_nonpanel_pred)

    n_components = min(n_components, Y_panel_scaled.shape[1], Y_panel_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=0)

    pcs_panel    = pca.fit_transform(Y_panel_scaled)
    pcs_nonpanel = pca.transform(Y_nonpanel_scaled)

    panel_pcs = pd.DataFrame(pcs_panel, columns=[f"PC{i+1}" for i in range(n_components)])
    panel_pcs["Variety"] = panel_df["Variety"].values
    panel_pcs["is_panel"] = True

    nonpanel_pcs = pd.DataFrame(pcs_nonpanel, columns=[f"PC{i+1}" for i in range(n_components)])
    nonpanel_pcs["Variety"] = nonpanel_df["Variety"].values
    nonpanel_pcs["is_panel"] = False

    pc_df = pd.concat([panel_pcs, nonpanel_pcs], ignore_index=True)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=sensory_cols
    )

    print("\nPCA explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
    print("Total projected varieties:", pc_df["Variety"].nunique())

    # -----------------------------
    # 5) Plots
    # -----------------------------
    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings_df, annot=True, cmap="coolwarm", center=0)
    plt.title("PCA Loadings of Sensory Traits")
    plt.tight_layout()
    plt.show()

    if n_components >= 2:
        plt.figure(figsize=(9, 7))
        plt.scatter(pc_df.loc[pc_df["is_panel"], "PC1"],
                    pc_df.loc[pc_df["is_panel"], "PC2"],
                    label="Panel", alpha=0.75)
        plt.scatter(pc_df.loc[~pc_df["is_panel"], "PC1"],
                    pc_df.loc[~pc_df["is_panel"], "PC2"],
                    marker="*", s=120, label="Non-panel", alpha=0.85)

        for _, row in pc_df.iterrows():
            plt.text(row["PC1"], row["PC2"], row["Variety"], fontsize=7, alpha=0.8)

        for feature in sensory_cols:
            x = loadings_df.loc[feature, "PC1"] * scaling_factor
            y = loadings_df.loc[feature, "PC2"] * scaling_factor
            plt.arrow(0, 0, x, y, alpha=0.5, head_width=0.08, length_includes_head=True)
            plt.text(x * 1.08, y * 1.08, feature, fontsize=8)

        plt.axhline(0, linestyle="--", linewidth=1)
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title("PCA Biplot: Varieties projected by Sensory PCA (KNN-imputed for non-panel)")
        plt.legend()
        plt.tight_layout()
        plt.grid(False)
        plt.show()

    # -----------------------------
    # 6) Save
    # -----------------------------
    if save_projection_path:
        os.makedirs(os.path.dirname(save_projection_path), exist_ok=True)
        pc_df.to_csv(save_projection_path, index=False)
        print(f"Projection saved to: {save_projection_path}")

    if save_loadings_path:
        os.makedirs(os.path.dirname(save_loadings_path), exist_ok=True)
        loadings_df.to_csv(save_loadings_path)
        print(f"Loadings saved to: {save_loadings_path}")

    return pc_df, pca, loadings_df



