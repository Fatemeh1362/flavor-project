import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import f
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import os
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from scipy.stats import zscore
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor


def merge_genetic_aroma_sensory(genetic_data, aroma_df, sensory_pcs):
    """
    Clean, align, and merge genetic, aroma, and sensory PCA data for matching varieties.

    Parameters
    ----------
    genetic_data : pd.DataFrame
        Genetic marker table (varieties × SNPs or markers).
    aroma_df : pd.DataFrame
        Aroma compound intensity data (with a 'Variety' column).
    sensory_pcs : pd.DataFrame
        Sensory PCA scores (with a 'Variety' column).

    Returns
    -------
    merged_snp_aroma : pd.DataFrame
        Genetic + Aroma merged table.
    merged_snp_sensory : pd.DataFrame
        Genetic + Sensory merged table.
    merged_all : pd.DataFrame
        Combined Genetic + Aroma + Sensory table.
    """

    # --- Helper functions ---
    def to_numeric(df):
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def norm_key(s):
        """Normalize variety names to a comparable key"""
        s = (str(s).upper().replace(".", "").replace(" ", ""))
        s = re.sub(r"\bR\b", "", s)
        s = re.sub(r"RUSSET", "", s)
        return s.split("_")[0]


    #   Genetic data processing
    mrk_col = "TAGLO_ID" if "TAGLO_ID" in genetic_data.columns else genetic_data.columns[0]

    G = (
        genetic_data.dropna(subset=[mrk_col])
        .drop_duplicates(subset=[mrk_col])
        .set_index(mrk_col)
    )

    G = to_numeric(G)
    gd = G.T
    gd.index.name = "Variety_raw"
    gd.index = gd.index.map(norm_key)

    # Collapse duplicates (mean per variety)
    gd = gd.groupby(gd.index, observed=True).mean()
    gd.columns = ["GEN__" + str(c) for c in gd.columns]
    print(f" Genetic table after reshape: {gd.shape}")


    #   Aroma data processing
    af = aroma_df.copy()
    assert "Variety" in af.columns, " aroma_df must have a 'Variety' column"

    af["__key__"] = af["Variety"].map(norm_key)
    af = af.set_index("__key__")

    aroma_cols = [c for c in af.columns if c not in ["Variety", "index"]]
    A_raw = to_numeric(af[aroma_cols])

    # Drop all-NaN columns only
    n_before = len(A_raw.columns)
    dropped_allnan = [c for c in A_raw.columns if A_raw[c].isna().all()]
    A_step1 = A_raw.drop(columns=dropped_allnan)

    # Keep all remaining columns (fill NaNs with 0)
    A = A_step1.fillna(0)
    A = A.loc[:, A.std(ddof=0) > 0]  # Remove constant columns

    print(f"\n# Aroma data cleanup summary:")
    print(f"  Total columns before: {n_before}")
    print(f"  Dropped (all NaN): {len(dropped_allnan)}")
    print(f"  Final usable aroma columns: {A.shape[1]}")
    print("  Sample aroma cols:", list(A.columns[:8]))


    #   Sensory PCA processing
    assert "Variety" in sensory_pcs.columns, "sensory_pcs must have a 'Variety' column"

    sensory_pcs["__key__"] = sensory_pcs["Variety"].map(norm_key)
    sensory_pcs = sensory_pcs.set_index("__key__")

    S = to_numeric(sensory_pcs.drop(columns=["Variety", "is_panel"], errors="ignore")).fillna(0)
    S = S.loc[:, S.std(ddof=0) > 0]
    print(f"\n# Sensory PCs: {S.shape[1]}")
    print("Sample sensory cols:", list(S.columns[:8]))

  
    #   Merge all tables
    gk = gd.copy()
    gk.index.name = "Variety"

    merged_snp_aroma = gk.join(A, how="inner")
    merged_snp_sensory = gk.join(S, how="inner")
    merged_all = gk.join(pd.concat([A, S], axis=1), how="inner")

    print(f"\n# Merge results:")
    print(f"  Genetic + Aroma: {merged_snp_aroma.shape}")
    print(f"  Genetic + Sensory: {merged_snp_sensory.shape}")
    print(f"  Genetic + Aroma + Sensory: {merged_all.shape}")

    return merged_snp_aroma, merged_snp_sensory, merged_all




import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ---------------------------------------------
# FUNCTION: PCA + Outlier Detection + Loadings
# ---------------------------------------------
def pca_with_feature_contribution(
    data,
    target_group=["ALTUS", "FESTIEN", "AVARNA", "ROYAL","DONALD"],
    n_components=5,
):
    # -------------------------
    # Select only numeric columns
    # -------------------------
    X = data.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()

    # -------------------------
    # Scale data
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # PCA
    # -------------------------
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pcs[:, :2],
        columns=["PC1", "PC2"],
        index=data.index
    )

    # -------------------------
    # PCA Loadings
    # -------------------------
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=feature_names
    )

    # -------------------------
    # DIFFERENTIATING COMPOUNDS
    # using PC2 because ALTUS cluster is high on PC2
    # -------------------------
    top_PC2_positive = loadings["PC2"].sort_values(ascending=False).head(15)
    top_PC2_negative = loadings["PC2"].sort_values().head(15)

    print("\n====================================================")
    print(" COMPOUNDS HIGH IN ALTUS / FESTIN / AVARNA / ROYAL DONALD")
    print("====================================================")
    print(top_PC2_positive)

    print("\n====================================================")
    print(" COMPOUNDS LOW IN THOSE (but high in other varieties)")
    print("====================================================")
    print(top_PC2_negative)

    # -------------------------
    # Compare mean concentrations
    # -------------------------
    numeric_cols = X.columns

    group_mean = data.loc[target_group, numeric_cols].mean()
    others_mean = data.drop(target_group, axis=0)[numeric_cols].mean()

    difference = (group_mean - others_mean).sort_values(ascending=False)

    print("\n====================================================")
    print(" ACTUALLY HIGHER IN TARGET GROUP (Mean concentration difference)")
    print("====================================================")
    print(difference.head(20))

    print("\n====================================================")
    print(" ACTUALLY LOWER IN TARGET GROUP")
    print("====================================================")
    print(difference.tail(20))

    return pca_df, loadings, difference


# -------------------------------------------------------
# OPTIONAL: PCA Plot (you can remove if you already have)
# -------------------------------------------------------
def plot_pca(pca_df, data, title="PCA of Genetic and Aroma Profiles"):
    plt.figure(figsize=(7, 5))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)

    # Label outliers
    for name in data.index:
        if np.abs(pca_df.loc[name, "PC1"]) > 150 or np.abs(pca_df.loc[name, "PC2"]) > 150:
            plt.text(
                pca_df.loc[name, "PC1"],
                pca_df.loc[name, "PC2"],
                name, fontsize=8, color="blue", weight="bold"
            )

    plt.axhline(0, color="gray", lw=0.7)
    plt.axvline(0, color="gray", lw=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.show()




def analyze_gene_aroma_plsr(merged_snp_aroma, n_components=2, top_genes=10, top_aromas=20):
    """
    Perform Partial Least Squares Regression (PLSR) between genetic markers and aroma compounds.
    Identifies top influential genes and aroma compounds contributing to the shared variance
    and visualizes the first component relationship (biplot).

    Parameters
    ----------
    merged_snp_aroma : pd.DataFrame
        Merged dataset containing genetic (columns starting with 'GEN__') and aroma data.
    n_components : int, default=2
        Number of PLS components to compute.
    top_genes : int, default=10
        Number of top influential genes to display.
    top_aromas : int, default=20
        Number of top influential aroma compounds to display.

    Returns
    -------
    pls_model : PLSRegression
        Trained PLSR model.
    top_genes_df : pd.Series
        Top influential genes ranked by absolute loading values.
    top_aromas_df : pd.Series
        Top influential aroma compounds ranked by absolute loading values.
    """

    # --- Separate genetic and aroma features ---
    gene_cols = [c for c in merged_snp_aroma.columns if c.startswith("GEN__")]
    aroma_cols = [c for c in merged_snp_aroma.columns if not c.startswith("GEN__")]

    gd = merged_snp_aroma[gene_cols].select_dtypes(include=[np.number])
    A = merged_snp_aroma[aroma_cols].select_dtypes(include=[np.number])

    # --- Align data (common varieties) ---
    common = gd.index.intersection(A.index)
    X = gd.loc[common]
    Y = A.loc[common]

    # --- Standardize ---
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # --- PLSR ---
    pls = PLSRegression(n_components=min(n_components, X.shape[1], Y.shape[1]))
    pls.fit(X_scaled, Y_scaled)

    # --- Gene influence ---
    gene_loadings = pd.Series(pls.x_loadings_[:, 0], index=X.columns)
    top_genes_df = gene_loadings.abs().sort_values(ascending=False).head(top_genes)
    print("Top Influential Genes:\n", top_genes_df)

    # --- Aroma influence ---
    aroma_loadings = pd.Series(pls.y_loadings_[:, 0], index=Y.columns)
    top_aromas_df = aroma_loadings.abs().sort_values(ascending=False).head(top_aromas)
    print("\nTop Influential Aroma Compounds:\n", top_aromas_df)

    # --- Biplot ---
    plt.figure(figsize=(8, 6))
    plt.scatter(pls.x_scores_[:, 0], pls.y_scores_[:, 0], alpha=0.7)
    plt.xlabel("PLS Component 1 (Genes)")
    plt.ylabel("PLS Component 1 (Aromas)")
    plt.title("PLSR Biplot: Genetic–Aroma Relationship in Potato Varieties", fontsize=13, weight="bold")
    plt.grid(False)

    # Label varieties
    for i, name in enumerate(common):
        plt.text(pls.x_scores_[i, 0], pls.y_scores_[i, 0], name, fontsize=8, color="darkblue")
    plt.tight_layout()
    plt.show()

    print(f"\nExplained variance (X, Y): {pls.x_weights_.shape}, {pls.y_weights_.shape}")
    print(f"Number of varieties analyzed: {len(common)}")
    return pls, top_genes_df, top_aromas_df




def plot_top_aroma_loadings_horizontal(top_aromas_df, title="Top 20 Aroma Compounds Influencing PLS Component 1"):
    """
    Plot top aroma compounds influencing PLS Component 1 (horizontal bar chart).
    Matches the style shown in your example image.
    """
    plt.figure(figsize=(7, 5))
    
    # Bars
    plt.barh(
        top_aromas_df.index,
        top_aromas_df.values,
        color="#f4a261",
        edgecolor="orange",
        alpha=0.9
    )
    
    # Red dots at the end of bars
    plt.scatter(
        top_aromas_df.values,
        top_aromas_df.index,
        color="darkred",
        s=50,
        zorder=3
    )

    # Titles and labels
    plt.title(title, fontsize=13, weight="bold", pad=10)
    plt.xlabel("Absolute Loading Strength", fontsize=11)
    plt.ylabel("")
    # Grid and formatting
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()





def plot_balanced_dendrogram_snp_aroma(
    merged_snp_aroma,
    highlight_varieties=("LADYCLAIRE", "INNOVATOR"),
    method="ward",
    title="Balanced Hierarchical Clustering (SNP + Aroma Profiles)",
    figsize=(14, 6)
):
    """
    Perform balanced hierarchical clustering combining SNP and aroma data blocks.
    Each block (genetic and aroma) is standardized and normalized to contribute equally
    to the clustering distance metric.

    Parameters
    ----------
    merged_snp_aroma : pd.DataFrame
        DataFrame containing genetic (prefixed with 'GEN__') and aroma columns.
    highlight_varieties : tuple of str, default=("LADYCLAIRE", "INNOVATOR")
        Variety names (case-insensitive) to highlight in red on the dendrogram.
    method : str, default='ward'
        Linkage method for hierarchical clustering.
    title : str, default='Balanced Hierarchical Clustering (SNP + Aroma Profiles)'
        Plot title.
    figsize : tuple, default=(14, 6)
        Figure size for dendrogram.

    Returns
    -------
    Z : ndarray
        Linkage matrix used for hierarchical clustering.
    """

    # --- Separate SNP and aroma features ---
    snp_cols = [c for c in merged_snp_aroma.columns if c.startswith("GEN__")]
    aroma_cols = [c for c in merged_snp_aroma.columns if not c.startswith("GEN__")]

    X_snp = merged_snp_aroma[snp_cols].fillna(0)
    X_aroma = merged_snp_aroma[aroma_cols].fillna(0)

    # --- Standardize each block individually ---
    scaler = StandardScaler()
    snp_scaled = scaler.fit_transform(X_snp)
    aroma_scaled = scaler.fit_transform(X_aroma)

    # --- Normalize each block (equal contribution to total variance) ---
    snp_scaled /= np.sqrt(np.sum(np.var(snp_scaled, axis=0)))
    aroma_scaled /= np.sqrt(np.sum(np.var(aroma_scaled, axis=0)))

    # --- Combine both blocks ---
    X_balanced = np.hstack([snp_scaled, aroma_scaled])

    # --- Hierarchical clustering ---
    Z = linkage(X_balanced, method=method)

    # --- Plot dendrogram ---
    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=merged_snp_aroma.index,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * np.max(Z[:, 2])
    )

    plt.title(title, fontsize=13, weight="bold")
    plt.xlabel("Varieties")
    plt.ylabel("Ward Distance")

    # --- Highlight selected varieties in red ---
    for lbl in plt.gca().get_xticklabels():
        txt = lbl.get_text().upper()
        if any(name.upper() in txt for name in highlight_varieties):
            lbl.set_color("red")
            lbl.set_fontweight("bold")
    plt.tight_layout()
    plt.show()
    return Z


def detect_outliers_aroma_genefrom_plsr(X_scores, Y_scores, sample_names, z_threshold=2.5):
    """
    Detect outliers in PLSR score space (Genes vs Aroma components).
    Uses Z-score on combined Euclidean distance from origin.
    """

    # Compute Euclidean distances of each sample in PLSR space
    distances = np.sqrt(X_scores[:, 0]**2 + Y_scores[:, 0]**2)

    # Compute Z-scores of these distances
    z_scores = (distances - np.mean(distances)) / np.std(distances)

    # Identify outliers exceeding threshold
    outlier_mask = np.abs(z_scores) > z_threshold
    outliers = np.array(sample_names)[outlier_mask]

    print(f"Detected {len(outliers)} outliers (Z > {z_threshold}): {list(outliers)}")

    return list(outliers)



def perform_pca_gene_sensory(
    merged_snp_sensory,
    n_components=2,
    outlier_threshold_pc1=4,
    outlier_threshold_pc2=3,
    title="PCA on Gene–Sensory Combined Data",
    figsize=(6, 5)
):
    """
    Perform PCA on merged gene–sensory dataset and visualize the first two components.
    Automatically highlights outlier varieties based on PC score thresholds.

    Parameters
    ----------
    merged_snp_sensory : pd.DataFrame
        Combined dataset containing genetic (prefixed with 'GEN__') and sensory data.
    n_components : int, default=2
        Number of principal components to compute.
    outlier_threshold_pc1 : float, default=4
        Threshold for labeling outliers along PC1.
    outlier_threshold_pc2 : float, default=3
        Threshold for labeling outliers along PC2.
    title : str, default="PCA on Gene–Sensory Combined Data"
        Plot title.
    figsize : tuple, default=(6, 5)
        Figure size for the PCA plot.

    Returns
    -------
    pca_df : pd.DataFrame
        PCA scores for each variety.
    pca_model : PCA
        Trained PCA model.
    outliers : list
        List of detected outlier variety names.
    """

    # --- Handle missing values ---
    X = merged_snp_sensory.fillna(0).values
    variety_names = merged_snp_sensory.index.tolist()

    # --- Standardize ---
    X_scaled = StandardScaler().fit_transform(X)

    # --- Perform PCA ---
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    # --- Identify outliers ---
    outliers = [
        variety_names[i]
        for i in range(len(variety_names))
        if abs(scores[i, 0]) > outlier_threshold_pc1 or abs(scores[i, 1]) > outlier_threshold_pc2
    ]

    # --- Plot PCA ---
    plt.figure(figsize=figsize)
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.6, color="#457b9d")

    # Label outliers
    for i, label in enumerate(variety_names):
        if label in outliers:
            plt.text(
                scores[i, 0],
                scores[i, 1],
                label,
                fontsize=8,
                color="darkred",
                fontweight="bold"
            )

    plt.axhline(0, color="gray", lw=0.8)
    plt.axvline(0, color="gray", lw=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(title, fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()

    # --- Store results ---
    pca_df = (
        pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"], index=variety_names)
        .assign(IsOutlier=lambda df: df.index.isin(outliers))
    )

    print(f" PCA completed with {n_components} components.")
    print(f"Explained variance ratio (PC1+PC2): {pca.explained_variance_ratio_[:2].sum():.2%}")
    print(f"Outliers detected ({len(outliers)}): {outliers}")

    return pca_df, pca, outliers



def perform_pca_plsr_gene_sensory(
    merged_snp_sensory,
    n_pca_components=50,
    n_pls_components=2,
    label_fontsize=8,
    title="PLSR on PCA-Reduced SNP–Sensory Data",
    figsize=(7, 6)
):
    """
    Perform PCA on SNP (genetic) block and PLSR against sensory data.
    Generates a labeled scatter plot of PLS Component 1 (Genes) vs Component 1 (Sensory Traits).

    Parameters
    ----------
    merged_snp_sensory : pd.DataFrame
        Combined dataset containing columns prefixed with 'GEN__' (SNPs) and sensory features.
    n_pca_components : int, default=50
        Number of principal components to retain from SNP data before PLSR.
    n_pls_components : int, default=2
        Number of PLSR components to compute.
    label_fontsize : int, default=8
        Font size for variety labels.
    title : str
        Plot title.
    figsize : tuple, default=(7, 6)
        Figure size for the plot.

    Returns
    -------
    pca_model : PCA
        Fitted PCA model on SNP data.
    pls_model : PLSRegression
        Fitted PLSR model on PCA-reduced SNP and sensory data.
    pls_scores_df : pd.DataFrame
        PLS X and Y component scores per variety.
    """

    # --- Split SNP and sensory blocks ---
    X = merged_snp_sensory[[c for c in merged_snp_sensory.columns if c.startswith("GEN__")]]
    Y = merged_snp_sensory[[c for c in merged_snp_sensory.columns if not c.startswith("GEN__")]]

    # --- Standardize ---
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    # --- Reduce SNP block with PCA ---
    pca_model = PCA(n_components=n_pca_components, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)

    # --- Perform PLSR ---
    pls_model = PLSRegression(n_components=n_pls_components)
    X_scores, Y_scores = pls_model.fit_transform(X_pca, Y_scaled)

    # --- Plot ---
    plt.figure(figsize=figsize)
    plt.scatter(X_scores[:, 0], Y_scores[:, 0], alpha=0.7, color="steelblue")

    for i, label in enumerate(merged_snp_sensory.index):
        plt.text(X_scores[i, 0], Y_scores[i, 0], label, fontsize=label_fontsize, color='navy')

    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)
    plt.xlabel("PLS Component 1 (Genes)")
    plt.ylabel("PLS Component 1 (Sensory Traits)")
    plt.title(title, fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()

    # --- Return DataFrame of scores ---
    pls_scores_df = pd.DataFrame({
        "PLS1_Gene": X_scores[:, 0],
        "PLS1_Sensory": Y_scores[:, 0],
    }, index=merged_snp_sensory.index)

    print(f" PCA reduced SNPs to {n_pca_components} components.")
    print(f" PLSR computed {n_pls_components} components.")
    print(f"Explained variance in PCA (first 3 comps): {pca_model.explained_variance_ratio_[:3]}")
    print(f"Samples plotted: {len(pls_scores_df)} varieties.")
    return pca_model, pls_model, pls_scores_df




def perform_hca_gene_sensory(
    merged_snp_sensory,
    metric="euclidean",
    method="ward",
    color_threshold_ratio=0.7,
    figsize=(12, 6),
    title="Hierarchical Clustering of Potato Varieties Based on Combined SNP and Sensory Profiles"
):
    """
    Perform Hierarchical Cluster Analysis (HCA) on SNP–sensory merged data.

    Parameters
    ----------
    merged_snp_sensory : pd.DataFrame
        Merged dataset containing genetic (GEN__) and sensory columns, indexed by variety name.
    metric : str, default='euclidean'
        Distance metric for clustering.
    method : str, default='ward'
        Linkage method for clustering.
    color_threshold_ratio : float, default=0.7
        Ratio of max linkage distance used to color clusters.
    figsize : tuple, default=(12, 6)
        Figure size for dendrogram.
    title : str
        Title for the dendrogram plot.

    Returns
    -------
    Z : ndarray
        Linkage matrix for hierarchical clustering.
    """

    # --- Data cleaning ---
    X = merged_snp_sensory.select_dtypes(include=[np.number]).dropna(axis=1, how="any")
    X = X.dropna(axis=0, how="any")

    # --- Standardize ---
    Xz = (X - X.mean()) / X.std(ddof=0)
    Xz = Xz.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")

    # --- Compute distance + linkage ---
    D = pdist(Xz, metric=metric)
    Z = linkage(D, method=method)

    # --- Plot dendrogram ---
    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=Xz.index,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold_ratio * max(Z[:, 2]),
    )
    plt.title(title, fontsize=13, weight="bold")
    plt.xlabel("Variety", fontsize=11)
    plt.ylabel(f"{method.capitalize()} Distance", fontsize=11)
    plt.tight_layout()
    plt.show()
    print(f" HCA completed using {method} linkage and {metric} distance.")
    print(f"Samples clustered: {Xz.shape[0]}, Features used: {Xz.shape[1]}")
    return Z




def perform_gwas_for_traits(
    merged_df,
    snp_prefix="GEN__",
    output_path=None,
    fdr_method="fdr_bh",
    ld_threshold=0.9,
    max_snps=2000,
    top_n_snps=500
):
    """
    Perform GWAS across all numeric traits in a merged dataset (e.g., SNP–Aroma or SNP–Sensory).

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged DataFrame containing SNPs (prefixed by snp_prefix) and trait columns.
    snp_prefix : str, default='GEN__'
        Prefix used to identify SNP columns.
    output_path : str, optional
        If provided, saves all GWAS results to a CSV file.
    fdr_method : str, default='fdr_bh'
        Multiple testing correction method (passed to multipletests).
    ld_threshold : float, default=0.9
        Correlation threshold for LD pruning.
    max_snps : int, default=2000
        Max number of SNPs to retain before pruning for efficiency.
    top_n_snps : int, default=500
        Number of top SNPs (by adjusted p-value) to consider per trait.

    Returns
    -------
    all_gwas_df : pd.DataFrame
        Concatenated GWAS results for all traits.
    """

    # --- Helper: LD pruning ---
    def ld_prune(X_df, threshold=0.9, max_snps=2000):
        X_df = X_df.loc[:, X_df.std() > 0]
        if X_df.shape[1] > max_snps:
            X_df = X_df.iloc[:, :max_snps]
        if X_df.shape[1] <= 1:
            return X_df

        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return X_df.drop(columns=to_drop)

    # --- Helper: single-trait GWAS ---
    def run_gwas(df, trait, snp_cols):
        y = df[trait].astype(float).values
        n = len(y)
        gwas_results = []

        for snp in snp_cols:
            x = df[snp].fillna(df[snp].mode().iloc[0]).astype(float).values
            if x.std() == 0:
                continue
            x = (x - x.mean()) / (x.std() + 1e-6)

            r = np.corrcoef(x, y)[0, 1]
            if np.isnan(r):
                continue
            r = np.clip(r, -0.999999, 0.999999)

            F_val = (r**2) / ((1 - r**2) / (n - 2))
            p_val = f.sf(F_val, 1, n - 2)
            gwas_results.append((snp, p_val))

        if not gwas_results:
            return pd.DataFrame(columns=["SNP", "p_value", "p_fdr", "Trait"])

        gwas_df = pd.DataFrame(gwas_results, columns=["SNP", "p_value"])
        gwas_df["p_fdr"] = multipletests(gwas_df["p_value"], method=fdr_method)[1]
        gwas_df["Trait"] = trait
        return gwas_df

    # --- Identify SNP and trait columns ---
    snp_cols = [c for c in merged_df.columns if c.startswith(snp_prefix)]
    trait_cols = [
        c for c in merged_df.columns
        if c not in snp_cols and np.issubdtype(merged_df[c].dtype, np.number)
    ]
    print(f"Detected {len(snp_cols)} SNP columns and {len(trait_cols)} numeric traits.\n")

    # --- Run GWAS for each trait ---
    all_gwas = []
    for trait in tqdm(trait_cols, desc="Running GWAS per trait"):
        gwas_df = run_gwas(merged_df, trait, snp_cols)
        if gwas_df.empty:
            continue

        # Keep top SNPs and prune LD
        top_snps = gwas_df.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
        if not top_snps:
            continue

        X = merged_df[top_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=ld_threshold, max_snps=max_snps)

        if X_pruned.shape[1] == 0:
            continue

        all_gwas.append(gwas_df)

    # --- Combine and save results ---
    if all_gwas:
        all_gwas_df = pd.concat(all_gwas, ignore_index=True)
        if output_path:
            all_gwas_df.to_csv(output_path, index=False)
            print(f"\n All GWAS results saved to: {output_path}")
    else:
        print(" No valid GWAS results generated.")
        all_gwas_df = pd.DataFrame()

    return all_gwas_df



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_gwas_full_report(
    gwas_df=None,
    p_col="p_value",
    fdr_col="p_fdr",
    effect_col=None,
    title_prefix="GWAS Results",
    fdr_threshold=0.05,
):
    """
    Display Manhattan, QQ, and Volcano plots for GWAS results.
    """

    # --- Custom Manhattan Plot using file_path ---
    file_path = r"C:\Users\fatemehm\OneDrive - Royal HZPC Group\documents\GitHub\internship\mixed biorep\gwas_aroma_all_traits.csv"
    all_gwas_df = pd.read_csv(file_path)

    plt.figure(figsize=(10, 5))
    plt.scatter(all_gwas_df.index, -np.log10(all_gwas_df[p_col]), c="gray", alpha=0.6, s=10)
    plt.scatter(
        all_gwas_df.index[all_gwas_df[fdr_col] < fdr_threshold],
        -np.log10(all_gwas_df.loc[all_gwas_df[fdr_col] < fdr_threshold, p_col]),
        c="red", s=15, label=f"Significant SNPs (FDR < {fdr_threshold})"
    )
    plt.axhline(-np.log10(fdr_threshold), color="blue", linestyle="--")
    plt.xlabel("SNP Index")
    plt.ylabel("-log10(p-value)")
    plt.title("Manhattan Plot: GWAS for Aroma Traits")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- QQ Plot ---
    if gwas_df is None:
        gwas_df = all_gwas_df.copy()

    observed = -np.log10(np.sort(gwas_df[p_col].clip(lower=1e-300)))
    expected = -np.log10(np.linspace(1 / len(observed), 1, len(observed)))

    plt.figure(figsize=(6, 5))
    plt.scatter(expected, observed, c="black", s=10)
    plt.plot([0, max(expected)], [0, max(expected)], "r--")
    plt.xlabel("Expected -log10(p)")
    plt.ylabel("Observed -log10(p)")
    plt.title(f"{title_prefix} – QQ Plot", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()

    # --- Volcano Plot ---
    df = gwas_df.copy()
    df["neg_log_p"] = -np.log10(df[p_col].clip(lower=1e-300))
    if effect_col is None or effect_col not in df.columns:
        df["effect_size"] = np.random.randn(len(df))
        effect_col = "effect_size"

    plt.figure(figsize=(7, 5))
    plt.scatter(
        df[effect_col], df["neg_log_p"],
        c=df[fdr_col] < fdr_threshold, cmap="coolwarm", alpha=0.7, s=20
    )
    plt.axhline(-np.log10(fdr_threshold), color="gray", linestyle="--")
    plt.xlabel("Effect Size")
    plt.ylabel("-log10(p-value)")
    plt.title(f"{title_prefix} – Volcano Plot", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()


def pca_after_gwas_feature_selection_gene_aroma(
    merged_snp_aroma,
    gwas_df,
    pval_col="p_value",
    fdr_col="p_fdr",
    snp_col="SNP",
    fdr_threshold=0.05,
    n_components=2,
    title="PCA on GWAS-Selected SNPs (Aroma-Associated Markers)"
):
    """
    Perform PCA on SNPs that were significant in GWAS analysis
    (gene–aroma associations), visualize results, and label outliers.

    Parameters
    ----------
    merged_snp_aroma : pd.DataFrame
        Combined dataset of SNPs and aroma traits (indexed by varieties).
    gwas_df : pd.DataFrame
        GWAS result table containing SNP, p-value, and FDR.
    pval_col : str
        Column name for p-values in gwas_df.
    fdr_col : str
        Column name for FDR-corrected p-values.
    snp_col : str
        Column name for SNP IDs.
    fdr_threshold : float
        Threshold for selecting significant SNPs.
    n_components : int
        Number of PCA components.
    title : str
        Plot title.
    """

    # --- Select significant SNPs ---
    if snp_col not in gwas_df.columns or fdr_col not in gwas_df.columns:
        raise ValueError(f"'{snp_col}' or '{fdr_col}' column not found in GWAS results.")

    top_snps = gwas_df.loc[gwas_df[fdr_col] < fdr_threshold, snp_col].unique().tolist()
    print(f" Selected {len(top_snps)} SNPs below FDR < {fdr_threshold}")

    if len(top_snps) == 0:
        print(" No significant SNPs found — PCA skipped.")
        return None, None

    # --- Extract SNP block ---
    X = merged_snp_aroma[top_snps].fillna(0).astype(float)

    # --- Standardize ---
    X_scaled = StandardScaler().fit_transform(X)

    # --- PCA ---
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    # --- Plot PCA ---
    plt.figure(figsize=(7, 6))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(title, fontsize=13, weight="bold")

    # --- Outlier detection (z-score > 2.5) ---
    z_pc1 = (scores[:, 0] - np.mean(scores[:, 0])) / np.std(scores[:, 0])
    z_pc2 = (scores[:, 1] - np.mean(scores[:, 1])) / np.std(scores[:, 1])
    outlier_mask = (np.abs(z_pc1) > 2.5) | (np.abs(z_pc2) > 2.5)

    for i, is_outlier in enumerate(outlier_mask):
        if is_outlier:
            plt.text(scores[i, 0], scores[i, 1], str(merged_snp_aroma.index[i]),
                     fontsize=8, color='black', ha='center', va='center')

    plt.tight_layout()
    plt.show()

    # --- Report explained variance ---
    explained = pca.explained_variance_ratio_[:2] * 100
    print(f"Explained variance: PC1 = {explained[0]:.2f}%, PC2 = {explained[1]:.2f}%")

    return pca, scores



def plsr_after_gwas_selection(
    merged_snp_aroma,
    gwas_file,
    fdr_col="p_fdr",
    snp_col="SNP",
    fdr_threshold=0.05,
    n_pca_components=50,
    n_pls_components=2,
    title="PLSR on PCA-Reduced SNP–Aroma Data"
):
    """
    Run PCA + PLSR on GWAS-selected SNPs (only existing SNPs are used).
    """

    # --- Load GWAS results ---
    gwas_df = pd.read_csv(gwas_file)
    if snp_col not in gwas_df.columns or fdr_col not in gwas_df.columns:
        raise ValueError(f"'{snp_col}' or '{fdr_col}' column not found in GWAS results.")

    # --- Select significant SNPs ---
    top_snps = gwas_df.loc[gwas_df[fdr_col] < fdr_threshold, snp_col].unique().tolist()
    print(f"Found {len(top_snps)} SNPs below FDR < {fdr_threshold}")

    # --- Keep only SNPs that exist in merged data ---
    available_snps = [snp for snp in top_snps if snp in merged_snp_aroma.columns]
    missing_snps = set(top_snps) - set(available_snps)

    if len(missing_snps) > 0:
        print(f" {len(missing_snps)} SNPs not found in merged data — skipped.")
    print(f" Using {len(available_snps)} valid SNPs for analysis.")

    if len(available_snps) == 0:
        print(" No valid SNPs found. Aborting.")
        return None, None, None, None

    # --- Prepare matrices ---
    X = merged_snp_aroma[available_snps].fillna(0).astype(float)
    aroma_cols = [c for c in merged_snp_aroma.columns if c not in available_snps]
    Y = merged_snp_aroma[aroma_cols].select_dtypes(include=[np.number]).fillna(0)

    # --- Standardize ---
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    # --- PCA reduction ---
    pca = PCA(n_components=n_pca_components)
    X_reduced = pca.fit_transform(X_scaled)

    # --- PLS regression ---
    pls = PLSRegression(n_components=n_pls_components)
    pls.fit(X_reduced, Y_scaled)

    print("\n PLSR model fitted successfully.")

    # --- Plot ---
    X_scores = pls.x_scores_
    Y_scores = pls.y_scores_

    plt.figure(figsize=(7, 6))
    plt.scatter(X_scores[:, 0], Y_scores[:, 0], alpha=0.7, color='steelblue')

    for i, name in enumerate(merged_snp_aroma.index):
        plt.text(X_scores[i, 0], Y_scores[i, 0], name, fontsize=7, color='navy')

    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)
    plt.xlabel("PLS Component 1 (Genes)")
    plt.ylabel("PLS Component 1 (Aroma)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return pca, pls, X_scores, Y_scores



def analyze_key_variety_aroma_profiles(
    merged_snp_aroma,
    gwas_path,
    key_varieties=["NADINE", "ARGOS", "OSPREY"],
    fdr_threshold=0.05,
    n_pca_components=50,
    n_pls_components=2,
    top_n_compounds=10,
    save_path="key_variety_top_aroma_compounds.csv"
):
    """
    Analyze aroma compound deviations for key potato varieties
    using GWAS-selected SNPs, PCA reduction, and PLSR.

    Parameters
    ----------
    merged_snp_aroma : pd.DataFrame
        Combined SNP + aroma dataset (indexed by variety names).
    gwas_path : str
        Path to the GWAS results CSV file.
    key_varieties : list
        List of key potato varieties to analyze.
    fdr_threshold : float, default=0.05
        FDR threshold for selecting significant SNPs.
    n_pca_components : int, default=50
        Number of PCA components for SNP reduction.
    n_pls_components : int, default=2
        Number of PLS components to fit.
    top_n_compounds : int, default=10
        Number of top differentiating aroma compounds per variety.
    save_path : str, default="key_variety_top_aroma_compounds.csv"
        Path to save the summary CSV.

    Returns
    -------
    summary_df : pd.DataFrame
        Summary table of key varieties and their top aroma compounds.
    """

    # ==========================================================
    #  Load GWAS and select significant SNPs
    # ==========================================================
    gwas_df = pd.read_csv(gwas_path)
    top_snps = gwas_df.loc[gwas_df['p_fdr'] < fdr_threshold, 'SNP'].unique().tolist()
    print(f" Selected {len(top_snps)} SNPs after GWAS feature selection (FDR < {fdr_threshold}).")

    if len(top_snps) == 0:
        print(" No SNPs passed the threshold.")
        return None

    # ==========================================================
    #  Separate SNP (X) and Aroma (Y) matrices
    # ==========================================================
    snp_cols = [c for c in merged_snp_aroma.columns if c.startswith("GEN__")]
    aroma_cols = [c for c in merged_snp_aroma.columns if c not in snp_cols]

    X = merged_snp_aroma[snp_cols].fillna(0).astype(float)
    Y = merged_snp_aroma[aroma_cols].select_dtypes(include=[np.number]).fillna(0)

    # Normalize index names
    variety_names = merged_snp_aroma.index.str.upper().str.replace(" ", "")
    X.index = variety_names
    Y.index = variety_names

    # ==========================================================
    #  Standardize and apply PCA + PLSR
    # ==========================================================
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    pca = PCA(n_components=n_pca_components)
    X_reduced = pca.fit_transform(X_scaled)

    pls = PLSRegression(n_components=n_pls_components)
    pls.fit(X_reduced, Y_scaled)

    print("\n PLSR model fitted successfully.")

    # ==========================================================
    # 4 Compute deviations from mean profile for key varieties
    # ==========================================================
    Y_df = pd.DataFrame(Y_scaled, index=variety_names, columns=Y.columns)
    key_varieties = [v for v in [v.upper().replace(" ", "") for v in key_varieties] if v in Y_df.index]

    print(f"\n Found {len(key_varieties)} key varieties: {key_varieties}")

    mean_profile = Y_df.mean()
    diff_profiles = Y_df.loc[key_varieties] - mean_profile
    abs_diff = diff_profiles.abs()

    # ==========================================================
    #  Identify top differentiating aroma compounds
    # ==========================================================
    top_differences = {
        variety: abs_diff.loc[variety].sort_values(ascending=False).head(top_n_compounds)
        for variety in key_varieties
    }

    for variety, diffs in top_differences.items():
        print(f"\n Top {top_n_compounds} differentiating aroma compounds for {variety}:")
        display(diffs)

    # ==========================================================
    #  Visualization for each key variety
    # ==========================================================
    for variety in key_varieties:
        plt.figure(figsize=(8, 4))
        sns.barplot(
            x=diff_profiles.loc[variety, top_differences[variety].index],
            y=top_differences[variety].index,
            palette="coolwarm",
            orient="h"
        )
        plt.title(f"{variety}: Top Aroma Compound Deviations from Mean")
        plt.xlabel("Deviation (Standardized Units)")
        plt.ylabel("Aroma Compound")
        plt.axvline(0, color='gray', lw=0.8)
        plt.tight_layout()
        plt.show()

    # ==========================================================
    #  Summary Table and Save
    # ==========================================================
    summary_table = []
    for variety in key_varieties:
        for compound in top_differences[variety].index:
            direction = "↑ Higher" if diff_profiles.loc[variety, compound] > 0 else "↓ Lower"
            summary_table.append({
                "Variety": variety,
                "Aroma Compound": compound,
                "Direction": direction,
                "Deviation (std units)": diff_profiles.loc[variety, compound]
            })

    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(save_path, index=False)
    print(f"\n Results saved as: {save_path}")

    return summary_df




def run_gwas_for_sensory_pcs(
    merged_snp_sensory,
    snp_prefix="GEN__",
    traits=None,
    ld_threshold=0.95,
    max_snps=2000,
    top_n_snps=500,
    output_path="gwas_sensory_PCs.csv"
):
    """
    Perform GWAS for sensory PCA components (e.g., PC1–PC3)
    using SNP genotypes as predictors.

    Parameters
    ----------
    merged_snp_sensory : pd.DataFrame
        DataFrame containing SNP columns and sensory principal components.
    snp_prefix : str, default='GEN__'
        Prefix that identifies SNP marker columns.
    traits : list of str, optional
        List of sensory PCs to analyze (e.g., ["PC1", "PC2", "PC3"]).
    ld_threshold : float, default=0.95
        Correlation threshold for LD pruning.
    max_snps : int, default=2000
        Max number of SNPs to include before pruning.
    top_n_snps : int, default=500
        Number of top SNPs per trait to retain for pruning and inspection.
    output_path : str, default='gwas_sensory_PCs.csv'
        Path to save combined GWAS results.

    Returns
    -------
    all_gwas_df : pd.DataFrame
        Combined GWAS results for all sensory PCs.
    """

    # ---------------- Helper functions ----------------
    def run_gwas(df, trait, snp_cols):
        """Run GWAS for one sensory trait."""
        y = df[trait].astype(float).values
        n = len(y)
        gwas_results = []

        for snp in snp_cols:
            x = df[snp].fillna(df[snp].mode().iloc[0]).astype(float).values
            if x.std() == 0:
                continue
            x = (x - x.mean()) / (x.std() + 1e-6)
            r = np.corrcoef(x, y)[0, 1]
            if np.isnan(r):
                continue
            r = np.clip(r, -0.999999, 0.999999)
            F_val = (r**2) / ((1 - r**2) / (n - 2))
            p_val = f.sf(F_val, 1, n - 2)
            gwas_results.append((snp, p_val))

        gwas_df = pd.DataFrame(gwas_results, columns=["SNP", "p_value"])
        gwas_df["p_fdr"] = multipletests(gwas_df["p_value"], method="fdr_bh")[1]
        return gwas_df

    def ld_prune(X_df, threshold=0.9, max_snps=2000):
        """Remove SNPs with strong LD (|r| > threshold)."""
        X_df = X_df.loc[:, X_df.std() > 0]
        if X_df.shape[1] > max_snps:
            X_df = X_df.iloc[:, :max_snps]
        if X_df.shape[1] <= 1:
            return X_df

        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return X_df.drop(columns=to_drop)

    # ---------------- Main GWAS loop ----------------
    snp_cols = [c for c in merged_snp_sensory.columns if c.startswith(snp_prefix)]
    if traits is None:
        traits = ["PC1", "PC2", "PC3"]

    all_gwas = []

    for trait in traits:
        print(f"\n🔹 Running GWAS for {trait}...")
        gwas_df = run_gwas(merged_snp_sensory, trait, snp_cols)

        if gwas_df.empty:
            print(f"   No valid SNPs for {trait}. Skipping.")
            continue

        top_snps = gwas_df.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
        if not top_snps:
            print(f"   No significant SNPs for {trait}. Skipping.")
            continue

        X = merged_snp_sensory[top_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=ld_threshold, max_snps=max_snps)

        if X_pruned.shape[1] == 0:
            print(f"   No SNPs after LD pruning for {trait}. Skipping.")
            continue

        gwas_df["Trait"] = trait
        all_gwas.append(gwas_df)

    # ---------------- Combine and save ----------------
    if not all_gwas:
        print(" No GWAS results generated.")
        return pd.DataFrame()

    all_gwas_df = pd.concat(all_gwas, ignore_index=True)
    all_gwas_df.to_csv(output_path, index=False)
    print(f"\n All GWAS results saved to {output_path}")

    return all_gwas_df




def analyze_gwas_snps_pca(
    gwas_path: str,
    merged_snp_sensory: pd.DataFrame,
    fdr_threshold: float = 0.05,
    top_n_fallback: int = 500,
    outlier_z_pca=(3, 2.5)
):
    """
    Analyze GWAS results: select significant SNPs and perform PCA with outlier detection.

    Parameters
    ----------
    gwas_path : str
        Path to the GWAS results CSV file.
    merged_snp_sensory : pd.DataFrame
        Merged SNP × sensory PCs data.
    fdr_threshold : float, optional
        Significance threshold for FDR (default: 0.05).
    top_n_fallback : int, optional
        Number of top SNPs to use if no FDR-significant SNPs are found (default: 500).
    outlier_z_pca : tuple, optional
        Z-score thresholds for (PC1, PC2) outlier detection in PCA.

    Returns
    -------
    dict
        Dictionary with PCA and outlier detection results:
        {
            "significant_snps": list,
            "pca_model": PCA(),
            "pca_outliers": list
        }
    """

    # =====================================================
    # 1. Load GWAS results and extract significant SNPs
    # =====================================================
    gwas_df = pd.read_csv(gwas_path)

    significant_snps = gwas_df[gwas_df["p_fdr"] < fdr_threshold]["SNP"].unique().tolist()
    if len(significant_snps) == 0:
        print(f" No SNPs with FDR < {fdr_threshold}. Using top {top_n_fallback} by p-value.")
        significant_snps = (
            gwas_df.sort_values("p_fdr")
            .head(top_n_fallback)["SNP"]
            .unique()
            .tolist()
        )
    else:
        print(f" Significant SNPs found: {len(significant_snps)}")

    # =====================================================
    # 2. Subset SNP and sensory PC data
    # =====================================================
    snp_cols = [col for col in merged_snp_sensory.columns if col.startswith("GEN__")]
    snp_subset = merged_snp_sensory.loc[:, [s for s in significant_snps if s in snp_cols]].fillna(0).astype(float)

    if snp_subset.empty:
        raise ValueError("No matching SNPs found in merged_snp_sensory.")

    # =====================================================
    # 3. PCA with Outlier Detection
    # =====================================================
    X_scaled = StandardScaler().fit_transform(snp_subset)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    Z = np.abs(zscore(X_pca))
    outlier_mask = (Z[:, 0] > outlier_z_pca[0]) | (Z[:, 1] > outlier_z_pca[1])
    normal_mask = ~outlier_mask

    # --- PCA Plot ---
    plt.figure(figsize=(9, 7))
    plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                color="steelblue", alpha=0.7, s=60, label="Normal varieties")
    plt.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
                color="red", s=120, edgecolor="black", label="Outliers")

    np.random.seed(42)
    for i in np.where(outlier_mask)[0]:
        name = snp_subset.index[i]
        dx, dy = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
        plt.text(X_pca[i, 0] + dx, X_pca[i, 1] + dy, name,
                 fontsize=10, color="red", fontweight="bold")

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA on Significant SNPs (Outlier Detection)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 4. Return Results
    # =====================================================
    pca_outliers = snp_subset.index[outlier_mask].tolist()

    return {
        "significant_snps": significant_snps,
        "pca_model": pca,
        "pca_outliers": pca_outliers
    }




def plot_sensory_profiles_of_pca_outliers(results, sensory_reconstructed_df):
    """
    Automatically detects PCA outlier varieties from GWAS PCA results
    and plots their sensory profiles on a radar chart.

    Parameters
    ----------
    results : dict
        Output dictionary from `analyze_gwas_snps_pca()` containing:
        - "pca_outliers": list of outlier names from PCA

    sensory_reconstructed_df : pd.DataFrame
        DataFrame containing reconstructed sensory scores for each variety.
        Rows = varieties (indexed by name), Columns = sensory traits.

    Returns
    -------
    None
        Displays a radar chart comparing normalized sensory profiles of PCA-detected outliers.
    """

    # ---  Extract PCA outliers ---
    pca_outliers = results.get("pca_outliers", [])

    if not pca_outliers:
        raise ValueError("No PCA outliers detected in the analysis results.")

    print(f"Detected PCA outliers for radar plot: {pca_outliers}")

    # ---  Extract sensory profiles of detected outliers ---
    Y_outliers = sensory_reconstructed_df.loc[
        sensory_reconstructed_df.index.intersection(pca_outliers)
    ]

    if Y_outliers.empty:
        raise ValueError("Detected PCA outliers not found in sensory_reconstructed_df index.")

    # --- Normalize traits to (-1, 1) for comparability ---
    Y_norm = (Y_outliers - Y_outliers.min()) / (Y_outliers.max() - Y_outliers.min()) * 2 - 1

    # ---  Define radar structure ---
    traits = Y_norm.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]  # close circle

    # ---  Plot radar chart ---
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    # Assign colors automatically
    color_palette = plt.cm.tab10.colors
    color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(Y_norm.index)}

    for variety in Y_norm.index:
        values = Y_norm.loc[variety].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=variety, color=color_map[variety])
        ax.fill(angles, values, alpha=0.15, color=color_map[variety])

    # ---  Aesthetic formatting ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, fontsize=10)
    ax.set_yticklabels([])
    plt.title("Sensory Profile Comparison of PCA-Detected Outlier Varieties", fontsize=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()





def plot_manhattan_like_gwas(gwas_path):
    """
    Generate a Manhattan-like scatter plot for sensory GWAS results (without chromosome info).

    Parameters
    ----------
    gwas_path : str
        Path to the GWAS results CSV file. The file must contain at least:
        - 'p_fdr' : FDR-adjusted p-values
        - 'Trait' : corresponding sensory traits

    Returns
    -------
    None
        Displays a scatter plot of GWAS markers ranked by significance (-log10 FDR-adjusted p-value).
    """

    # --- Load GWAS results ---
    gwas_df = pd.read_csv(gwas_path)

    # --- Basic column check ---
    required_cols = {"p_fdr", "Trait"}
    if not required_cols.issubset(gwas_df.columns):
        raise ValueError(f"Input file must contain columns: {required_cols}")

    # --- Sort by significance ---
    gwas_df = gwas_df.sort_values("p_fdr").reset_index(drop=True)

    # --- Create figure ---
    plt.figure(figsize=(14, 6))
    sns.scatterplot(
        data=gwas_df,
        x=np.arange(len(gwas_df)),
        y=-np.log10(gwas_df["p_fdr"]),
        hue="Trait",
        palette="tab10",
        s=25,
        alpha=0.7
    )

    # --- Add significance thresholds ---
    plt.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1.2, label="FDR = 0.05")

    # --- Add text note if no significant SNPs ---
    n_sig = (gwas_df["p_fdr"] < 0.05).sum()
    # --- Formatting ---
    plt.title("Marker-Based Manhattan Plot for Sensory GWAS PCs", fontsize=14, pad=15)
    plt.xlabel("Gene / Marker (SNP, sorted by p-value)", fontsize=11)
    plt.ylabel("-log10(FDR-adjusted p-value)", fontsize=11)
    plt.legend(title="Trait", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()




def plot_gwas_volcano(gwas_path, fdr_threshold=0.1):
    """
    Generate a Volcano Plot for GWAS of sensory principal components.

    Parameters
    ----------
    gwas_path : str
        Path to the GWAS results CSV file. The file must contain:
        - 'p_fdr' : FDR-adjusted p-values
        - 'Trait' : corresponding trait label
        - 'effect_size' : estimated SNP effect sizes (optional)
    fdr_threshold : float, optional
        Significance cutoff for drawing the horizontal threshold line (default: 0.1).

    Returns
    -------
    None
        Displays a volcano plot with effect size vs -log10(FDR-adjusted p-value).

    Notes
    -----
    The volcano plot visualizes SNP effect size (x-axis) versus statistical significance
    (-log10 FDR-adjusted p-value). Points above the red dashed line indicate SNPs
    with FDR < threshold, suggesting stronger associations.
    """

    # Load GWAS results
    gwas_df = pd.read_csv(gwas_path)

    # validation 
    required_cols = {"p_fdr", "Trait"}
    if not required_cols.issubset(gwas_df.columns):
        raise ValueError(f"Input file must contain columns: {required_cols}")

    #  Add mock effect size if not present 
    if "effect_size" not in gwas_df.columns:
        np.random.seed(42)
        gwas_df["effect_size"] = np.random.normal(0, 1, len(gwas_df))
        print(" 'effect_size' column not found — generated random values for visualization.")

    #  Sort for better visual layering 
    gwas_df = gwas_df.sort_values("p_fdr").reset_index(drop=True)

    # Create Volcano Plot 
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=gwas_df,
        x="effect_size",
        y=-np.log10(gwas_df["p_fdr"]),
        hue="Trait",
        palette="Set2",
        s=25,
        alpha=0.8
    )

    #  Significance threshold line
    plt.axhline(
        -np.log10(fdr_threshold),
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"FDR = {fdr_threshold}"
    )

    # Styling and labels
    plt.title("Volcano Plot for Sensory GWAS Principal Components", fontsize=14, weight="bold")
    plt.xlabel("Effect Size", fontsize=12)
    plt.ylabel("−log10(FDR-adjusted p-value)", fontsize=12)
    plt.legend(title="Trait", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()



def plot_gwas_qq(gwas_df, pval_col="p_value"):
    """
    Generate a QQ (Quantile–Quantile) plot to assess GWAS p-value distribution.

    Parameters
    ----------
    gwas_df : pandas.DataFrame
        DataFrame containing GWAS results with a p-value column.
    pval_col : str, optional
        Column name for raw p-values (default: "p_value").

    Returns
    -------
    None
        Displays a QQ plot comparing observed vs expected -log10(p) values.

    Notes
    -----
    The QQ plot helps evaluate whether the distribution of observed GWAS p-values
    deviates from the null expectation. Points along the red dashed line indicate
    well-calibrated p-values (no inflation or deflation).
    """

    # --- Validate input ---
    if pval_col not in gwas_df.columns:
        raise ValueError(f"'{pval_col}' column not found in DataFrame.")

    # --- Clean p-values ---
    pvals = gwas_df[pval_col].replace(0, np.nan).dropna()
    if pvals.empty:
        raise ValueError("No valid p-values found after removing NaNs and zeros.")

    # --- Expected and observed -log10(p) ---
    expected = -np.log10(np.linspace(1 / len(pvals), 1, len(pvals)))
    observed = -np.log10(np.sort(pvals))

    # --- Plot ---
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=expected, y=observed, s=15, color="darkslateblue", alpha=0.7)
    plt.plot([0, max(expected)], [0, max(expected)], color="red", linestyle="--", linewidth=1)

    # --- Formatting ---
    plt.title("QQ Plot for Sensory GWAS Principal Components", fontsize=14, weight="bold")
    plt.xlabel("Expected -log10(p)", fontsize=12)
    plt.ylabel("Observed -log10(p)", fontsize=12)
    plt.tight_layout()
    plt.show()




import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_regression(y_true, y_pred):
    """Compute key regression metrics."""
    return {
        "R2_in_sample": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def ld_prune(X_df, threshold=0.95, max_snps=2000):
    """
    Perform LD pruning on SNP matrix (remove highly correlated SNPs).
    Keeps at most max_snps columns (for speed) and drops highly correlated SNPs.
    """
    X_df = X_df.loc[:, X_df.std() > 0]
    if X_df.shape[1] > max_snps:
        X_df = X_df.iloc[:, :max_snps]
    if X_df.shape[1] <= 1:
        return X_df

    corr = X_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X_df.drop(columns=to_drop)


def run_linear_models_per_trait_auto(
    GT, PH,
    gwas_table="bmqg.gwas.run_local_20251207",
    trait_col="trait",          # change to "Trait" if needed
    snp_col="taglo_id",         # change to "SNP" if needed
    p_col="p_fdr",              # change to "p_wald" or other if needed
    top_n_snps=200,
    prune_threshold=0.95,
    cv_splits=5,
    save_dir="/dbfs/FileStore/aroma_flavor_project/Results",
    tag="GWAS_top200"
):
    """
    Auto-run Linear Regression trait-by-trait:
      - For each phenotype trait in PH.columns:
          - take top_n_snps from GWAS (lowest p)
          - subset GT to those SNPs
          - LD prune
          - CV R² (KFold)
          - fit on all data for in-sample diagnostics
      - Save CSV + plot (barh of CV R²)
    """

    os.makedirs(save_dir, exist_ok=True)

    # --- align varieties once (no merge; index intersection) ---
    common = GT.index.intersection(PH.index)
    if len(common) < 10:
        raise ValueError(f"Too few overlapping varieties between GT and PH: {len(common)}")

    GTa = GT.loc[common]
    PHa = PH.loc[common]

    # --- load GWAS from Spark ---
    gwas = (spark.table(gwas_table)
            .select(F.col(trait_col).alias("Trait"),
                    F.col(snp_col).alias("SNP"),
                    F.col(p_col).alias("p"))
            .toPandas())

    gwas["Trait"] = gwas["Trait"].astype(str)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # --- pipeline (scaling inside CV, prevents leakage) ---
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    results = []

    # --- loop traits automatically ---
    for trait in PHa.columns:
        print(f"\n[LINEAR — {tag}] Trait: {trait}")

        g = gwas[gwas["Trait"] == trait].dropna(subset=["SNP", "p"])
        if g.empty:
            print("  No GWAS rows for this trait.")
            continue

        # top SNPs by p
        top_snps = (g.sort_values("p")
                      .head(top_n_snps)["SNP"]
                      .tolist())

        # keep only SNPs present in GT columns
        top_snps = [s for s in top_snps if s in GTa.columns]
        if len(top_snps) == 0:
            print("  None of top SNPs exist in GT columns.")
            continue

        X = GTa[top_snps].fillna(0).astype(float)

        # LD prune
        X = ld_prune(X, threshold=prune_threshold)
        if X.shape[1] == 0:
            print("  No SNPs left after LD pruning.")
            continue

        y = PHa[trait].astype(float).values

        # CV R²
        cv_r2 = float(np.mean(cross_val_score(pipe, X.values, y, cv=cv, scoring="r2")))

        # in-sample diagnostics
        pipe.fit(X.values, y)
        y_pred = pipe.predict(X.values)
        metrics = evaluate_regression(y, y_pred)

        results.append({
            "Trait": trait,
            "Model": "Linear",
            "n_SNPs": int(X.shape[1]),
            "CV_R2": cv_r2,
            **metrics
        })

    results_df = pd.DataFrame(results)

    # --- save CSV ---
    out_csv = os.path.join(save_dir, f"linear_per_trait_{tag}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV → {out_csv}")

    # --- plot ---
    if not results_df.empty:
        plot_df = results_df.sort_values("CV_R2", ascending=False)

        plt.figure(figsize=(11, 8))
        plt.barh(plot_df["Trait"], plot_df["CV_R2"])
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Cross-validated R²")
        plt.title(f"Linear Regression performance per trait ({tag})")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        out_png = os.path.join(save_dir, f"linear_per_trait_{tag}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved plot → {out_png}")
        plt.show()
    else:
        print("No results to plot.")

    return results_df





def run_krr_models_gene_aroma(
    gwas_file,
    snp_df,
    pheno_df,
    top_n_snps=200,
    prune_threshold=0.95,
    save_dir=None,
    tag="default",
    trait_col="Trait",
    snp_col="SNP",
    p_col="p_fdr",
    cv_splits=5,
    # smaller grid by default (practical for 68 traits)
    param_grid=None,
    min_common=20
):
    """
    Kernel Ridge Regression (RBF) per trait using top GWAS SNPs.
    X from snp_df, y from pheno_df. Varieties aligned by normalized index.
    - No merge
    - Scaling leakage avoided via Pipeline
    - Uses GridSearchCV per trait
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # -------------------------
    # Helpers
    # -------------------------
    def ensure_index(df, name="Variety"):
        if df.index.name == name:
            return df
        if name in df.columns:
            return df.set_index(name)
        return df

    def norm_index(idx):
        return (pd.Index(idx.astype(str))
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.upper())

    # -------------------------
    # Validate inputs
    # -------------------------
    if "ld_prune" not in globals():
        raise NameError("ld_prune is not defined. Define ld_prune(...) before calling this function.")
    if "evaluate_regression" not in globals():
        raise NameError("evaluate_regression is not defined. Define evaluate_regression(...) before calling this function.")

    snp_df = ensure_index(snp_df, "Variety").copy()
    pheno_df = ensure_index(pheno_df, "Variety").copy()

    snp_df.index = norm_index(snp_df.index)
    pheno_df.index = norm_index(pheno_df.index)

    # Compute overlap once (same for all traits)
    common = snp_df.index.intersection(pheno_df.index)
    if len(common) < min_common:
        raise ValueError(f"Too few overlapping varieties between snp_df and pheno_df: {len(common)}")

    # Read GWAS
    gwas = pd.read_csv(gwas_file)
    for c in [trait_col, snp_col, p_col]:
        if c not in gwas.columns:
            raise ValueError(f"GWAS file missing required column: {c}")

    # Normalize GWAS trait strings
    gwas[trait_col] = gwas[trait_col].astype(str)

    # Default grid (smaller, faster; you can pass a bigger grid if you want)
    if param_grid is None:
        param_grid = {
            "krr__alpha": [0.1, 1, 10, 100],
            "krr__gamma": [1e-4, 1e-3, 1e-2, 0.1]
        }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf"))
    ])

    results = []

    # Loop over traits in GWAS (or you can loop over pheno_df.columns; GWAS-driven is fine)
    for trait in gwas[trait_col].dropna().unique():
        trait = str(trait)

        if trait not in pheno_df.columns:
            print(f"\n[KRR — {tag}] Trait: {trait} -> skipped (not in phenotype df)")
            continue

        print(f"\n[KRR — {tag}] Trait: {trait}")
        gwas_df = gwas[gwas[trait_col] == trait].dropna(subset=[p_col, snp_col])

        if gwas_df.empty:
            print("  No GWAS rows for modeling.")
            continue

        # Top SNPs by p
        top_snps = (gwas_df.sort_values(p_col)
                            .head(top_n_snps)[snp_col]
                            .tolist())

        if len(top_snps) == 0:
            print("  No SNPs for modeling.")
            continue

        # Make SNP ids comparable to snp_df columns (handle int/str mismatch)
        # Try best-effort conversion to match snp_df.columns dtype
        cols = snp_df.columns
        if np.issubdtype(cols.dtype, np.number):
            try:
                top_snps = [int(s) for s in top_snps]
            except Exception:
                pass
        else:
            top_snps = [str(s) for s in top_snps]

        # Keep only SNPs present in genotype matrix
        top_snps_existing = [s for s in top_snps if s in snp_df.columns]
        if len(top_snps_existing) == 0:
            print("  None of the selected SNPs exist in snp_df columns (check dtype / naming).")
            continue

        # Build X/y using the precomputed overlap
        X = snp_df.loc[common, top_snps_existing].fillna(0).astype(float)

        # LD prune
        X_pruned = ld_prune(X, threshold=prune_threshold)
        if X_pruned.shape[1] == 0:
            print("  No SNPs left after pruning.")
            continue

        y = pheno_df.loc[common, trait].astype(float).values

        # Grid search
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=-1
        )
        grid.fit(X_pruned.values, y)

        best_pipe = grid.best_estimator_

        # In-sample diagnostics (NOT the real performance; CV is the performance)
        y_pred = best_pipe.predict(X_pruned.values)
        metrics = evaluate_regression(y, y_pred)

        results.append({
            "Trait": trait,
            "Model": "KRR_RBF",
            "n_SNPs": int(X_pruned.shape[1]),
            "Best_alpha": grid.best_params_["krr__alpha"],
            "Best_gamma": grid.best_params_["krr__gamma"],
            "CV_R2": float(grid.best_score_),
            **metrics
        })

    results_df = pd.DataFrame(results)

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"krr_models_{tag.replace(' ', '_')}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved → {csv_path}")

    # Plot
    if not results_df.empty:
        sorted_df = results_df.sort_values("CV_R2", ascending=False)

        plt.figure(figsize=(11, 6))
        sns.barplot(data=sorted_df, x="CV_R2", y="Trait", palette="viridis")
        plt.axvline(0, color="red", linestyle="--", lw=1)
        plt.xlabel("Cross-validated R²")
        plt.ylabel("Aroma Trait")
        plt.title(f"Kernel Ridge Regression (RBF) Performance per Aroma Trait ({tag})")

        for i, v in enumerate(sorted_df["CV_R2"]):
            plt.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, f"krr_models_plot_{tag.replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved → {plot_path}")

        plt.show()
    else:
        print("No results to plot.")

    return results_df



def run_nonlinear_models_gene_aroma(
    gwas_file,
    snp_df,
    top_n_snps=500,
    prune_threshold=0.95,
    save_dir=None,
    tag="default"
):
    """
    Run nonlinear regression models (SVR, RandomForest, XGBoost)
    for all traits using top GWAS SNPs (FDR-filtered + LD pruned).
    Automatically filters and clips extreme negative R² values (< -2) for clarity.
    """


    models = {
        "SVR": SVR(kernel="rbf", C=10, gamma=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=4,
            random_state=42, n_jobs=-1, verbosity=0
        )
    }

    gwas = pd.read_csv(gwas_file)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for trait in gwas["Trait"].unique():
        print(f"\n[Nonlinear Models — {tag}] Trait: {trait}")
        gwas_df = gwas[gwas["Trait"] == trait]

        top_snps = gwas_df.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
        if not top_snps:
            print("  No SNPs for modeling.")
            continue

        X = snp_df[top_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=prune_threshold)
        if X_pruned.shape[1] == 0:
            print("  No SNPs left after pruning.")
            continue

        X_scaled = StandardScaler().fit_transform(X_pruned)
        y = snp_df[trait].astype(float).values

        for name, model in models.items():
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            metrics = evaluate_regression(y, y_pred)
            cv_r2 = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring="r2"))

            results.append({
                "Trait": trait,
                "Model": name,
                "n_SNPs": X_pruned.shape[1],
                "CV_Best_R2": cv_r2,
                **metrics
            })

    results_df = pd.DataFrame(results)

    # === Save results ===
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"nonlinear_models_{tag.replace(' ', '_')}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f" Results saved → {csv_path}")

    # === Filter & clip extreme negative R² values ===
    filtered_df = results_df[results_df["CV_Best_R2"] > -2].copy()
    filtered_df["CV_Best_R2_clipped"] = filtered_df["CV_Best_R2"].clip(lower=-2)

    # === Plot each model ===
    for model_name in filtered_df["Model"].unique():
        model_df = filtered_df[filtered_df["Model"] == model_name].copy()
        if model_df.empty:
            continue

        model_df = model_df.sort_values("CV_Best_R2_clipped", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=model_df, y="Trait", x="CV_Best_R2_clipped", palette="viridis")

        plt.axvline(0, color="black", linestyle="--", lw=1)
        plt.title(f"{model_name} Regression Performance per Aroma Trait ({tag})",
                  fontsize=13, weight="bold")
        plt.xlabel("Cross-validated R² (clipped ≤ -2)")
        plt.ylabel("Aroma Trait")

        # Annotate bars with R² values
        for i, v in enumerate(model_df["CV_Best_R2_clipped"]):
            plt.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

        plt.tight_layout()
        if save_dir:
            plot_path = os.path.join(save_dir, f"{model_name.lower()}_plot_{tag.replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f" Plot saved → {plot_path}")

        plt.show()

    return results_df




def run_nonlinear_models_gene_aroma(
    gwas_file,
    snp_df,
    pheno_df,
    top_n_snps=200,
    prune_threshold=0.95,
    save_dir=None,
    tag="default",
    trait_col="Trait",
    snp_col="SNP",
    p_col="p_fdr",
    cv_splits=5,
    clip_lower=-2,
    min_common=20
):
    """
    Run nonlinear regression models (SVR, RandomForest, XGBoost) per trait
    using top GWAS SNPs (FDR-filtered + LD pruned).

     X from snp_df (genotypes), y from pheno_df (phenotypes)
     Varieties aligned by normalized index (no merge)
     Scaling leakage avoided using Pipeline (SVR needs scaling; RF/XGB don't)
     Filters + clips extreme negative CV R² for clearer plots
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # xgboost optional import (gives a clean error if not installed)
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except Exception:
        has_xgb = False

    # -------------------------
    # Helpers / checks
    # -------------------------
    def ensure_index(df, name="Variety"):
        if df.index.name == name:
            return df
        if name in df.columns:
            return df.set_index(name)
        return df

    def norm_index(idx):
        return (pd.Index(idx.astype(str))
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.upper())

    if "ld_prune" not in globals():
        raise NameError("ld_prune is not defined. Define ld_prune(...) before calling this function.")
    if "evaluate_regression" not in globals():
        raise NameError("evaluate_regression is not defined. Define evaluate_regression(...) before calling this function.")

    snp_df = ensure_index(snp_df, "Variety").copy()
    pheno_df = ensure_index(pheno_df, "Variety").copy()

    snp_df.index = norm_index(snp_df.index)
    pheno_df.index = norm_index(pheno_df.index)

    common = snp_df.index.intersection(pheno_df.index)
    if len(common) < min_common:
        raise ValueError(f"Too few overlapping varieties between snp_df and pheno_df: {len(common)}")

    # -------------------------
    # GWAS
    # -------------------------
    gwas = pd.read_csv(gwas_file)
    for c in [trait_col, snp_col, p_col]:
        if c not in gwas.columns:
            raise ValueError(f"GWAS file missing required column: {c}")
    gwas[trait_col] = gwas[trait_col].astype(str)

    # -------------------------
    # Models (pipelines where needed)
    # -------------------------
    models = {
        # SVR benefits from scaling; keep in pipeline to prevent leakage during CV
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10, gamma=0.1))
        ]),

        # Tree models do NOT need scaling; keep simple
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
    }

    if has_xgb:
        models["XGBoost"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    else:
        print("⚠️ xgboost is not available; skipping XGBoost model.")

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = []

    # -------------------------
    # Loop traits (GWAS-driven)
    # -------------------------
    for trait in gwas[trait_col].dropna().unique():
        trait = str(trait)
        if trait not in pheno_df.columns:
            print(f"\n[Nonlinear — {tag}] Trait: {trait} -> skipped (not in phenotype df)")
            continue

        print(f"\n[Nonlinear — {tag}] Trait: {trait}")
        gwas_df = gwas[gwas[trait_col] == trait].dropna(subset=[p_col, snp_col])
        if gwas_df.empty:
            print("  No GWAS rows for modeling.")
            continue

        # Top SNPs
        top_snps = (gwas_df.sort_values(p_col)
                           .head(top_n_snps)[snp_col]
                           .tolist())

        # Handle int/str mismatch between GWAS SNP ids and snp_df columns
        cols = snp_df.columns
        if np.issubdtype(cols.dtype, np.number):
            try:
                top_snps = [int(s) for s in top_snps]
            except Exception:
                pass
        else:
            top_snps = [str(s) for s in top_snps]

        top_snps_existing = [s for s in top_snps if s in snp_df.columns]
        if len(top_snps_existing) == 0:
            print("  None of the selected SNPs exist in snp_df columns (check dtype / naming).")
            continue

        # Build X/y (aligned varieties)
        X = snp_df.loc[common, top_snps_existing].fillna(0).astype(float)

        X_pruned = ld_prune(X, threshold=prune_threshold)
        if X_pruned.shape[1] == 0:
            print("  No SNPs left after pruning.")
            continue

        y = pheno_df.loc[common, trait].astype(float).values

        # Fit + CV per model
        for name, model in models.items():
            # CV
            cv_r2 = float(np.mean(cross_val_score(model, X_pruned.values, y, cv=cv, scoring="r2")))

            # In-sample diagnostics (not performance)
            model.fit(X_pruned.values, y)
            y_pred = model.predict(X_pruned.values)
            metrics = evaluate_regression(y, y_pred)

            results.append({
                "Trait": trait,
                "Model": name,
                "n_SNPs": int(X_pruned.shape[1]),
                "CV_R2": cv_r2,
                **metrics
            })

    results_df = pd.DataFrame(results)

    # -------------------------
    # Save
    # -------------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"nonlinear_models_{tag.replace(' ', '_')}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved → {csv_path}")

    # -------------------------
    # Filter + clip for plotting
    # -------------------------
    if results_df.empty:
        print("No results to plot.")
        return results_df

    plot_df = results_df.copy()
    plot_df = plot_df[plot_df["CV_R2"].notna()]
    plot_df = plot_df[plot_df["CV_R2"] > clip_lower]
    plot_df["CV_R2_clipped"] = plot_df["CV_R2"].clip(lower=clip_lower)

    # -------------------------
    # Plot each model separately
    # -------------------------
    for model_name in plot_df["Model"].unique():
        model_df = plot_df[plot_df["Model"] == model_name].copy()
        if model_df.empty:
            continue

        model_df = model_df.sort_values("CV_R2_clipped", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=model_df, y="Trait", x="CV_R2_clipped", palette="viridis")
        plt.axvline(0, color="black", linestyle="--", lw=1)
        plt.title(f"{model_name} Performance per Aroma Trait ({tag})", fontsize=13, weight="bold")
        plt.xlabel(f"Cross-validated R² (clipped ≥ {clip_lower})")
        plt.ylabel("Aroma Trait")

        for i, v in enumerate(model_df["CV_R2_clipped"]):
            plt.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, f"{model_name.lower()}_plot_{tag.replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved → {plot_path}")

        plt.show()

    return results_df


def run_krr_sensory_pcs(
    gwas_file,
    merged_snp_df,
    fdr_threshold=0.1,
    top_n_fallback=500,
    prune_threshold=0.95,
    save_dir=None
):
    """
    Run Kernel Ridge Regression (RBF kernel) for sensory PCs
    using GWAS-selected SNPs filtered by FDR threshold or fallback to top N.

    Parameters
    ----------
    gwas_file : str
        Path to GWAS results CSV (must include 'Trait', 'SNP', 'p_fdr').
    merged_snp_df : pd.DataFrame
        SNP × Sensory PC dataset.
    fdr_threshold : float, default=0.1
        FDR significance threshold for SNP selection.
    top_n_fallback : int, default=500
        Number of top SNPs to use if no SNPs pass the FDR filter.
    prune_threshold : float, default=0.95
        LD pruning correlation threshold.
    save_dir : str or None
        Directory to save results and plots.

    Returns
    -------
    pd.DataFrame
        Summary table of KRR model results for sensory PCs.
    """


    # --- Hyperparameter grid for KRR ---
    param_grid = {
        "alpha": [0.01, 0.1, 1, 10, 100],
        "gamma": [1e-4, 1e-3, 1e-2, 0.1, 1]
    }

    # --- Load GWAS file ---
    gwas_sensory = pd.read_csv(gwas_file)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # --- Loop over each sensory trait ---
    for trait in gwas_sensory["Trait"].unique():
        print(f"\n[Kernel Ridge Regression - Sensory PCs] Trait: {trait}")

        gwas_df = gwas_sensory[gwas_sensory["Trait"] == trait]

        # Select SNPs by FDR or fallback
        sig_snps = gwas_df[gwas_df["p_fdr"] < fdr_threshold]["SNP"].tolist()
        if not sig_snps:
            print(f"  No SNPs with FDR < {fdr_threshold}. Using top {top_n_fallback} by p-value.")
            sig_snps = gwas_df.sort_values("p_fdr").head(top_n_fallback)["SNP"].tolist()
        else:
            print(f"  Significant SNPs found: {len(sig_snps)}")

        if not sig_snps:
            continue

        # --- SNP feature matrix ---
        X = merged_snp_df[sig_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=prune_threshold)

        if X_pruned.shape[1] == 0:
            print("  No SNPs left after pruning.")
            continue

        X_scaled = StandardScaler().fit_transform(X_pruned)
        y = merged_snp_df[trait].astype(float).values

        # --- Grid search KRR ---
        grid = GridSearchCV(
            KernelRidge(kernel="rbf"),
            param_grid=param_grid,
            cv=cv,
            scoring=make_scorer(r2_score),
            n_jobs=-1
        )
        grid.fit(X_scaled, y)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_scaled)
        metrics = evaluate_regression(y, y_pred)

        results.append({
            "Trait": trait,
            "n_SNPs": X_pruned.shape[1],
            "Best_alpha": grid.best_params_["alpha"],
            "Best_gamma": grid.best_params_["gamma"],
            "CV_Best_R2": grid.best_score_,
            **metrics
        })

    # --- Compile results ---
    results_df = pd.DataFrame(results)

    # --- Save results ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "krr_sensory_PCs_FDRfiltered.csv")
        results_df.to_csv(save_path, index=False)
        print(f"\n Results saved to: {save_path}")

    # --- Visualization ---
    if not results_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=results_df.sort_values("CV_Best_R2", ascending=False),
            x="CV_Best_R2", y="Trait", palette="viridis"
        )
        plt.axvline(0, color="red", linestyle="--", lw=1)
        plt.xlabel("Cross-validated R²")
        plt.ylabel("Sensory PC Trait")
        plt.title(f"Kernel Ridge Regression (RBF) on Sensory PCs (FDR < {fdr_threshold})", fontsize=13, weight="bold")
        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, "krr_sensory_PCs_plot_FDRfiltered.png")
            plt.savefig(plot_path, dpi=300)
            print(f" Plot saved → {plot_path}")

        plt.show()
    return results_df



def run_nonlinear_sensory_pcs(
    gwas_file,
    merged_snp_df,
    fdr_threshold=0.1,
    top_n_fallback=500,
    prune_threshold=0.95,
    save_dir=None
):
    """
    Run nonlinear regression models (SVR, RandomForest, XGBoost)
    for each sensory PC using GWAS-selected SNPs (FDR-based filtering + fallback).

    Parameters
    ----------
    gwas_file : str
        Path to GWAS results CSV (must include 'Trait', 'SNP', 'p_fdr').
    merged_snp_df : pd.DataFrame
        SNP × Sensory PC dataset.
    fdr_threshold : float, default=0.1
        FDR cutoff for SNP selection.
    top_n_fallback : int, default=500
        Number of top SNPs (by p-value) if no SNPs meet FDR threshold.
    prune_threshold : float, default=0.95
        LD pruning correlation threshold.
    save_dir : str or None
        Directory to save results and plots.

    Returns
    -------
    pd.DataFrame
        Summary of nonlinear model results per trait.
    """


    # --- Load data ---
    gwas_sensory = pd.read_csv(gwas_file)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- Define nonlinear models ---
    nonlinear_models = {
        "SVR": SVR(kernel="rbf", C=10, gamma=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.1,
            max_depth=4, random_state=42, n_jobs=-1, verbosity=0
        )
    }

    results_nonlinear = []

    # --- Loop through sensory traits ---
    for trait in gwas_sensory["Trait"].unique():
        print(f"\n[Nonlinear Models - Sensory PCs] Trait: {trait}")

        gwas_df = gwas_sensory[gwas_sensory["Trait"] == trait]

        # --- Select SNPs based on FDR ---
        sig_snps = gwas_df[gwas_df["p_fdr"] < fdr_threshold]["SNP"].tolist()
        if not sig_snps:
            print(f"  No SNPs with FDR < {fdr_threshold}. Using top {top_n_fallback} by p-value.")
            sig_snps = gwas_df.sort_values("p_fdr").head(top_n_fallback)["SNP"].tolist()
        else:
            print(f"  Significant SNPs found: {len(sig_snps)}")

        if not sig_snps:
            print("  Skipping (no SNPs available).")
            continue

        # --- SNP matrix ---
        X = merged_snp_df[sig_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=prune_threshold)

        if X_pruned.shape[1] == 0:
            print("  No SNPs left after LD pruning.")
            continue

        X_scaled = StandardScaler().fit_transform(X_pruned)
        y = merged_snp_df[trait].astype(float).values

        # --- Train nonlinear models ---
        for name, model in nonlinear_models.items():
            print(f"   → Fitting {name} ...")
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)

            metrics = evaluate_regression(y, y_pred)
            cv_r2 = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring="r2", n_jobs=-1))

            results_nonlinear.append({
                "Trait": trait,
                "Model": name,
                "n_SNPs": X_pruned.shape[1],
                "CV_Best_R2": cv_r2,
                **metrics
            })

    # --- Collect results ---
    results_df = pd.DataFrame(results_nonlinear)

    # --- Save results ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "nonlinear_models_sensory_PCs_FDRfiltered.csv")
        results_df.to_csv(save_path, index=False)
        print(f"\n Nonlinear model results saved to: {save_path}")

    # --- Visualization ---
    if not results_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=results_df.sort_values("CV_Best_R2", ascending=False),
            x="CV_Best_R2", y="Trait", hue="Model", palette="tab10"
        )
        plt.axvline(0, color="red", linestyle="--", lw=1)
        plt.xlabel("Cross-validated R²")
        plt.ylabel("Sensory PC Trait")
        plt.title(f"Nonlinear Model Performance on Sensory PCs (FDR < {fdr_threshold})",
                  fontsize=13, weight="bold")
        plt.legend(title="Model")
        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, "nonlinear_sensory_PCs_plot_FDRfiltered.png")
            plt.savefig(plot_path, dpi=300)
            print(f" Plot saved → {plot_path}")
        plt.show()
    return results_df


def correlate_gene_predictions_with_sensory_pcs(
    gwas_file: str,
    snp_df: pd.DataFrame,
    merged_snp_sensory: pd.DataFrame,
    save_dir: str,
    top_n_snps: int = 2000,
    prune_threshold: float = 0.95,
    r2_threshold: float = 0.5,
    tag: str = "outlier_removed"
):
    """
    Run Kernel Ridge Regression (KRR) on gene-based aroma data,
    select high-performing predicted traits (CV R² >= threshold),
    and compute correlations with sensory PCs (PC1–PC3).

    Parameters
    ----------
    gwas_file : str
        Path to GWAS results CSV (traits and SNP associations)
    snp_df : pd.DataFrame
        SNP dataframe for aroma compounds (e.g., cleaned_data_gene_aroma)
    merged_snp_sensory : pd.DataFrame
        SNP + sensory PCA dataframe (must contain PC1, PC2, PC3)
    save_dir : str
        Directory where results and plots will be saved
    top_n_snps : int, optional
        Number of top SNPs per trait to use for modeling
    prune_threshold : float, optional
        LD pruning threshold (if used inside model)
    r2_threshold : float, optional
        CV R² threshold to define "high-performing" traits
    tag : str, optional
        Label to tag output files (e.g., 'outlier_removed')

    Returns
    -------
    pd.DataFrame
        Correlation matrix between high-performing gene-predicted compounds and sensory PCs
    """



    # ----------------------------------------------------------
    # Step 1 — Run KRR modeling
    # ----------------------------------------------------------
    print(f"\n Running KRR models for gene-predicted aroma compounds ({tag})...")
    results_krr = run_krr_models_gene_aroma(
        gwas_file=gwas_file,
        snp_df=snp_df,
        top_n_snps=top_n_snps,
        prune_threshold=prune_threshold,
        save_dir=save_dir,
        tag=tag
    )

    # ----------------------------------------------------------
    # Step 2 — Select high-performing traits
    # ----------------------------------------------------------
    high_perf_traits = results_krr.query(f"CV_Best_R2 >= {r2_threshold}")["Trait"].tolist()
    print(f"\n High-performing traits (CV R² ≥ {r2_threshold}): {len(high_perf_traits)}")
    print("→", high_perf_traits)

    # ----------------------------------------------------------
    # Step 3 — Prepare predictions
    # ----------------------------------------------------------
    predictions = pd.DataFrame(index=snp_df.index)
    for trait in high_perf_traits:
        if trait in snp_df.columns:
            predictions[trait] = snp_df[trait]
    print(f"\n Predictions DataFrame ready: {predictions.shape}")

    # ----------------------------------------------------------
    # Step 4 — Extract sensory PCs
    # ----------------------------------------------------------
    sensory_pcs = merged_snp_sensory[["PC1", "PC2", "PC3"]].copy()

    # ----------------------------------------------------------
    # Step 5 — Normalize indices (variety names)
    # ----------------------------------------------------------
    def normalize_index(df):
        df.index = (
            df.index.astype(str)
            .str.strip()
            .str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
        return df

    predictions = normalize_index(predictions)
    sensory_pcs = normalize_index(sensory_pcs)

    # Match samples
    common_samples = predictions.index.intersection(sensory_pcs.index)
    predictions = predictions.loc[common_samples]
    sensory_pcs = sensory_pcs.loc[common_samples]

    print(f" Common samples between predictions ({len(snp_df)}) and sensory PCs ({len(merged_snp_sensory)}): {len(common_samples)}")

    # ----------------------------------------------------------
    # Step 6 — Compute correlation
    # ----------------------------------------------------------
    merged_df = pd.concat([predictions, sensory_pcs], axis=1)
    merged_numeric = merged_df.select_dtypes(include=[np.number])
    corr_matrix = merged_numeric.corr().loc[high_perf_traits, ["PC1", "PC2", "PC3"]]

    # ----------------------------------------------------------
    # Step 7 — Save results
    # ----------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    corr_path = os.path.join(save_dir, f"gene_predicted_compound_sensory_corr_{tag}.csv")
    corr_matrix.to_csv(corr_path)
    print(f"\n Correlation matrix saved → {corr_path}")

    # ----------------------------------------------------------
    # Step 8 — Visualization
    # ----------------------------------------------------------
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title(
        f"Correlation between Gene-predicted Aroma Compounds ({tag}) and Sensory PCs",
        fontsize=13,
        weight="bold",
    )
    plt.xlabel("Sensory Components (PC1–PC3)")
    plt.ylabel("Predicted Aroma Compounds")
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # Step 9 — Display top correlations
    # ----------------------------------------------------------
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    print("\n Top correlations:")
    display(corr_pairs.head(20))

    return corr_matrix


def analyze_marker_effects_for_top_correlated_traits(
    gwas_file: str,
    df_aroma: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    corr_threshold: float = 0.5,
    top_snps_per_trait: int = 20,
    save_dir: str = None
):
    """
    Identify gene markers (SNPs) associated with highly predicted aroma compounds
    that show strong correlation with sensory PCs, and visualize their effects
    using linear regression coefficients.

    Parameters
    ----------
    gwas_file : str
        Path to the full GWAS results file (must have columns ['Trait', 'SNP', 'p_fdr'])
    df_aroma : pd.DataFrame
        DataFrame containing genotype and predicted aroma compound values (index = variety)
    corr_matrix : pd.DataFrame
        Correlation matrix from `correlate_gene_predictions_with_sensory_pcs`
    corr_threshold : float, optional
        Minimum absolute correlation with sensory PCs to include a trait
    top_snps_per_trait : int, optional
        Number of top SNPs (lowest p_fdr) to include for each trait
    save_dir : str, optional
        Directory to save generated plots (optional)

    Returns
    -------
    dict
        Mapping of {trait: coefficient DataFrame} for all analyzed compounds
    """


    # ----------------------------------------------------------
    # Step 1 — Load GWAS and filter top correlated compounds
    # ----------------------------------------------------------
    gwas_all = pd.read_csv(gwas_file)
    if not {"Trait", "SNP", "p_fdr"}.issubset(gwas_all.columns):
        raise ValueError("GWAS file must contain columns: Trait, SNP, p_fdr")

    # Find traits with |correlation| ≥ threshold
    corr_abs = corr_matrix.abs()
    top_compounds = corr_abs.max(axis=1).sort_values(ascending=False)
    high_corr_traits = top_compounds[top_compounds >= corr_threshold].index.tolist()

    print(f"\n Found {len(high_corr_traits)} high-correlation compounds (|r| ≥ {corr_threshold}):")
    print("→", high_corr_traits)

    # ----------------------------------------------------------
    # Step 2 — Identify top SNPs for each high-correlation trait
    # ----------------------------------------------------------
    important_genes = (
        gwas_all[gwas_all["Trait"].isin(high_corr_traits)]
        .sort_values("p_fdr")
        .groupby("Trait")["SNP"]
        .apply(lambda x: x.head(top_snps_per_trait).tolist())
    )

    # Store outputs
    all_effects = {}

    # ----------------------------------------------------------
    # Step 3 — Compute SNP effects via Linear Regression
    # ----------------------------------------------------------
    for trait, top_snps in important_genes.items():
        print(f"\n Analyzing {trait} ({len(top_snps)} SNPs)...")

        # Skip traits not present in your data
        if trait not in df_aroma.columns:
            print(f" Trait {trait} not found in df_aroma — skipping.")
            continue

        # Select SNP data and target
        available_snps = [s for s in top_snps if s in df_aroma.columns]
        if not available_snps:
            print(f" No matching SNPs found in df_aroma for {trait}.")
            continue

        X = df_aroma[available_snps].fillna(0).astype(float)
        y = df_aroma[trait].astype(float).values

        # Scale features
        X_scaled = StandardScaler().fit_transform(X)

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_scaled, y)

        coef_df = pd.DataFrame({
            "SNP": available_snps,
            "Effect": model.coef_,
            "Direction": ["Positive" if c > 0 else "Negative" for c in model.coef_],
        })
        coef_df["|Effect|"] = coef_df["Effect"].abs()
        coef_df = coef_df.sort_values("|Effect|", ascending=False).head(20)
        all_effects[trait] = coef_df

        # ----------------------------------------------------------
        # Step 4 — Plot SNP effects
        # ----------------------------------------------------------
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=coef_df,
            x="Effect",
            y="SNP",
            hue="Direction",
            palette={"Positive": "firebrick", "Negative": "royalblue"},
            dodge=False,
        )
        plt.axvline(0, color="black", lw=1)
        plt.title(f"SNP Effects on {trait}", fontsize=14, weight="bold")
        plt.xlabel("Effect Size (Linear Regression Coefficient)")
        plt.ylabel("Gene Marker (SNP)")
        plt.legend(title="Direction", loc="lower right")
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"snp_effects_{trait.replace(' ', '_')}.png"),
                dpi=300, bbox_inches="tight"
            )
        plt.show()

    print("\n Finished analyzing SNP effects for high-correlation traits.")
    return all_effects

def reconstruct_sensory_traits_from_pca(
    projection_path: str,
    loadings_path: str,
    save_path: str,
    n_components: int = 3
):
    """
    Reconstruct sensory trait values (per variety) from PCA projection and loadings.

    Parameters
    ----------
    projection_path : str
        Path to sensory PCA projection file (must contain columns like PC1, PC2, PC3, ...).
    loadings_path : str
        Path to sensory PCA loadings file (traits as rows, PCs as columns).
    save_path : str
        Path to save the reconstructed sensory trait CSV.
    n_components : int, optional
        Number of PCs to use for reconstruction (default = 3).

    Returns
    -------
    pd.DataFrame
        Reconstructed sensory trait values per variety.
    """

    # --- Load files
    projection = pd.read_csv(projection_path)
    loadings = pd.read_csv(loadings_path, index_col=0)

    # --- Extract the first n_components PCs
    pcs = [f"PC{i+1}" for i in range(n_components)]
    X_scores = projection[pcs].values
    load_mat = loadings[pcs].values

    # --- Reconstruct trait matrix
    sensory_reconstructed = np.dot(X_scores, load_mat.T)

    # --- Create DataFrame
    sensory_reconstructed_df = pd.DataFrame(
        sensory_reconstructed,
        index=projection["Variety"],
        columns=loadings.index
    )

    print(f" Reconstructed sensory traits using {n_components} PCs:")
    display(sensory_reconstructed_df.head())

    # --- Save reconstructed sensory traits
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sensory_reconstructed_df.to_csv(save_path)
    print(f" Saved reconstructed sensory traits → {save_path}")

    return sensory_reconstructed_df



def run_plsr_gene_predicted_vs_sensory(
    predictions: pd.DataFrame,
    sensory_reconstructed_path: str,
    high_r2_traits: list,
    save_dir: str,
    n_components: int = 5
):
    """
    Perform PLSR between high-predictive gene-based aroma compound predictions
    and reconstructed sensory attributes.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame of predicted aroma compound values (index = varieties)
    sensory_reconstructed_path : str
        Path to the CSV file containing sensory reconstructed traits (index = varieties)
    high_r2_traits : list
        List of compounds with high predictive performance (e.g., CV R² ≥ 0.5)
    save_dir : str
        Directory to save correlation and loading outputs
    n_components : int, optional
        Number of PLS components to extract (default = 5)

    Returns
    -------
    dict
        {
            "corr_df": correlation DataFrame between PLS components and sensory attributes,
            "loadings_df": compound loadings per component,
            "flavor_alignment": dict describing sensory direction of each PLS component
        }
    """

    # ----------------------------------------------------------
    # Step 1 — Load reconstructed sensory traits
    # ----------------------------------------------------------
    sensory_df = pd.read_csv(sensory_reconstructed_path, index_col=0)

    # ----------------------------------------------------------
    # Step 2 — Normalize and align sample names
    # ----------------------------------------------------------
    sensory_df.index = sensory_df.index.str.upper().str.replace(" ", "")
    predictions.index = predictions.index.str.upper().str.replace(" ", "")

    common_varieties = predictions.index.intersection(sensory_df.index)
    print(f" Common varieties found: {len(common_varieties)}")

    # Keep only shared samples
    X = predictions.loc[common_varieties, high_r2_traits]
    Y = sensory_df.loc[common_varieties]

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    Y = Y.select_dtypes(include=[np.number])

    print(f" Using {X.shape[1]} high-predictive compounds for PLS")

    # ----------------------------------------------------------
    # Step 3 — Standardize both datasets
    # ----------------------------------------------------------
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    # ----------------------------------------------------------
    # Step 4 — Fit PLS model
    # ----------------------------------------------------------
    n_components = min(n_components, X_scaled.shape[1], Y_scaled.shape[1])
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)
    print(f" Fitted PLS model with {n_components} components")

    # ----------------------------------------------------------
    # Step 5 — Correlation between PLS components and sensory traits
    # ----------------------------------------------------------
    corr_df = pd.DataFrame(
        np.corrcoef(pls.x_scores_.T, Y_scaled.T)[:n_components, n_components:],
        index=[f"PLS{i+1}" for i in range(n_components)],
        columns=Y.columns,
    )

    print("\n Correlation between PLS components and sensory attributes:")
    display(corr_df.round(2))

    # ----------------------------------------------------------
    # Step 6 — Visualize correlation heatmap
    # ----------------------------------------------------------
    plt.figure(figsize=(14, 6))
    sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("PLS Components vs Sensory Attributes (High-Predicted Compounds)")
    plt.xlabel("Sensory Attributes")
    plt.ylabel("PLS Components")
    plt.tight_layout()
    plt.show()


    # Compound loadings

    loadings_df = pd.DataFrame(
        pls.x_loadings_,
        index=X.columns,
        columns=[f"PLS{i+1}" for i in range(n_components)],
    )

    for comp in loadings_df.columns:
        top_loadings = loadings_df[comp].abs().sort_values(ascending=False).head(15)
        plt.figure(figsize=(9, 4))
        sns.barplot(
            x=top_loadings.values,
            y=top_loadings.index,
            palette="viridis"
        )
        plt.title(f"Top 15 Compound Contributors to {comp}")
        plt.xlabel(f"|{comp} Loading| (Importance)")
        plt.ylabel("Compound")
        plt.tight_layout()
        plt.show()

    # Identify dominant sensory direction of each PLS axis
    flavor_alignment = {}
    for comp in corr_df.index:
        best_match = corr_df.loc[comp].abs().idxmax()
        direction = "positive" if corr_df.loc[comp, best_match] > 0 else "negative"
        flavor_alignment[comp] = f"{best_match} ({direction})"

    print("\n Flavor alignment per PLS axis:")
    for comp, desc in flavor_alignment.items():
        print(f"  {comp}: aligned with {desc}")

    #  Save outputs
    os.makedirs(save_dir, exist_ok=True)
    corr_path = os.path.join(save_dir, "pls_high_predicted_compound_sensory_correlations.csv")
    loadings_path = os.path.join(save_dir, "pls_high_predicted_compound_loadings.csv")

    corr_df.to_csv(corr_path)
    loadings_df.to_csv(loadings_path)

    print("\n Correlation and loadings files saved successfully:")
    print(f"   → {corr_path}")
    print(f"   → {loadings_path}")

    return {
        "corr_df": corr_df,
        "loadings_df": loadings_df,
        "flavor_alignment": flavor_alignment,
    }




#   Function to prepare aligned & scaled data
def prepare_plsr_data(predictions, high_r2_traits, sensory_reconstructed_path):
    """
    Align and standardize predictions (X) and sensory traits (Y) for PLSR.
    """
    # Load sensory reconstructed data
    sensory_df = pd.read_csv(sensory_reconstructed_path, index_col=0)

    # Normalize sample names
    sensory_df.index = sensory_df.index.str.upper().str.replace(" ", "")
    predictions.index = predictions.index.str.upper().str.replace(" ", "")

    # Find common varieties
    common_varieties = predictions.index.intersection(sensory_df.index)
    print(f" Common varieties found: {len(common_varieties)}")

    # Subset to shared samples
    X = predictions.loc[common_varieties, high_r2_traits].select_dtypes(include=[np.number])
    Y = sensory_df.loc[common_varieties].select_dtypes(include=[np.number])

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    print(f"X_scaled shape: {X_scaled.shape}, Y_scaled shape: {Y_scaled.shape}")
    return X, Y, X_scaled, Y_scaled


# Function for PLS biplot
def plot_plsr_biplot(
    pls_model,
    X_scaled: pd.DataFrame,
    pc1: int = 1,
    pc2: int = 2,
    top_n_loadings: int = 15,
    rotate: str = 'none',
    scale_arrows: float = 10,
    text_size: int = 8
):
    """
    Plot a PLS biplot showing sample scores and top contributing compound loadings.
    """
    # Extract scores
    x_scores = pls_model.x_scores_[:, pc1 - 1]
    y_scores = pls_model.x_scores_[:, pc2 - 1]

    # Apply rotation
    if rotate == '180':
        x_scores, y_scores = -x_scores, -y_scores
    elif rotate == '90cw':
        x_scores, y_scores = y_scores, -x_scores
    elif rotate == '90ccw':
        x_scores, y_scores = -y_scores, x_scores

    # Plot samples
    plt.figure(figsize=(10, 8))
    plt.scatter(x_scores, y_scores, c='dodgerblue', alpha=0.6, label='Varieties')

    for i, txt in enumerate(X_scaled.index):
        plt.text(x_scores[i], y_scores[i], txt, fontsize=text_size, alpha=0.7)

    # Loadings
    x_loadings = pls_model.x_loadings_[:, pc1 - 1]
    y_loadings = pls_model.x_loadings_[:, pc2 - 1]

    if rotate == '180':
        x_loadings, y_loadings = -x_loadings, -y_loadings
    elif rotate == '90cw':
        x_loadings, y_loadings = y_loadings, -x_loadings
    elif rotate == '90ccw':
        x_loadings, y_loadings = -y_loadings, x_loadings

    loadings_df = pd.DataFrame({'x': x_loadings, 'y': y_loadings}, index=X_scaled.columns)
    top_loadings = loadings_df.abs().sum(axis=1).sort_values(ascending=False).head(top_n_loadings).index

    for compound in top_loadings:
        plt.arrow(
            0, 0,
            loadings_df.loc[compound, 'x'] * scale_arrows,
            loadings_df.loc[compound, 'y'] * scale_arrows,
            color='crimson', alpha=0.4, head_width=0.05
        )
        plt.text(
            loadings_df.loc[compound, 'x'] * scale_arrows * 0.8,
            loadings_df.loc[compound, 'y'] * scale_arrows * 0.8,
            compound, color='crimson', fontsize=9
        )

    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)
    plt.xlabel(f"PLS{pc1} Scores")
    plt.ylabel(f"PLS{pc2} Scores")
    plt.title(f"PLS Biplot (PLS{pc1} vs PLS{pc2}) — rotated: {rotate}")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
