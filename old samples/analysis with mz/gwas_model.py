from databricks.sdk.runtime import *


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
