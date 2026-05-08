

"""Author: Fatemeh Monfared
Module: data_prepration.py
This module provides reusable, company-ready utilities to prepare TD–GC–MS peak tables for analysis. It cleans and standardizes QC and sample data, links peak intensities to master metadata (injection order / QC levels), and evaluates QC performance (linearity and stability). It then applies LOESS-based drift correction using stable reference peaks and produces QC-ranked, QC4-normalized outputs (wide and long formats) that are ready for downstream analysis. This module should be used for running  final_data_prepration.ipynb notebook."""



import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
from scipy.stats import linregress
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns


def clean_qc_data(qc_data_mixed, save_path=None):
    """
    Extracts columns ending with 'T1' and cleans QC mixed data from MsMetrix.
    This includes computing a single representative retention time (tR_best),
    removing redundant columns, and optionally saving the cleaned dataset.

    Parameters
    ----------
    qc_data_mixed : pandas.DataFrame
        Raw QC data containing columns such as 'Peak', 'tR1', 'tR2', and 'tR'.

    save_path : str, optional
        If provided, saves the cleaned DataFrame to this CSV path.

    Returns
    -------
    cleaned_qc_mixed : pandas.DataFrame
        Cleaned DataFrame with selected T1 columns and 'tR_best' retention time.

    t1_cols : list
        List of columns ending with 'T1' that were selected.
    """

    # --- Step 1: Clean column names ---
    qc_data_mixed.columns = qc_data_mixed.columns.str.strip()

    # --- Step 2: Define metadata columns ---
    metadata_cols = ['Peak', 'tR1', 'tR2', 'tR', 'm/z']

    # --- Step 3: Select columns that end with "T1" ---
    t1_cols = [c for c in qc_data_mixed.columns if c.strip().endswith("T1")]

    # --- Step 4: Extract metadata + T1 columns ---
    qc_data_T1 = qc_data_mixed[metadata_cols + t1_cols]

    print(" Extracted T1 columns:", len(t1_cols))
    print("Filtered DataFrame shape:", qc_data_T1.shape)

    # --- Step 5: Compute tR_best as the average of tR1 and tR2 ---
    qc_data_T1['tR_best'] = (qc_data_T1['tR1'] + qc_data_T1['tR2']) / 2

    # --- Step 6: Check consistency between tR_best and tR ---
    diff_avg_vs_t = (qc_data_T1['tR_best'] - qc_data_T1['tR']).abs()
    print(f"Mean |avg(tR1,tR2) - tR|: {diff_avg_vs_t.mean():.6f}")
    print(f"Max  |avg(tR1,tR2) - tR|: {diff_avg_vs_t.max():.6f}")

    # --- Step 7: Remove old retention time columns ---
    to_keep = [col for col in qc_data_T1.columns if col not in ['tR1', 'tR2', 'tR']]
    cleaned_qc_mixed = qc_data_T1[to_keep]

    # --- Step 8: Reorder columns to place tR_best after Peak ---
    cols = cleaned_qc_mixed.columns.tolist()
    if 'Peak' in cols and 'tR_best' in cols:
        cols.insert(cols.index('Peak') + 1, cols.pop(cols.index('tR_best')))
        cleaned_qc_mixed = cleaned_qc_mixed[cols]

    # --- Step 9: Optionally save to CSV ---
    if save_path:
        cleaned_qc_mixed.to_csv(save_path, index=False)
        print(f" Cleaned QC data saved to: {save_path}")

    # --- Step 10: Return cleaned data and selected column list ---
    return cleaned_qc_mixed, t1_cols






import numpy as np

import pandas as pd

def load_sample_data(input_path, sample_rows=None):
    """
    Load sample / QC / master tables using config-driven header row if provided.
    """

    # -------------------------------------------------
    # 1) If header row explicitly provided → trust config
    # -------------------------------------------------
    if sample_rows is not None:
        data = pd.read_excel(input_path, header=sample_rows)
        print(f" Header row taken from config: {sample_rows}")
        display(data.head())
        return data, sample_rows

    # -------------------------------------------------
    # 2) Otherwise: auto-detect (fallback only)
    # -------------------------------------------------
    df_sample = pd.read_excel(input_path, header=None, nrows=50)

    header_row = None

    for i in range(len(df_sample)):
        row = df_sample.iloc[i].astype(str).str.strip()
        if "Variety" in row.values:
            header_row = i
            break

    if header_row is None:
        non_null_counts = df_sample.notna().sum(axis=1)
        header_row = non_null_counts.idxmax()

    data = pd.read_excel(input_path, header=header_row)

    print(f" Header auto-detected at row: {header_row}")
    display(data.head())

    return data, header_row








def prepare_and_clean_potato_data(input_path, final_normalized_ids, output_path=None):
    """
    Load, normalize, and filter potato mixed biorep data, then analyze
    and clean retention times by computing a representative 'tR_best'.

    Parameters
    ----------
    input_path : str
        Path to the Excel file containing the raw mixed biorep data.

    final_normalized_ids : list
        List of normalized sample IDs to retain (e.g., ['S96_1_B1_D5_T1', ...]).

    output_path : str, optional
        If provided, saves the final cleaned dataset to this Excel file.

    Returns
    -------
    selected_mixed_data : pandas.DataFrame
        Final cleaned DataFrame containing:
        - 'Peak', 'tR_best', 'm/z', and selected sample columns.

    diffs : dict
        Dictionary with Series of retention time differences for QC.
    """

    # Load the raw Excel data without assuming a header 
    df_raw = pd.read_excel(input_path, header=None)

    #  Detect the header row (row where the first column equals 'Peak')
    header_row = df_raw[df_raw.iloc[:, 0].astype(str).str.strip() == "Peak"].index[0]
    potato_data = pd.read_excel(input_path, header=header_row)

    # Define metadata columns to retain 
    metadata_cols = ['Peak', 'tR1', 'tR2', 'tR', 'm/z']

    # Identify all sample columns (excluding metadata and ID Names) 
    sample_cols = [c for c in potato_data.columns if c not in metadata_cols and c != "ID Names"]

    # Define helper function to normalize sample IDs 
    def normalize_id(col):
        try:
            sid = col.split()[1]
        except IndexError:
            sid = col
        sid = re.sub(r'_T[0-9]+_', '_', sid)
        parts = sid.split("_")
        cleaned = []
        for p in parts:
            if re.fullmatch(r'T[0-9]+', p) and p not in ['T1', 'T2']:
                continue
            cleaned.append(p)
        return "_".join(cleaned)

    #  Create mapping of normalized ID → actual column name 
    norm_to_col = {normalize_id(c): c for c in sample_cols}

    #  Select actual column names matching the desired normalized IDs 
    selected_columns = [norm_to_col[nid] for nid in final_normalized_ids if nid in norm_to_col]

    #  Report missing sample IDs for QC 
    missing_ids = [nid for nid in final_normalized_ids if nid not in norm_to_col]
    if missing_ids:
        print(f" {len(missing_ids)} sample IDs were not found in the dataset.")
        print("Example missing IDs:", missing_ids[:10])

    # Create filtered dataset with metadata + selected samples 
    potato_mixed_filtered = potato_data[metadata_cols + selected_columns]

    # Analyze retention time differences 
    diff_t_vs_1 = (potato_mixed_filtered['tR'] - potato_mixed_filtered['tR1']).abs()
    diff_t_vs_2 = (potato_mixed_filtered['tR'] - potato_mixed_filtered['tR2']).abs()
    diff_1_vs_2 = (potato_mixed_filtered['tR1'] - potato_mixed_filtered['tR2']).abs()
    avg_t = (potato_mixed_filtered['tR1'] + potato_mixed_filtered['tR2']) / 2
    diff_avg_vs_t = (avg_t - potato_mixed_filtered['tR']).abs()

    # Print descriptive statistics 
    print("\nRetention Time Differences Summary:")
    print("abs(tR - tR1):\n", diff_t_vs_1.describe(), "\n")
    print("abs(tR - tR2):\n", diff_t_vs_2.describe(), "\n")
    print("abs(tR1 - tR2):\n", diff_1_vs_2.describe(), "\n")
    print("abs(avg(tR1,tR2) - tR):\n", diff_avg_vs_t.describe(), "\n")

    # Rename tR to tR_best 
    selected_mixed_data = potato_mixed_filtered.rename(columns={'tR': 'tR_best'})

        # Keep metadata and all selected sample columns
    columns_to_keep = ['Peak', 'tR_best', 'm/z'] + selected_columns
    selected_mixed_data = selected_mixed_data[columns_to_keep].copy()

    selected_mixed_data = selected_mixed_data[columns_to_keep].copy()

    print(f" Final dataset shape: {selected_mixed_data.shape}")

    # Save if output path provided 
    if output_path:
        selected_mixed_data.to_excel(output_path, index=False)
        print(f" Final cleaned data saved to: {output_path}")

    # --- Step 15: Return results ---
    diffs = {
        'diff_t_vs_1': diff_t_vs_1,
        'diff_t_vs_2': diff_t_vs_2,
        'diff_1_vs_2': diff_1_vs_2,
        'diff_avg_vs_t': diff_avg_vs_t
    }

    return selected_mixed_data, diffs




def average_gcms_intensities_by_variety(selected_mixed_data, master_table_path, output_path=None):
    """
    Combine replicate GC–MS intensity columns by variety using a master table mapping,
    clean the resulting column names, and optionally save to file.

    Parameters
    ----------
    selected_mixed_data : pandas.DataFrame
        Processed GC–MS dataset (already filtered and cleaned) containing:
        - 'Peak', 'tR_best', 'm/z' columns
        - Intensity columns starting with 's' (sample columns)

    master_table_path : str
        Path to the master table CSV file containing 'Filename' and 'Sample' columns.
        'Filename' must match the identifiers in intensity column names.

    output_path : str, optional
        If provided, the cleaned, averaged dataset (by variety) will be saved as a CSV file.

    Returns
    -------
    cleaned_df_mix : pandas.DataFrame
        Final cleaned DataFrame with averaged intensities per variety and simplified column names.
    """

    # Load and clean the master table 
    master_table = pd.read_csv(master_table_path)
    master_table["Filename"] = master_table["Filename"].astype(str).str.strip().str.lower()
    master_table["Sample"] = master_table["Sample"].astype(str).str.strip()

    # Create filename → variety mapping 
    filename_to_variety = master_table.set_index("Filename")["Sample"].to_dict()

    # Identify GC–MS intensity columns 
    intensity_cols = [col for col in selected_mixed_data.columns if col.lower().startswith("s")]
    print(f" Found {len(intensity_cols)} GC–MS intensity columns.")

    #  Map intensity columns → filenames and varieties 
    column_to_filename = {col: col.split(" ", 1)[-1].strip().lower() for col in intensity_cols}
    column_to_variety = {col: filename_to_variety.get(fname, None) for col, fname in column_to_filename.items()}

    # Keep only columns that successfully map to a variety
    mapped_cols = {col: variety for col, variety in column_to_variety.items() if variety is not None}
    print(f" Mapped {len(mapped_cols)} of {len(intensity_cols)} intensity columns to varieties.")

    # Show unmapped columns if any 
    unmapped = {col: fname for col, fname in column_to_filename.items() if fname not in filename_to_variety}
    if unmapped:
        print(f" {len(unmapped)} columns did not match any master table filenames.")
        print("Example unmapped entries:")
        for c, f in list(unmapped.items())[:10]:
            print(f"  {c}  →  {f}")

    if not mapped_cols:
        raise ValueError(" No matching columns found between GC–MS data and master table filenames!")

    # Initialize base DataFrame with metadata 
    renamed_data = selected_mixed_data[["Peak", "tR_best", "m/z"]].copy()

    #  Add intensity data grouped by variety 
    for col, variety in mapped_cols.items():
        if variety not in renamed_data:
            renamed_data[variety] = selected_mixed_data[col]
        else:
            renamed_data[variety] += selected_mixed_data[col]  # Sum replicate intensities

    # Average replicate intensities by variety 
    variety_counts = Counter(mapped_cols.values())
    for variety, count in variety_counts.items():
        renamed_data[variety] = renamed_data[variety] / count

    print(f" Averaged intensities for {len(variety_counts)} unique varieties.")

    # Clean column names (remove any leftover symbols) 
    cleaned_df_mix = renamed_data.copy()
    cleaned_df_mix.columns = [
        col.split(";")[-1].strip() if ";" in col else col for col in cleaned_df_mix.columns
    ]

    # Save output if requested
    if output_path:
        cleaned_df_mix.to_csv(output_path, index=False)
        print(f" Cleaned GC–MS variety data saved to: {output_path}")

    #  Summary 
    print(f" Final dataset shape: {cleaned_df_mix.shape}")
    print(f" Total unique varieties: {len(cleaned_df_mix.columns) - 3}")

    return cleaned_df_mix

def match_alkanes_to_peaks(cleaned_data, alkanes_df, rt_tolerance=0.05):
    """
    Match expected alkane retention times with observed GC–MS peaks.

    Parameters
    ----------
    cleaned_data : pandas.DataFrame
        GC–MS data containing 'tR_best' (or similar) and intensity columns.
    alkanes_df : pandas.DataFrame
        DataFrame containing columns ['Compound', 'tR'] with expected alkane RTs.
    rt_tolerance : float, optional
        Allowed retention time deviation for matching (in minutes).

    Returns
    -------
    pandas.DataFrame
        Table of matched alkanes with columns:
        ['Compound', 'Expected_tR', 'tR_best', 'Intensity', 'ΔtR'].
    """


    # Detect retention time column automatically
    rt_col_candidates = ["tR_best", "tR", "RT", "Retention Time", "tR(min)", "RT (min)"]
    rt_col = next((c for c in rt_col_candidates if c in cleaned_data.columns), None)
    if rt_col is None:
        raise KeyError("No retention time column found in data. Expected one of: " + ", ".join(rt_col_candidates))
    print(f" Detected retention time column: {rt_col}")

    # Compute mean intensity across all sample columns
    intensity_cols = [c for c in cleaned_data.columns if c not in ["Peak", rt_col, "m/z"]]
    cleaned_data["Intensity"] = cleaned_data[intensity_cols].mean(axis=1)

    # Match peaks
    matches = []
    for _, row in alkanes_df.iterrows():
        compound = row["Compound"]
        expected_tr = row["tR"]

        subset = cleaned_data[
            cleaned_data[rt_col].between(expected_tr - rt_tolerance, expected_tr + rt_tolerance)
        ]

        if not subset.empty:
            best_peak = subset.loc[subset["Intensity"].idxmax()]
            matches.append({
                "Compound": compound,
                "Expected_tR": expected_tr,
                "tR_best": best_peak[rt_col],
                "Intensity": best_peak["Intensity"],
                "ΔtR": abs(best_peak[rt_col] - expected_tr)
            })
        else:
            matches.append({
                "Compound": compound,
                "Expected_tR": expected_tr,
                "tR_best": np.nan,
                "Intensity": np.nan,
                "ΔtR": np.nan
            })

    return pd.DataFrame(matches)





def plot_alkane_matches(matches, save_path=None):
    """
    Plot reference (expected) vs matched retention times to visualize alkane matching accuracy.
    """

    plt.figure(figsize=(7, 5))
    plt.scatter(matches["Expected_tR"], matches["tR_best"], s=80, c="teal", edgecolor="k", alpha=0.8)

    # Ideal match line
    plt.plot(matches["Expected_tR"], matches["Expected_tR"], "r--", label="Ideal match (y = x)")

    # Annotate each alkane label and deviation
    for _, row in matches.iterrows():
        if not pd.isna(row["tR_best"]):
            plt.text(
                row["Expected_tR"], row["tR_best"] + 0.05,
                f"{row['Compound']} (Δ={row['ΔtR']:.2f})",
                ha="center", fontsize=9
            )

    plt.xlabel("Expected RT (min)", fontsize=12)
    plt.ylabel("Observed RT (min)", fontsize=12)
    plt.title("Alkane Retention Time Alignment", fontsize=13)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f" Plot saved to: {save_path}")
    else:
        plt.show()

    

def merge_qc_peak_intensities_with_metadata(cleaned_qc_mixed, master_table, qc_pattern=r"QC[1-4]$"):
    """
    Merge QC GC–MS data with metadata and retain only QC1–QC4 samples.
    """

    # Ensure 'Intensity' column won't conflict 
    if "Intensity" in cleaned_qc_mixed.columns:
        cleaned_qc_mixed = cleaned_qc_mixed.drop(columns=["Intensity"])

    # --- Step 1: Reshape wide QC data to long format ---
    df_long_qc = cleaned_qc_mixed.melt(
        id_vars=["Peak", "tR_best", "m/z"],
        var_name="SampleCol",
        value_name="Intensity"
    )

    # Extract filename from column name 
    df_long_qc["Filename"] = df_long_qc["SampleCol"].str.split().str[1]

    #  Merge with master table to add metadata 
    df_merged_qc = df_long_qc.merge(
        master_table[["Filename", "Sample", "Day", "Injection_order"]],
        on="Filename",
        how="left"
    )

    # Keep only QC1–QC4 samples 
    df_qc_mixed = df_merged_qc[df_merged_qc["Sample"].str.match(qc_pattern, na=False)]

    #  Summary
    print(f" Filtered QC samples found: {df_qc_mixed['Sample'].nunique()} → {df_qc_mixed['Sample'].unique().tolist()}")
    print(f" Final QC dataset shape: {df_qc_mixed.shape}")

    return df_qc_mixed





def analyze_qc_intensity_stability(alkane_matches, df_qc_mixed, min_samples=3, rt_tolerance=0.2):
    """
    Analyze QC Peak Intensities across QC injections per day (stable alkane peaks).
    Matches alkane RTs to QC peaks and computes R² per day.
    """

    results = []
    rt_col = "tR_best" if "tR_best" in df_qc_mixed.columns else "tR"

    #  Match each alkane to its nearest QC peak 
    for _, row in alkane_matches.iterrows():
        compound = row["Compound"]
        expected_rt = row.get("tR_best", row.get("tR"))
        if pd.isna(expected_rt):
            continue

        # Select peaks near expected alkane RT
        subset = df_qc_mixed[
            df_qc_mixed[rt_col].between(expected_rt - rt_tolerance, expected_rt + rt_tolerance)
        ].copy()

        if subset.empty:
            continue

        # Use the most intense peak (the alkane match)
        best_peak_row = subset.loc[subset["Intensity"].idxmax()]
        peak_id = best_peak_row["Peak"]
        matched_rt = best_peak_row[rt_col]

        #  Fit regression QC1–QC4 per day 
        for day, sub in df_qc_mixed[df_qc_mixed["Peak"] == peak_id].groupby("Day"):
            sub = sub.copy()
            sub["QC_num"] = sub["Sample"].str.extract(r"QC(\d+)").astype(float)

            if len(sub) >= min_samples and sub["QC_num"].notna().all():
                X = sub["QC_num"].values.reshape(-1, 1)
                y = sub["Intensity"].values

                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)

                results.append({
                    "Compound": compound,
                    "Peak": peak_id,
                    "Day": day,
                    "tR_best": matched_rt,
                    "m/z": sub["m/z"].iloc[0],
                    "R2": r2
                })

    # Combine results 
    results_df_mix = pd.DataFrame(results)

    print(f" QC stability analysis complete: {len(results_df_mix)} regression models fitted.")
    if not results_df_mix.empty:
        print(f"Mean R² across all models: {results_df_mix['R2'].mean():.3f}")
    else:
        print(" No regression models fitted — check QC naming or RT tolerance.")

    return results_df_mix





def run_qc_trend_dashboard(df_qc_mixed, results_df_mix, port=8052):
    """
    Launch an interactive Plotly Dash dashboard for visualizing QC linear trends
    across different experimental days.

    Parameters
    ----------
    df_qc_mixed : pandas.DataFrame
        Long-format QC data containing:
        - 'Peak', 'Sample', 'Day', 'Intensity', 'tR_best', and 'm/z'.

    results_df_mix : pandas.DataFrame
        Output from the QC stability analysis function containing:
        - 'Compound', 'Peak', 'Day', 'R2', 'tR_best', 'm/z'.

    port : int, optional (default=8052)
        Port number to run the dashboard on.

    Returns
    -------
    None
        Runs the Dash server locally at http://127.0.0.1:<port>/
    """

    app = Dash(__name__)
    app.title = "QC Linear Trend Dashboard"

    # App Layout 
    app.layout = html.Div([
        html.H1("QC Linear Trend Dashboard", style={"textAlign": "center"}),

        html.Label("Select Day:"),
        dcc.Dropdown(
            id="day-dropdown",
            options=[{"label": str(day), "value": day} for day in sorted(df_qc_mixed["Day"].unique())],
            value=sorted(df_qc_mixed["Day"].unique())[0],
            clearable=False
        ),

        dcc.Graph(id="qc-trend-plots")
    ])

    # Callback to update plots 
    @app.callback(
        Output("qc-trend-plots", "figure"),
        Input("day-dropdown", "value")
    )
    def update_plots(selected_day):
        qc_concentrations = {
            "QC1": 312.5,
            "QC2": 625,
            "QC3": 1250,
            "QC4": 2500
        }

        # Filter for selected day
        day_results = results_df_mix[results_df_mix["Day"] == selected_day]
        subplot_titles, traces = [], []

        for _, row_data in day_results.iterrows():
            sub = df_qc_mixed[
                (df_qc_mixed["Peak"] == row_data["Peak"]) &
                (df_qc_mixed["Day"] == selected_day)
            ].copy()

            sub["QC_label"] = sub["Sample"].str.extract(r"(QC\d+)")
            sub["Concentration"] = sub["QC_label"].map(qc_concentrations)

            X = sub["Concentration"].values.reshape(-1, 1)
            y = sub["Intensity"].values

            # Fit regression model if enough points exist
            if len(sub) > 1:
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                r2 = model.score(X, y)
            else:
                y_pred = y
                r2 = np.nan

            subplot_titles.append(
                f"{row_data['Compound']} (Peak {int(row_data['Peak'])}, R²={r2:.2f})"
            )
            traces.append((sub["Concentration"], y, y_pred, row_data))

        # Create subplot grid
        n_plots = len(traces)
        rows = (n_plots // 3) + (n_plots % 3 > 0)
        fig = make_subplots(rows=rows, cols=3, subplot_titles=subplot_titles)

        row_idx, col_idx = 1, 1
        for x_vals, y_vals, y_pred, row_data in traces:
            # Scatter plot (observed)
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals, mode="markers",
                    name=f"{row_data['Compound']} Peak {int(row_data['Peak'])} obs"
                ),
                row=row_idx, col=col_idx
            )

            # Regression line
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_pred, mode="lines",
                    name=f"{row_data['Compound']} Peak {int(row_data['Peak'])} fit"
                ),
                row=row_idx, col=col_idx
            )

            fig.update_xaxes(title_text="Concentration (pg/µL)", row=row_idx, col=col_idx)
            fig.update_yaxes(title_text="Intensity", row=row_idx, col=col_idx)

            col_idx += 1
            if col_idx > 3:
                col_idx = 1
                row_idx += 1

        fig.update_layout(
            height=300 * rows,
            width=1100,
            title_text=f"QC Linear Trends – Day {selected_day}",
            showlegend=False
        )

        return fig

    # Run the app 
    print(f" Dashboard running at: http://127.0.0.1:{port}/")
    app.run(debug=True, port=port)






def assess_stable_peak_injection_order_trend(df_qc_mixed, stable_peaks):
    """
    Assess stable peaks vs. injection order trend.

    This function checks if GC–MS peaks identified as stable (via ANOVA)
    show any drift or systematic trend in intensity over the injection order.
    It performs linear regression between injection order and log-transformed
    intensities for each stable peak.

    Parameters
    ----------
    df_qc_mixed : pandas.DataFrame
        QC long-format dataset containing 'Peak', 'Intensity', 
        'Injection_order', 'tR_best', and 'm/z'.

    stable_peaks : pandas.DataFrame
        DataFrame containing at least a 'Peak' column (e.g., peaks deemed stable by ANOVA).

    Returns
    -------
    stable_injection_df : pandas.DataFrame
        DataFrame summarizing regression results per stable peak with columns:
        ['Peak', 'tR_best', 'm/z', 'Slope', 'R2_injection', 'p_val'].
        - 'Slope' indicates the direction of trend (positive/negative drift).
        - 'R2_injection' quantifies the fit quality.
        - 'p_val' shows the significance of the trend.
    """

    stable_injection_check = []

    for peak in stable_peaks["Peak"].unique():
        # Extract data for this peak
        sub = df_qc_mixed[df_qc_mixed["Peak"] == peak].dropna(
            subset=["Intensity", "Injection_order"]
        )

        # Skip if too few data points
        if len(sub) < 2:
            continue

        # Perform linear regression (log-transformed intensity)
        slope, intercept, r, p, stderr = linregress(
            sub["Injection_order"], np.log10(sub["Intensity"] + 1)
        )

        stable_injection_check.append({
            "Peak": int(peak),
            "tR_best": sub["tR_best"].iloc[0],
            "m/z": sub["m/z"].iloc[0],
            "Slope": slope,
            "R2_injection": r ** 2,
            "p_val": p
        })

    stable_injection_df = pd.DataFrame(stable_injection_check)

    print(f" Checked {len(stable_injection_df)} stable peaks for injection-order drift.")
    print(stable_injection_df.sort_values('R2_injection', ascending=False).head())

    return stable_injection_df



def plot_injection_stability_loess(df_qc_mixed, stable_peaks, frac=0.3, max_plots=9):
    """
    Plot LOESS-smoothed QC intensity trends vs injection order for stable peaks.

    Parameters
    ----------
    df_qc_mixed : pandas.DataFrame
        Long-format QC data containing columns:
        ['Peak', 'Injection_order', 'Intensity', 'tR_best', 'm/z'].
    
    stable_peaks : pandas.DataFrame
        DataFrame (e.g., from ANOVA results) containing at least a 'Peak' column
        listing peaks considered stable across days.

    frac : float, optional (default=0.3)
        The fraction of data used when estimating each y-value in the LOESS smoother.

    max_plots : int, optional (default=9)
        Maximum number of peaks to plot per figure.

    Returns
    -------
    None
        Displays a grid of LOESS plots for the selected stable peaks.
    """

    stable_ids = stable_peaks["Peak"].unique()

    plt.figure(figsize=(14, 10))

    for i, peak in enumerate(stable_ids[:max_plots], 1):
        sub = df_qc_mixed[df_qc_mixed["Peak"] == peak].dropna(
            subset=["Intensity", "Injection_order"]
        )
        if sub.empty:
            continue

        x = sub["Injection_order"].values
        y = sub["Intensity"].values

        # Apply LOESS smoothing
        loess_fit = lowess(y, x, frac=frac, return_sorted=True)

        plt.subplot(3, 3, i)
        plt.scatter(x, y, alpha=0.6, label=f"Peak {peak}", color="steelblue")
        plt.plot(loess_fit[:, 0], loess_fit[:, 1], color="red", linewidth=2, label="LOESS")

        plt.title(f"Peak {peak}\n tR={sub['tR_best'].iloc[0]:.2f}, m/z={sub['m/z'].iloc[0]}")
        plt.xlabel("Injection order")
        plt.ylabel("Intensity")
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_qc_injection_stability_linear(df_qc_mixed, stable_peaks, qc_levels=None, max_plots=9):
    """
    Plot QC peak injection stability using linear regression trends.

    This function fits a linear regression line for each QC level (QC1–QC4)
    across injection order to evaluate the stability of peak intensities.
    Peaks with low R² (< 0.2) are considered stable.

    Parameters
    ----------
    df_qc_mixed : pandas.DataFrame
        Long-format QC dataset containing columns:
        ['Peak', 'Injection_order', 'Intensity', 'Sample', 'tR_best', 'm/z'].

    stable_peaks : list or array-like
        List of peak IDs considered stable or of interest (e.g., [170, 278, 573, ...]).

    qc_levels : list of str, optional (default=['QC1', 'QC2', 'QC3', 'QC4'])
        List of QC group labels to analyze.

    max_plots : int, optional (default=9)
        Maximum number of peaks to display per figure.

    Returns
    -------
    None
        Displays a grid of linear regression plots showing injection stability.
    """
    if qc_levels is None:
        qc_levels = ["QC1", "QC2", "QC3", "QC4"]

    plt.figure(figsize=(14, 10))

    for i, peak in enumerate(stable_peaks[:max_plots], 1):
        sub = df_qc_mixed[df_qc_mixed["Peak"] == peak].dropna(
            subset=["Intensity", "Injection_order"]
        )
        if sub.empty:
            continue

        plt.subplot(3, 3, i)
        max_r2 = 0  # track maximum R² across QC groups

        for qc in qc_levels:
            sub_qc = sub[sub["Sample"].str.startswith(qc)]
            if sub_qc.empty:
                continue

            x = sub_qc["Injection_order"].values
            y = sub_qc["Intensity"].values

            if len(sub_qc) > 1:
                model = LinearRegression().fit(x.reshape(-1, 1), y)
                y_pred = model.predict(x.reshape(-1, 1))
                r2 = model.score(x.reshape(-1, 1), y)
                max_r2 = max(max_r2, r2)
            else:
                y_pred = y
                r2 = np.nan

            plt.scatter(x, y, alpha=0.6, label=f"{qc} (R²={r2:.2f})")
            plt.plot(x, y_pred, linestyle="--")

        # classify stability
        status = "Stable " if max_r2 < 0.2 else "Unstable ⚠"

        plt.title(
            f"Peak {peak} ({status})\n"
            f"tR={sub['tR_best'].iloc[0]:.2f}, m/z={sub['m/z'].iloc[0]}"
        )
        plt.xlabel("Injection order")
        plt.ylabel("Intensity")
        plt.legend(fontsize=7)

    plt.tight_layout()
    plt.show()



def global_drift_correction(
    alkan_qc,
    cleaned_df_mix,
    stable_peaks=[278, 362, 403, 508],
    frac=0.3,
    delta=0.0,
    plot=True,
    save_path=None
):
    """
    Apply global LOESS drift correction using stable alkane QC peaks
    to both QC and averaged sample datasets.
    """

    # --- Step 1: Fit LOESS on stable QC peaks ---
    alkan_stable = alkan_qc[alkan_qc["Peak"].isin(stable_peaks)].copy()
    alkan_stable["Log_Intensity"] = np.log10(alkan_stable["Intensity"] + 1)

    loess_fit = lowess(
        endog=alkan_stable["Log_Intensity"],
        exog=alkan_stable["Injection_order"],
        frac=frac,
        delta=delta,
        return_sorted=True
    )

    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            alkan_stable["Injection_order"],
            alkan_stable["Log_Intensity"],
            alpha=0.3,
            label="Stable QC peaks"
        )
        plt.plot(loess_fit[:, 0], loess_fit[:, 1], color="red", lw=2, label="LOESS fit")
        plt.xlabel("Injection order")
        plt.ylabel("Log10 Intensity")
        plt.title("Global drift model (LOESS fit on stable QC peaks)")
        plt.legend()
        plt.show()

    # --- Step 2: Apply drift correction to QC data ---
    alkan_qc_corrected = alkan_qc.copy()
    drift_qc = np.interp(alkan_qc_corrected["Injection_order"], loess_fit[:, 0], loess_fit[:, 1])
    y_log = np.log10(alkan_qc_corrected["Intensity"] + 1)
    y_corr = y_log - (drift_qc - drift_qc.mean())
    alkan_qc_corrected["Intensity_corrected"] = 10**y_corr - 1
    print(f" QC drift correction applied ({len(alkan_qc_corrected)} rows)")

    # --- Step 3: Apply correction to sample dataset ---
    df_corrected = cleaned_df_mix.copy()

    # Melt the sample dataframe so that each measurement has an injection order
    melted = df_corrected.melt(
        id_vars=["Peak", "tR_best", "m/z"],
        var_name="Sample",
        value_name="Intensity"
    ).reset_index(drop=True)
    melted["Injection_order"] = np.arange(1, len(melted) + 1)

    # Interpolate drift per injection order
    drift_sample = np.interp(melted["Injection_order"], loess_fit[:, 0], loess_fit[:, 1])
    log_intensity = np.log10(melted["Intensity"] + 1)
    log_corrected = log_intensity - (drift_sample - drift_sample.mean())
    melted["Intensity_corrected"] = 10**log_corrected - 1

    # Pivot back to wide format
    df_corrected = melted.pivot_table(
        index=["Peak", "tR_best", "m/z"],
        columns="Sample",
        values="Intensity_corrected"
    ).reset_index()

    print(f" Sample drift correction applied ({df_corrected.shape[1]-3} samples)")

    #  QC R² evaluation 
    qc_eval = []
    for peak in alkan_qc["Peak"].unique():
        sub = alkan_qc[alkan_qc["Peak"] == peak]
        if len(sub) > 2:
            X = sub["Injection_order"].values.reshape(-1, 1)
            y_raw = np.log10(sub["Intensity"] + 1)
            drift_vals = np.interp(sub["Injection_order"], loess_fit[:, 0], loess_fit[:, 1])
            y_corr = y_raw - (drift_vals - drift_vals.mean())
            r2_raw = LinearRegression().fit(X, y_raw).score(X, y_raw)
            r2_corr = LinearRegression().fit(X, y_corr).score(X, y_corr)
            qc_eval.append((peak, r2_raw, r2_corr))

    qc_eval_df = pd.DataFrame(qc_eval, columns=["Peak", "R2_before", "R2_after"])


    # Variance evaluation across varieties 
    sample_eval = []
    for i, row in df_corrected.iterrows():
        raw_vals = np.log10(cleaned_df_mix.iloc[i, 3:] + 1)
        corr_vals = np.log10(df_corrected.iloc[i, 3:] + 1)
        sample_eval.append((row["Peak"], np.var(raw_vals), np.var(corr_vals)))

    sample_eval_df = pd.DataFrame(sample_eval, columns=["Peak", "Var_before", "Var_after"])

    if plot:
        plt.figure(figsize=(7, 5))
        sns.boxplot(
            data=pd.melt(sample_eval_df, id_vars=["Peak"], var_name="Stage", value_name="Variance"),
            x="Stage", y="Variance"
        )
        plt.title("Variance reduction across varieties")
        plt.show()

    # Save outputs 
    if save_path:
        df_corrected.to_csv(f"{save_path}/corrected_samples.csv", index=False)
        alkan_qc_corrected.to_csv(f"{save_path}/corrected_qc.csv", index=False)
        qc_eval_df.to_csv(f"{save_path}/qc_eval.csv", index=False)
        sample_eval_df.to_csv(f"{save_path}/sample_eval.csv", index=False)
        print(f" Results saved to {save_path}")

    print(" Global drift correction complete.")
    return df_corrected, alkan_qc_corrected, qc_eval_df, sample_eval_df



def rank_qcs(alkan_qc, alkan_qc_corrected):
    """
    Evaluate and rank QC samples based on drift correction performance.

    Parameters
    ----------
    df_qc_mixed : DataFrame
        Original QC dataset (before correction) containing columns:
        ['Sample', 'Injection_order', 'Intensity', 'Peak', ...]
    alkan_qc_corrected : DataFrame
        Drift-corrected QC dataset with ['Sample', 'Injection_order', 'Intensity_corrected'].

    Returns
    -------
    qc_df : DataFrame
        Summary of R², variance, and slope before/after correction for each QC.
    """

    qc_stats = []

    for qc in alkan_qc["Sample"].unique():
        if not str(qc).startswith("QC"):
            continue

        # Filter for the same QC level before and after correction
        raw = alkan_qc[alkan_qc["Sample"] == qc].copy()
        corr = alkan_qc_corrected[alkan_qc_corrected["Sample"] == qc].copy()

        if len(raw) < 5 or len(corr) < 5:
            continue

        #  R² before 
        X_raw = raw["Injection_order"].values.reshape(-1, 1)
        y_raw = np.log10(raw["Intensity"].values + 1)
        r2_raw = LinearRegression().fit(X_raw, y_raw).score(X_raw, y_raw)

        # --- R² after ---
        X_corr = corr["Injection_order"].values.reshape(-1, 1)
        y_corr = np.log10(corr["Intensity_corrected"].values + 1)
        r2_corr = LinearRegression().fit(X_corr, y_corr).score(X_corr, y_corr)

        # Variance 
        var_before = np.var(y_raw)
        var_after = np.var(y_corr)

        # Slope 
        slope_raw = LinearRegression().fit(X_raw, y_raw).coef_[0]
        slope_corr = LinearRegression().fit(X_corr, y_corr).coef_[0]

        qc_stats.append({
            "QC": qc,
            "Var_before": var_before,
            "Var_after": var_after,
            "R2_before": r2_raw,
            "R2_after": r2_corr,
            "Slope_before": slope_raw,
            "Slope_after": slope_corr,
            "ΔVar(%)": (1 - var_after / var_before) * 100,
            "ΔR²(%)": (r2_corr - r2_raw) * 100
        })

    qc_df = pd.DataFrame(qc_stats)

    if qc_df.empty:
        print(" No valid QC samples found for ranking.")
        return qc_df

    #  Combined performance score
    qc_df["Score"] = qc_df["ΔVar(%)"] + qc_df["ΔR²(%)"] - abs(qc_df["Slope_after"]) * 1000
    qc_df = qc_df.sort_values("Score", ascending=False).reset_index(drop=True)

    #  Plot summary heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        qc_df.set_index("QC")[["Slope_after", "Var_after", "R2_after"]],
        annot=True, fmt=".3f", cmap="coolwarm", cbar=True
    )
    plt.title("QC ranking based on drift correction performance")
    plt.show()

    print(" QC ranking complete.")
    return qc_df


def normalize_with_qc4_meaned_samples(df_corrected, alkan_qc_corrected):
    """
    Normalize variety-level averaged GC–MS data using QC4 medians per peak.
    Also visualize normalization effect with a global before/after boxplot.
    """

    # Compute QC4 median per Peak 
    qc4_ref = (
        alkan_qc_corrected[alkan_qc_corrected["Sample"] == "QC4"]  #  changed QC_label → Sample
        .groupby("Peak")["Intensity_corrected"]
        .median()
        .rename("QC4_median")
        .reset_index()
    )
    print(f" QC4 medians computed for {len(qc4_ref)} peaks")

    #  Global fallback QC4 median-
    global_qc4_median = (
        alkan_qc_corrected.loc[alkan_qc_corrected["Sample"] == "QC4", "Intensity_corrected"]
        .median()
    )
    print(f" Global QC4 median intensity = {global_qc4_median:.4f}")

    #  Prepare df_corrected 
    if "Sample" not in df_corrected.columns:
        df_corrected["Sample"] = "AllSamples"
        print(" 'Sample' column not found — added default value 'AllSamples'.")

    id_vars = ["Sample", "Peak", "tR_best", "m/z"]
    id_vars = [c for c in id_vars if c in df_corrected.columns]
    value_vars = [c for c in df_corrected.columns if c not in id_vars]

    #Melt wide to long 
    df_long = df_corrected.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Variety",
        value_name="Intensity_corrected"
    )
    print(f" Long-format data created: {df_long.shape}")

    #  Merge with QC4 reference 
    df_long = df_long.merge(qc4_ref, on="Peak", how="left")

    # Fill missing QC4 medians
    missing_before = df_long["QC4_median"].isna().sum()
    df_long["QC4_median"] = df_long["QC4_median"].fillna(global_qc4_median)
    missing_after = df_long["QC4_median"].isna().sum()
    print(f" Filled {missing_before - missing_after} missing QC4 medians using global QC4 median.")

    #  Normalize 
    df_long["Ratio_normalized"] = df_long["Intensity_corrected"] / df_long["QC4_median"]
    df_long["Log10_normalized"] = np.log10(df_long["Ratio_normalized"] + 1)

    #  Pivot back to wide format
    df_wide = df_long.pivot_table(
        index=["Sample", "Peak", "tR_best", "m/z"],
        columns="Variety",
        values="Log10_normalized"
    ).reset_index()

    print(" Normalization by QC4 completed (for averaged varieties).")

    # Global before/after comparison 
    df_plot = pd.DataFrame({
        "Before": np.log10(df_long["Intensity_corrected"] + 1),
        "After": df_long["Log10_normalized"]
    })

    df_melt = df_plot.melt(var_name="Stage", value_name="Log10 Intensity")

    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df_melt, x="Stage", y="Log10 Intensity", palette=["#4c72b0", "#dd8452"])
    plt.title("QC4 normalization effect on sample intensities")
    plt.xlabel("")
    plt.ylabel("Log10 Intensity")
    plt.tight_layout()
    plt.show()

    print(" Boxplot comparison (Before vs After) generated successfully.")

    return df_wide, df_long
