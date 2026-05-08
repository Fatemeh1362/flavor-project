# Aroma & Flavor Project — End-to-End Databricks Pipeline

## Executive Summary

This project delivers a fully reproducible, modular pipeline for analyzing potato aroma and flavor traits by integrating:

* GC-MS chemical data
* Aroma phenotypes (MSMetrix)
* Genomic data (taglotype / GWAS)

The system enables:

* Data preparation
* Compound-level analysis
* Genome-wide association studies (GWAS)
* Predictive modeling (PLSR, ML models)
* Statistical validation (PCA, effect size, dosage)
* Cross-dataset comparison (old vs new harvested potatoes)
* Cross-population validation (American dataset)

The pipeline is designed for scalability, reproducibility, and direct application in breeding decisions.

---

## 1. Project Architecture

```text
Data Layer (DBFS)
   ↓
Configuration (YAML)
   ↓
Processing (Notebooks)
   ↓
Analysis (GWAS + Stats + Models)
   ↓
Outputs (Structured Results)
   ↓
Dashboards
```

---

## 2. Data Layer

### Location

```bash
/Volumes/bmqg/default_bronze/fatemeh/data/
```

---

### Data Categories

#### 2.1 Phenotypic Data (Aroma Matrix)

* MSMetrix outputs
* Peak intensities per sample
* QC datasets

Examples:

* American Samples MSMetrix.xlsx
* Qcs American Samples MSMetrix.xlsx
* QCs New Harvest MsMetrix Data.xlsx
* New Harvest potato samples.xlsx
* qc_mixed_biorep.xlsx
* sample_mix_bioreps.xlsx

---

#### 2.2 Chemical / GC-MS Data

* Compound peak tables
* Identified volatiles
* Relative intensities

Examples:

* GCMS Steamed 96 set 2024 Mastertable.csv
* GCMS Steamed 96 set 2025.xlsx
* GCMS SWI KW 2025.xlsx

---

#### 2.3 Sensory Data

* Sensorial profiling SD 2005.xlsx

Links chemical signals to human perception.

---

## 3. Configuration

```bash
/Volumes/bmqg/default_bronze/fatemeh/config_mixed.yaml
```

Single source of truth for:

* Data paths
* GWAS inputs
* Taglotype tables
* Parameters (thresholds, filters)
* Output locations

 Always validate before running notebooks.

---

## 4. Outputs

### Location

```bash
/Volumes/bmqg/default_bronze/fatemeh/final_project/
```

---

### Output Structure

#### GWAS & Manhattan

* main_manhattan_plot/
* main_manhattan_plot_newharvested/
* manhattan_plot_beta/
* manhattan_plot_beta_newharvested/
* manhattan_lead_taglo/
* manhattan_lead_taglo_newharvested/

---

#### Allelic Effects

* allelic_effect_plots/
* allelic_effect_plots_newharvested/

---

#### Modeling Results

* model_results/
* model_results_newharvested/

---

#### PLSR

* plsr/
* plsr_newharvested/

---

#### Processed Data

* csv_outputs/
* csv_outputs_newharvested/
* csv_outputs_american/

---

#### PCA & Combined Analysis

* pca_new_manhattan_plot/
* pca_old_manhattan_plot/

---

#### Signal Processing

* signal_vs_noise_taglo/

---

#### General Results

* results/

---

#### GWAS Supporting Files

* mohammad_GemmaFiles/

---

## 5. Reusable Modules

```bash
/Volumes/bmqg/default_bronze/fatemeh/final_project/modules/
```

### Modules

* **PLSR.py** → PLS regression modeling
* **effect_size_gemma.py** → effect size calculation
* **dosage_linearity.py** → dosage validation
* **shared_signal.py** → shared loci detection
* **cross_trait_signals.py** → cross-trait analysis
* **manhattan_module.py** → Manhattan plots

---

## 6. Notebooks

```bash
/Workspace/Users/fatemeh.monfared@hzpc.com/aroma_flavor_project/final_flavor_project/
```

---

## 7. Core Pipeline (Execution Order)

1. final_data_preparation
2. final_compound_analysis
3. models

---

### Key Steps

#### Data Preparation

* Cleaning
* Normalization
* Genotype–phenotype alignment

#### Compound Analysis

* GC-MS feature extraction
* Aroma matrix construction

#### Modeling

* Model training
* Evaluation

---

## 8. GWAS

### Notebooks

* gwas_67_aroma
* gwas_aroma_newharvested
* gwas_pca_trait_old
* gwas_pca_trait_new

---

### Outputs

* `bmqg.gwas.run_local_20251207`
* `bmqg.gwas.run_local_20260305_newharvested`
* `bmqg.gwas.run_local_20260318_american`

These tables feed all downstream analyses.

---

## 9. Dataset Structure

The project is organized into dataset-specific pipelines with consistent processing steps.

---

### 9.1 Old Potato (Baseline)

Folder: `old_potato/`

Includes:

* Data preparation & compound analysis
* GWAS (`gwas_67_aroma`)
* Visualization:

  * manhattan_plot
  * manhattanplot_beta
  * manhattanplot_tagloid
  
* Statistical analysis:

  * PCA
  * cross-trait analysis
* Validation:

  * effect_size_gemma
  * dosage_linearity
* Modeling:

  * models
  * QTL_level_plsr
* Signal processing:

  * real_noise_tagloids
  * Shared GWAS Signals

✔ Reference dataset for comparisons

---

### 9.2 New Harvested Potato

Folder: `new_harvested_potato/`

Includes mirrored pipeline:

* final_data_preparation_newharvested
* final_compound_analysis_newharvested
* gwas_aroma_newharvested
* models_newharvested

Additional:

* PCA_newharvested
* cross-trait analysis_newharvested
* effect_size_gemma_newharvested
* dosage_linearity_newharvested
* real_noise_tagloids_newharvested
* Shared GWAS Signals_newharvested
* aroma_pca_kmeans

✔ Enables direct comparison with old dataset

---

### 9.3 Aging Effect Analysis

Folder: `Aging effect/`

Includes:

* compare_old_new harvested potato
* comparing gwas_pca_old_new
* new_old_pca_manhattanplot
* gwas_pca_trait_old
* gwas_pca_trait_new
* aroma_cluster_results

✔ Focus:

* Storage impact on aroma
* PCA and GWAS comparison

---

### 9.4 American Dataset

Folder: `American potato/`

Includes:

* final_data_preparation_american
* final_compound_analysis_american
* data_preparation.py
* compound_analysis.py

✔ Used for cross-population validation

---

### 9.5 Merging

Notebook:

```
merging american_newharvested potato
```

✔ Combines American and new harvested datasets for joint analysis

---

### 9.6 Dashboards

Folder: `dashboard/`

Includes:

* manhattanplot_dashboard
* effectsize_dashboard
* dosage_linearity_dashboard
* dosage_filtering_dashboard
* locus_zoom_Dashboard
* plsr_dashboard
* plsr_newharvested_dashboard
* variety_compounds_Dashboard

✔ Provides interactive visualization and interpretation

---

## Structure Summary

```text
old_potato/            → baseline pipeline
new_harvested_potato/ → mirrored pipeline
Aging effect/         → comparison layer
American potato/      → external validation
merging notebook      → integration
dashboard/            → visualization
```

---

## 10. Workflow Summary

```text
Data (DBFS)
   ↓
config_mixed.yaml
   ↓
Data Preparation
   ↓
Compound Analysis
   ↓
GWAS
   ↓
PCA / Manhattan / Effect Size / PLSR
   ↓
Old vs New Comparison
   ↓
Merge (American + New)
   ↓
Dashboards
```

---

## 11. Design Principles

* Centralized configuration
* Modular architecture
* Reproducibility
* Dataset consistency
* Scalability

---

## 12. Business & Scientific Value

* Identification of aroma-related genetic markers
* Understanding storage (aging) effects
* Cross-population validation
* Support for breeding decisions
* Interpretable analytics

---

## Author

Fatemeh Monfared
HZPC / Hanze University of Applied Sciences
2026
