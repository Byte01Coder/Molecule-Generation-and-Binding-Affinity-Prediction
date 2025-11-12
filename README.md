## Dataset Link
[Click Here](https://drive.google.com/file/d/1RZAYNHajpPTkmu8xdr94SdphC5ERW8VO/view?usp=sharing)

---

## Project Overview
This project is a **three-stage AI-powered drug discovery pipeline** that:

1. **Generates** novel drug-like molecules conditioned on protein sequences (Deep Learning).  
2. **Predicts** binding affinity between molecules and proteins (Machine Learning).  
3. **Deploys** an interactive web app for real-time predictions (Streamlit).

---

## Molecule Generation with Deep Learning (`mol.py`)

### Objective
Generate novel **SMILES strings** (drug-like molecules) that are likely to bind to a given protein target.

### Key Components
- **Protein Encoder:**  
  Uses *ESM-2 (Evolutionary Scale Modeling)* to convert protein sequences into embeddings.  
- **Drug Decoder:**  
  Uses *GPT-2 (fine-tuned for SMILES generation)* with cross-attention to condition molecule generation on protein embeddings.

### Workflow
#### 1. Data Loading & Preprocessing
- **Input:** `BindingDB.csv` (contains protein sequences, SMILES, and binding affinities).  
- **Filters:**  
  - Keeps only high-affinity interactions *(Affinity ≤ 1000 nM)*.  
  - Removes invalid SMILES and large molecules *(MolWt ≤ 600)*.  
  - Limits protein sequence length *(≤ 512 amino acids)*.

#### 2. Model Architecture
- **Protein Encoder (ProteinEncoder):**  
  - Uses `ESM-2-tiny` (`facebook/esm2_t6_8M_UR50D`) for efficiency.  
  - Extracts protein embeddings → projects to lower dimension (256-D).  
- **Drug Decoder (DrugDecoder):**  
  - Uses GPT-2 with cross-attention layers to incorporate protein context.  
  - Only 2 cross-attention layers for memory efficiency.

#### 3. Training & Generation
- **Training:** Minimizes cross-entropy loss for SMILES generation.  
- **Generation:**  
  - Given a protein sequence, the model generates multiple SMILES candidates.  
  - Uses *top-k sampling (k=50)* for diverse yet high-quality outputs.

#### 4. Output
- A CSV file containing generated SMILES molecules.

---

## Binding Affinity Prediction with XGBoost (Stage 2)

### Objective
Predict how strongly a molecule (SMILES) binds to a protein (measured in **pKd**, where higher = stronger binding).

### Key Components
- **Feature Extraction:**  
  Computes molecular descriptors (RDKit) + protein sequence features (amino acid composition).  
- **XGBoost Model:**  
  Optimized using `RandomizedSearchCV`.

### Workflow
#### 1. Feature Extraction
- **Ligand Features (SMILES → 10 features):**  
  - `MolWt`, `NumHDonors`, `NumHAcceptors`, `TPSA`, `MolLogP`, etc.  
- **Target Features (Protein Sequence → 20 features):**  
  - Normalized amino acid frequencies (A, C, D, E, ..., Y).

#### 2. Model Training
- **Data Split:** 80% train, 20% test.  
- **Scaling:** `StandardScaler` applied to features.  
- **Hyperparameter Tuning:**  
  Uses `RandomizedSearchCV` to optimize `n_estimators`, `max_depth`, `learning_rate`, etc.  
- **Evaluation Metrics:**  
  - MAE, MSE, RMSE, R² (saved in `metrics.json`).

#### 3. Output
- `xgb_model.pkl` — Trained XGBoost model.  
- `scaler.pkl` — Feature scaler for preprocessing.  
- `metrics.json` — Performance metrics.

---

## Streamlit Web App for Deployment (Stage 3)

### Objective
Provide a user-friendly interface to:
- Upload generated SMILES (from Stage 1).  
- Input a protein sequence.  
- Predict binding affinities (using the XGBoost model).

### Key Features
- **File Upload:**  
  Accepts a CSV file with a `SMILES` column.  
- **Protein Sequence Input:**  
  Users paste a protein sequence (e.g., `"MKWVTFISLLFLFSSAYSRGV..."`).  
- **Prediction & Visualization:**  
  For each SMILES:  
  - Computes ligand + protein features.  
  - Scales features using `scaler.pkl`.  
  - Predicts pKd using `xgb_model.pkl`.  
  - Displays molecule structure (RDKit visualization).  
- **Model Metrics:**  
  Displays MAE, MSE, RMSE, and R² in the sidebar.

### Workflow Integration
1. **Input →** Generated SMILES (from Stage 1).  
2. **Prediction →** Uses XGBoost model (from Stage 2).  
3. **Output →** Interactive predictions displayed in the web app.
