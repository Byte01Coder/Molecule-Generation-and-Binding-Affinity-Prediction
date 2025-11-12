import pandas as pd
from DeepPurpose import utils, CompoundPred
import os

# ------------------ Config ------------------
input_csv = "./saved_models/generated_molecules.csv"
output_csv = "./saved_models/predicted_affinities_deeppurpose.csv"
target_protein = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAK"

# ------------------ Load SMILES ------------------
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"CSV not found at {input_csv}")

df = pd.read_csv(input_csv)
if "SMILES" not in df.columns:
    raise ValueError("The input CSV must contain a 'SMILES' column.")

smiles_list = df["SMILES"].dropna().tolist()
if not smiles_list:
    raise ValueError("No valid SMILES found in the CSV.")

# ------------------ Use Official Model ------------------
model_name = 'MPNN_CNN_BindingDB'

print(f"🔄 Loading DeepPurpose model: {model_name}")
model = CompoundPred.model_pretrained(model_name, pretrained_dir='pretrained_model')

# ------------------ Predict ------------------
print("🔬 Predicting binding affinities...")
protein_list = [target_protein] * len(smiles_list)
affinities = model.predict(smiles_list, protein_list)

# ------------------ Save Results ------------------
output_df = pd.DataFrame({
    "SMILES": smiles_list,
    "Predicted_Affinity": affinities
})

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
output_df.to_csv(output_csv, index=False)

print(f"\n✅ Predictions complete! Results saved to: {output_csv}")
