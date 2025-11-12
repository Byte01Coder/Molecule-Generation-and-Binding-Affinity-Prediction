import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

# Load data
df = pd.read_csv('BindingDB.csv')
df = df.dropna(subset=['SMILES', 'Seq', 'Affinity'])
df = df[df['Affinity'] > 0]

# Feature functions
def ligand_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(10)
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.MolMR(mol),
        Descriptors.RingCount(mol)
    ])

def target_features(seq):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq = seq.upper()
    total = len(seq)
    return np.array([seq.count(aa)/total if total > 0 else 0 for aa in amino_acids])

# Feature extraction
X = []
for _, row in df.iterrows():
    ligand_feat = ligand_features(row['SMILES'])
    target_feat = target_features(row['Seq'])
    X.append(np.concatenate((ligand_feat, target_feat)))

X = np.array(X)
y = np.log1p(df['Affinity'].values)  # Log-transform target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomizedSearchCV for XGBoost
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

model = XGBRegressor(random_state=42)
random_search = RandomizedSearchCV(model, param_dist, n_iter=20, scoring='r2', cv=3, verbose=1, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

best_model = random_search.best_estimator_

# Evaluation
y_pred_log = best_model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)  # Inverse log transform
y_true = np.expm1(y_test)

mse = mean_squared_error(y_true, y_pred)
metrics = {
    'MAE': mean_absolute_error(y_true, y_pred),
    'MSE': mse,
    'RMSE': np.sqrt(mse),
    'R2 Score': r2_score(y_true, y_pred)
}

print(metrics)
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# Save model and scaler
joblib.dump(best_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model and scaler saved successfully!")
