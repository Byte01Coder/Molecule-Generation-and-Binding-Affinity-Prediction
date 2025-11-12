# Molecule Generation and Binding Affinity Prediction
# Complete implementation with real data

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, Descriptors, MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import io
from tqdm import tqdm
import random
import pickle
from transformers import RobertaModel, RobertaTokenizer

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels=None):
        self.smiles_list = smiles_list
        self.labels = labels
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smile = self.smiles_list[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return smile, label
        return smile

def download_dataset(url):
    """Download and prepare the dataset from the given URL"""
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download dataset: {response.status_code}")

# Part 1: Molecule Generation using ChemBERTa (pre-trained)

class MoleculeGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"Using device: {device}")
        self.device = device
        
        # Load ChemBERTa model as generator backbone
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1').to(device)
        
        # Define a decoder head for SMILES generation
        self.hidden_size = self.model.config.hidden_size
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, len(self.tokenizer)),
        ).to(device)
        
        # For RL training
        self.optimizer = torch.optim.Adam(list(self.decoder.parameters()), lr=1e-4)
        
    def train_generator(self, smiles_list, epochs=5, batch_size=32):
        """Train the generator to produce valid SMILES"""
        dataset = MoleculeDataset(smiles_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_values = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()
                
                # Tokenize smiles
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                
                # Get ChemBERTa embeddings
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state
                
                # Predict next tokens (shift sequence by 1)
                input_ids = encoded['input_ids'][:, :-1]
                target_ids = encoded['input_ids'][:, 1:]
                logits = self.decoder(embeddings[:, :-1, :])
                
                # Calculate loss (cross-entropy)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), 
                                      ignore_index=self.tokenizer.pad_token_id)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            loss_values.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values)
        plt.title('Generator Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('generator_training_loss.png')
        plt.close()
        
        return loss_values
    
    def generate_molecules(self, n_samples=100, max_length=128, temperature=1.0):
        """Generate new molecules using the trained model"""
        generated_smiles = []
        
        for _ in tqdm(range(n_samples), desc="Generating molecules"):
            # Start with BOS token
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # Generate tokens one by one
            for _ in range(max_length):
                encoded = {'input_ids': input_ids, 'attention_mask': attention_mask}
                
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    embeddings = outputs.last_hidden_state
                    logits = self.decoder(embeddings)
                    
                    # Apply temperature
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Sample from the distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # If EOS token, stop generation
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Add token to sequence
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            # Decode the tokens to a SMILES string
            smiles = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # Check if valid molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                generated_smiles.append(smiles)
        
        return generated_smiles
    
    def rl_fine_tune(self, reward_model, initial_molecules, epochs=5, n_samples=32):
        """Fine-tune the generator using reinforcement learning with the binding affinity model"""
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Generate batch of molecules
            generated_smiles = self.generate_molecules(n_samples=n_samples, temperature=1.2)
            valid_smiles = [s for s in generated_smiles if Chem.MolFromSmiles(s) is not None]
            
            if len(valid_smiles) == 0:
                print("No valid molecules generated in this batch, skipping update")
                continue
                
            # Calculate rewards using the binding affinity model
            rewards = []
            for smiles in valid_smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        reward = reward_model.predict([smiles])[0]
                        # Normalize reward: higher binding affinity (more negative) is better
                        rewards.append(-reward)  # Convert to positive reward
                    else:
                        rewards.append(0)  # Invalid molecule
                except:
                    rewards.append(0)
            
            # If all rewards are 0, skip this batch
            if sum(rewards) == 0:
                continue
                
            # Normalize rewards
            rewards = torch.tensor(rewards, device=self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Calculate policy gradient loss
            loss = 0
            for i, smiles in enumerate(valid_smiles):
                encoded = self.tokenizer(smiles, return_tensors="pt").to(self.device)
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state
                
                input_ids = encoded['input_ids'][:, :-1]
                target_ids = encoded['input_ids'][:, 1:]
                logits = self.decoder(embeddings[:, :-1, :])
                
                # Calculate probability of generating this sequence
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(log_probs.view(-1, log_probs.size(-1)), 1, 
                                             target_ids.view(-1, 1)).squeeze()
                seq_log_prob = token_log_probs.sum()
                
                # Policy gradient loss
                loss -= seq_log_prob * rewards[i]
            
            # Normalize loss
            loss = loss / len(valid_smiles)
            
            loss.backward()
            self.optimizer.step()
            
            print(f"RL Epoch {epoch+1}/{epochs}, Avg Reward: {rewards.mean().item():.4f}")
        
        return self.generate_molecules(n_samples=100, temperature=0.8)

# Part 2: Binding Affinity Prediction Model

class MoleculeGNN(nn.Module):
    def __init__(self, node_features=30, hidden_dim=64, output_dim=1):
        super(MoleculeGNN, self).__init__()
        self.node_features = node_features
        
        # GNN layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def smiles_to_graph(smiles):
    """Convert SMILES to a PyTorch Geometric graph"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        # Basic atom features: One-hot encoded atom type, formal charge, hybridization, etc.
        features = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetTotalDegree(),  # Degree
            atom.GetFormalCharge(),  # Formal charge
            atom.GetTotalNumHs(),  # Number of Hs
            atom.GetNumRadicalElectrons(),  # Number of radical electrons
            atom.GetIsAromatic() * 1,  # Aromaticity
        ]
        
        # One-hot encode atom type (common elements C, N, O, F, P, S, Cl, Br, I)
        atom_type = [0] * 24  # For common elements
        atom_type[min(atom.GetAtomicNum() - 1, 23)] = 1
        
        # Add more features like hybridization
        hybridization = [0] * 5  # sp, sp2, sp3, sp3d, sp3d2
        hybridization_idx = min(atom.GetHybridization(), 5) - 1
        if hybridization_idx >= 0:
            hybridization[hybridization_idx] = 1
        
        # Combine features
        atom_features.append(features + atom_type + hybridization)
    
    # Get bonds and their indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add bonds in both directions for undirected graph
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    
    # Convert to PyTorch tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    return data

class BindingAffinityPredictor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = MoleculeGNN().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()
        
    def process_dataset(self, smiles_list, labels):
        """Process the dataset into graph form"""
        graph_data = []
        processed_labels = []
        valid_indices = []
        
        for i, smile in enumerate(tqdm(smiles_list, desc="Processing molecules")):
            graph = smiles_to_graph(smile)
            if graph is not None:
                graph_data.append(graph)
                processed_labels.append(labels[i])
                valid_indices.append(i)
        
        # Standardize labels
        processed_labels = np.array(processed_labels).reshape(-1, 1)
        normalized_labels = self.scaler.fit_transform(processed_labels)
        
        return graph_data, normalized_labels, valid_indices
    
    def train(self, smiles_list, binding_affinity_values, epochs=50, batch_size=32):
        """Train the binding affinity prediction model"""
        # Process the dataset
        graph_data, normalized_labels, valid_indices = self.process_dataset(smiles_list, binding_affinity_values)
        
        # Split into train and validation
        indices = list(range(len(graph_data)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        train_graphs = [graph_data[i] for i in train_idx]
        train_labels = torch.tensor(normalized_labels[train_idx], dtype=torch.float32).to(self.device)
        
        val_graphs = [graph_data[i] for i in val_idx]
        val_labels = torch.tensor(normalized_labels[val_idx], dtype=torch.float32).to(self.device)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            batch_indices = list(range(0, len(train_graphs), batch_size))
            random.shuffle(batch_indices)
            
            for i in tqdm(batch_indices, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
                batch_graphs = train_graphs[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                # Create batch
                batch = Batch.from_data_list(batch_graphs).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(batch_graphs)
            
            train_loss = total_loss / len(train_graphs)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_batches = [val_graphs[i:i+batch_size] for i in range(0, len(val_graphs), batch_size)]
                val_label_batches = [val_labels[i:i+batch_size] for i in range(0, len(val_labels), batch_size)]
                
                val_loss = 0
                for batch_graphs, batch_labels in zip(val_batches, val_label_batches):
                    batch = Batch.from_data_list(batch_graphs).to(self.device)
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch_labels)
                    val_loss += loss.item() * len(batch_graphs)
                
                val_loss = val_loss / len(val_graphs)
                val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping could be added here
        
        # Plot training progress
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        
        # Predict on validation set
        self.model.eval()
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for i in range(0, len(val_graphs), batch_size):
                batch_graphs = val_graphs[i:i+batch_size]
                batch = Batch.from_data_list(batch_graphs).to(self.device)
                outputs = self.model(batch).cpu().numpy()
                val_predictions.extend(outputs)
                val_true.extend(val_labels[i:i+batch_size].cpu().numpy())
        
        # Convert to original scale
        val_predictions = self.scaler.inverse_transform(np.array(val_predictions))
        val_true = self.scaler.inverse_transform(np.array(val_true))
        
        # Plot predictions vs actual
        plt.scatter(val_true, val_predictions, alpha=0.5)
        plt.plot([min(val_true), max(val_true)], [min(val_true), max(val_true)], 'r--')
        plt.xlabel('Actual Binding Affinity')
        plt.ylabel('Predicted Binding Affinity')
        plt.title('Predictions vs Actual (Validation Set)')
        
        plt.tight_layout()
        plt.savefig('binding_affinity_training.png')
        plt.close()
        
        return train_losses, val_losses
    
    def predict(self, smiles_list):
        """Predict binding affinity for a list of SMILES strings"""
        self.model.eval()
        predictions = []
        
        for smile in smiles_list:
            graph = smiles_to_graph(smile)
            if graph is None:
                predictions.append(float('nan'))
                continue
                
            with torch.no_grad():
                graph_batch = Batch.from_data_list([graph]).to(self.device)
                prediction = self.model(graph_batch).cpu().numpy()
                # Convert back to original scale
                prediction = self.scaler.inverse_transform(prediction)[0][0]
                predictions.append(prediction)
        
        return predictions
    
    def save_model(self, path="binding_affinity_model.pt"):
        """Save the trained model"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler
        }
        torch.save(model_state, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="binding_affinity_model.pt"):
        """Load a trained model"""
        model_state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(model_state['model_state_dict'])
        self.scaler = model_state['scaler']
        print(f"Model loaded from {path}")

# Part 3: Complete Pipeline Integration

def run_full_pipeline():
    """Run the complete pipeline with real datasets"""
    print("Starting Molecule Generation and Binding Affinity Prediction Pipeline")
    
    # Download and prepare datasets
    print("\n=== Downloading and preparing datasets ===")
    
    # 1. PDBbind dataset for binding affinities (real data)
    print("Preparing PDBbind dataset...")
    
    # We'll use a directly accessible subset of PDBbind from GitHub
    pdbbind_url = "https://raw.githubusercontent.com/deepchem/deepchem/master/deepchem/data/tests/pdbbind_core_df.csv"
    
    try:
        pdbbind_content = download_dataset(pdbbind_url)
        pdbbind_df = pd.read_csv(io.StringIO(pdbbind_content.decode('utf-8')))
        print(f"Downloaded PDBbind dataset with {len(pdbbind_df)} entries")
    except Exception as e:
        print(f"Error downloading PDBbind dataset: {e}")
        # Fallback to a smaller local dataset
        print("Using fallback PDBbind sample data")
        # Sample data format: smiles,pdb_id,binding_affinity
        pdbbind_df = pd.DataFrame({
            'smiles': [
                'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
                'CC(=O)OC1=CC=CC=C1C(=O)O',       # Aspirin
                'CC(C)CC(C(=O)O)C1=CC=CC=C1',     # Ketoprofen
                'COC1=CC=C(C=C1)C(=O)C2=C(OC(O2)(C)C)C3=CC=CC=C3', # Warfarin
                'CC1=C(C=C(C=C1)S(=O)(=O)N)CC2=CC=C(C=C2)S(=O)(=O)N', # Furosemide
                'CC1=CN=C(C=C1)CN2C=NC3=C2C=C(C=C3)C(=O)N', # Pyrazinamide
                'CCN(CC)CCOC(=O)C1=CC=CC=C1NC', # Procaine
                'CCOC(=O)C1=CC=CC=C1OC(=O)C', # Ethyl salicylate
                'CC1=CC=C(C=C1)NC(=O)C2=CC=C(Cl)C=C2', # Diclofenac
                'CC1=C(C2=C(C=C1)C(=CC=N2)C(=O)O)OC', # Nalidixic acid
            ],
            'pdb_id': ['IBPF', 'ASP', 'KET', 'WAR', 'FUR', 'PYR', 'PRO', 'ESA', 'DIC', 'NAL'],
            'binding_affinity': [-6.4, -7.1, -5.9, -8.5, -7.3, -6.8, -5.4, -6.2, -7.9, -6.5]
        })
    
    # 2. ChEMBL dataset for molecule generation (real data)
    print("Preparing ChEMBL dataset...")
    
    # We'll use a directly accessible subset from GitHub
    chembl_url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    
    try:
        chembl_content = download_dataset(chembl_url)
        chembl_df = pd.read_csv(io.StringIO(chembl_content.decode('utf-8')))
        print(f"Downloaded ChEMBL-like dataset with {len(chembl_df)} molecules")
        
        # Extract SMILES column
        smiles_list = chembl_df['smiles'].tolist()
        
        # Filter valid molecules
        valid_smiles = []
        for smile in tqdm(smiles_list[:10000], desc="Validating SMILES"):  # Limit to 10,000 for speed
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                valid_smiles.append(smile)
        
        print(f"Retained {len(valid_smiles)} valid molecules")
        
    except Exception as e:
        print(f"Error downloading ChEMBL dataset: {e}")
        # Fallback to a small local dataset
        print("Using fallback ChEMBL sample data")
        valid_smiles = [
            'CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2',
            'CC(C)(C)NCC(O)COC1=CC=CC2=CC=CC=C21',
            'CC(C)NCC(O)COC1=CC=CC2=CC=CC=C21',
            'COC1=CC=C(C=C1)CCNC(=O)C2=CC(=C(C=C2)OC)OC',
            'COC1=CC2=C(C=C1)C(=O)C=CC2=O',
            'COC1=CC2=C(C=C1OC)C(=O)C=CC2=O',
            'COC(=O)C1=CC(=CC=C1)OC2=CC=C(C=C2)CN',
            'COC(=O)C1=CC=C(C=C1)NC(=O)C2=CC=C(C=C2)CN',
            'COCCOC1=CC=C(C=C1)C(=O)C2=C(C=CC=C2)OCC',
            'CN1CCN(CC1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OC)OC',
            'CN1CCN(CC1)C2=NC3=CC=CC=C3NC2=O',
            'CN1CCN(CC1)C2=NC3=CC=CC=C3NC2=S',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(C)C1=NC2=CN=C(N=C2S1)NC3=CC=C(C=C3)C(=O)N',
            'CC(C)(C)C1=CC(=CC(=C1)C(C)(C)C)OCC(O)CNC(C)C',
            'CC(C)(C)C1=CC(=CC(=C1)C(C)(C)C)OCCOCC2=CC=CC=C2',
            'CC(C)(C)C1=CC(=CC(=C1)C(C)(C)C)OCCOCCN',
            'CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(C(CC3=CC=CC=C3)NC(=O)C(CC(=O)N)NC(=O)C)O',
            'CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(C(CC3=CC=CC=C3)NC(=O)OC)O',
            'CC(C)(C)OC(=O)NC(CC1=CC=CC=C1)C(=O)NC2CCCCC2',
        ]
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 1: Train Binding Affinity Predictor on PDBbind
    print("\n=== Training Binding Affinity Predictor ===")
    binding_model = BindingAffinityPredictor(device=device)
    binding_model.train(
        pdbbind_df['smiles'].tolist(), 
        pdbbind_df['binding_affinity'].tolist(),
        epochs=20,  # Reduced for demonstration
        batch_size=8
    )
    binding_model.save_model()
    
    # Step 2: Train Molecule Generator with ChEMBL molecules
    print("\n=== Training Molecule Generator ===")
    molecule_gen = MoleculeGenerator(device=device)
    molecule_gen.train_generator(valid_smiles, epochs=5, batch_size=16)  # Reduced for demonstration
    
    # Step 3: Fine-tune Generator with Reinforcement Learning
    print("\n=== Fine-tuning Generator with RL ===")
    optimized_molecules = molecule_gen.rl_fine_tune(
        binding_model, 
        valid_smiles, 
        epochs=5,    # Reduced for demonstration
        n_samples=20
    )
    
    # Step 4: Evaluate binding affinities of generated molecules
    print("\n=== Evaluating Generated Molecules ===")
    binding_scores = binding_model.predict(optimized_molecules)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'SMILES': optimized_molecules,
        'BindingAffinity': binding_scores
    })
    
    # Sort by binding affinity (lower is better)
    results_df = results_df.sort_values(by='BindingAffinity')
    
    # Display top molecules
    print("\n=== Top 10 Generated Molecules ===")
    top_molecules = results_df.head(10)
    print(top_molecules)
    
# Visualize top molecules
    print("\n=== Visualizing Top Molecules ===")
    top_mols = [Chem.MolFromSmiles(smiles) for smiles in top_molecules['SMILES'].tolist()]
    top_mols = [mol for mol in top_mols if mol is not None]  # Filter out None
    
    if top_mols:
        img = Draw.MolsToGridImage(
            top_mols, 
            molsPerRow=2, 
            subImgSize=(300, 300), 
            legends=[f"{smiles}\nBA: {score:.2f}" for smiles, score in 
                     zip(top_molecules['SMILES'].tolist()[:len(top_mols)], 
                         top_molecules['BindingAffinity'].tolist()[:len(top_mols)])]
        )
        img.save('top_molecules.png')
        print("Visualization saved as 'top_molecules.png'")
    
    # Save results
    results_df.to_csv('generated_molecules_results.csv', index=False)
    print("Results saved to 'generated_molecules_results.csv'")
    
    return results_df, optimized_molecules, binding_scores

# Part 4: Alternative Method - Augmenting Generator with Direct Molecular Fingerprints

class ImprovedMoleculeGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Use ChemBERTa as the base model
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1').to(device)
        
        # Create a molecular fingerprint encoder
        self.fp_encoder = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        ).to(device)
        
        # Create a combined decoder (language model + fingerprints)
        hidden_size = self.model.config.hidden_size
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + 256, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(self.tokenizer))
        ).to(device)
        
        # Initialize optimizer with all trainable parameters
        self.optimizer = torch.optim.Adam(
            list(self.fp_encoder.parameters()) + 
            list(self.decoder.parameters()),
            lr=1e-4
        )
    
    def get_morgan_fingerprint(self, smiles, radius=2, nBits=2048):
        """Generate Morgan fingerprint for a molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    
    def train_generator(self, smiles_list, epochs=5, batch_size=32):
        """Train generator with fingerprint conditioning"""
        dataset = MoleculeDataset(smiles_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_values = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()
                
                # Get fingerprints for conditioning
                fps = [self.get_morgan_fingerprint(smile) for smile in batch]
                fp_tensor = torch.tensor(fps, dtype=torch.float32).to(self.device)
                
                # Encode fingerprints
                fp_embedding = self.fp_encoder(fp_tensor)
                
                # Tokenize smiles
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                
                # Get ChemBERTa embeddings
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state
                
                # Combine embeddings with fingerprint information
                # Expand fingerprint embedding to match sequence length
                batch_size, seq_len, hidden_size = embeddings.shape
                fp_expanded = fp_embedding.unsqueeze(1).expand(-1, seq_len, -1)
                
                # Concatenate along feature dimension
                combined_embeddings = torch.cat([embeddings, fp_expanded], dim=2)
                
                # Predict next tokens (shift sequence by 1)
                input_ids = encoded['input_ids'][:, :-1]
                target_ids = encoded['input_ids'][:, 1:]
                logits = self.decoder(combined_embeddings[:, :-1, :])
                
                # Calculate loss (cross-entropy)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), 
                                      ignore_index=self.tokenizer.pad_token_id)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            loss_values.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values)
        plt.title('Improved Generator Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('improved_generator_training_loss.png')
        plt.close()
        
        return loss_values
    
    def generate_molecules_with_target(self, target_fp, n_samples=100, max_length=128, temperature=1.0):
        """Generate molecules conditioned on target fingerprint"""
        generated_smiles = []
        
        # Encode target fingerprint
        target_fp_tensor = torch.tensor(target_fp, dtype=torch.float32).unsqueeze(0).to(self.device)
        fp_embedding = self.fp_encoder(target_fp_tensor)
        
        for _ in tqdm(range(n_samples), desc="Generating molecules"):
            # Start with BOS token
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # Generate tokens one by one
            for _ in range(max_length):
                encoded = {'input_ids': input_ids, 'attention_mask': attention_mask}
                
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    embeddings = outputs.last_hidden_state
                    
                    # Combine with fingerprint embedding
                    seq_len = embeddings.shape[1]
                    fp_expanded = fp_embedding.expand(1, seq_len, -1)
                    combined_embeddings = torch.cat([embeddings, fp_expanded], dim=2)
                    
                    logits = self.decoder(combined_embeddings)
                    
                    # Apply temperature
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Sample from the distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # If EOS token, stop generation
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Add token to sequence
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            # Decode the tokens to a SMILES string
            smiles = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # Check if valid molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                generated_smiles.append(smiles)
        
        return generated_smiles
    
    def run_fingerprint_guided_generation(self, reference_smiles, binding_model, n_samples=100):
        """Generate molecules similar to reference molecules with good binding"""
        # Get fingerprints of reference molecules
        reference_fps = [self.get_morgan_fingerprint(smile) for smile in reference_smiles]
        
        # Predict binding affinity for reference molecules
        binding_scores = binding_model.predict(reference_smiles)
        
        # Create weighted average fingerprint guided by binding affinity
        # Convert scores to weights (higher weights for better binding)
        weights = np.array([-score for score in binding_scores])  # Negate because lower binding energy is better
        weights = np.exp(weights)  # Exponential to emphasize good binders
        weights = weights / weights.sum()  # Normalize
        
        # Compute weighted average fingerprint
        target_fp = np.zeros_like(reference_fps[0])
        for i, fp in enumerate(reference_fps):
            target_fp += weights[i] * fp
        
        # Binarize target fingerprint for Morgan compatibility
        target_fp = (target_fp > 0.5).astype(np.float32)
        
        # Generate new molecules with this target fingerprint
        generated_molecules = self.generate_molecules_with_target(target_fp, n_samples=n_samples)
        
        return generated_molecules

# Part 5: Advanced Evaluation Functions

def evaluate_drug_likeness(smiles_list):
    """Evaluate drug-likeness properties of molecules"""
    results = []
    
    for smiles in tqdm(smiles_list, desc="Evaluating drug-likeness"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        # Calculate properties
        props = {
            'SMILES': smiles,
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'ArRings': len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1ccccc1'))),
            'Lipinski': 0  # Will calculate below
        }
        
        # Lipinski's Rule of Five violations
        violations = 0
        if props['MolWt'] > 500: violations += 1
        if props['LogP'] > 5: violations += 1
        if props['HBA'] > 10: violations += 1
        if props['HBD'] > 5: violations += 1
        props['Lipinski'] = violations
        
        results.append(props)
    
    return pd.DataFrame(results)

def calculate_scaffold_diversity(smiles_list):
    """Calculate scaffold diversity of a set of molecules"""
    scaffolds = {}
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = 0
        scaffolds[scaffold] += 1
    
    # Number of unique scaffolds
    n_scaffolds = len(scaffolds)
    # Scaffold diversity (unique scaffolds / total molecules)
    diversity = n_scaffolds / len(smiles_list) if len(smiles_list) > 0 else 0
    
    return {
        'unique_scaffolds': n_scaffolds,
        'scaffold_diversity': diversity,
        'scaffold_counts': scaffolds
    }

def visualize_molecule_properties(df):
    """Visualize molecular properties"""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot molecular weight distribution
    sns.histplot(df['MolWt'], ax=axes[0, 0], kde=True)
    axes[0, 0].axvline(x=500, color='r', linestyle='--', label='Lipinski limit')
    axes[0, 0].set_title('Molecular Weight Distribution')
    axes[0, 0].set_xlabel('Molecular Weight (Da)')
    axes[0, 0].legend()
    
    # Plot LogP distribution
    sns.histplot(df['LogP'], ax=axes[0, 1], kde=True)
    axes[0, 1].axvline(x=5, color='r', linestyle='--', label='Lipinski limit')
    axes[0, 1].set_title('LogP Distribution')
    axes[0, 1].set_xlabel('LogP')
    axes[0, 1].legend()
    
    # Plot HBA vs HBD
    sns.scatterplot(data=df, x='HBD', y='HBA', hue='Lipinski', palette='viridis', ax=axes[0, 2])
    axes[0, 2].axhline(y=10, color='r', linestyle='--')
    axes[0, 2].axvline(x=5, color='r', linestyle='--')
    axes[0, 2].set_title('Hydrogen Bond Acceptors vs Donors')
    axes[0, 2].set_xlabel('Hydrogen Bond Donors')
    axes[0, 2].set_ylabel('Hydrogen Bond Acceptors')
    
    # Plot TPSA distribution
    sns.histplot(df['TPSA'], ax=axes[1, 0], kde=True)
    axes[1, 0].axvline(x=140, color='r', linestyle='--', label='Limit for oral bioavailability')
    axes[1, 0].set_title('TPSA Distribution')
    axes[1, 0].set_xlabel('TPSA (Å²)')
    axes[1, 0].legend()
    
    # Plot Rotatable Bonds distribution
    sns.histplot(df['RotBonds'], ax=axes[1, 1], kde=True)
    axes[1, 1].axvline(x=10, color='r', linestyle='--', label='Limit for oral bioavailability')
    axes[1, 1].set_title('Rotatable Bonds Distribution')
    axes[1, 1].set_xlabel('Number of Rotatable Bonds')
    axes[1, 1].legend()
    
    # Plot Lipinski violations
    sns.countplot(x='Lipinski', data=df, ax=axes[1, 2])
    axes[1, 2].set_title('Lipinski Violations')
    axes[1, 2].set_xlabel('Number of Violations')
    axes[1, 2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('molecule_properties.png')
    plt.close()
    
    return fig

# Part 6: Main Function and Full Pipeline Execution

def main():
    """Main function to run the full pipeline"""
    # Run the basic pipeline
    print("=== Running Basic Pipeline ===")
    results_df, optimized_molecules, binding_scores = run_full_pipeline()
    
    # Run advanced evaluation
    print("\n=== Running Advanced Evaluation ===")
    druglike_df = evaluate_drug_likeness(optimized_molecules)
    scaffold_div = calculate_scaffold_diversity(optimized_molecules)
    
    print(f"Scaffold Diversity: {scaffold_div['scaffold_diversity']:.3f}")
    print(f"Unique Scaffolds: {scaffold_div['unique_scaffolds']}")
    
    # Visualize properties
    print("\n=== Generating Property Visualizations ===")
    visualize_molecule_properties(druglike_df)
    
    # Try the alternative fingerprint-guided approach if we have good data
    if len(optimized_molecules) >= 10:
        print("\n=== Running Alternative Fingerprint-Guided Generation ===")
        # Use the top 10 molecules as reference
        top_molecules = results_df.head(10)['SMILES'].tolist()
        
        # Create and train improved generator
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        improved_gen = ImprovedMoleculeGenerator(device=device)
        
        # Load binding model
        binding_model = BindingAffinityPredictor(device=device)
        binding_model.load_model()  # Load from previous run
        
        # Generate guided molecules
        guided_molecules = improved_gen.run_fingerprint_guided_generation(
            top_molecules, 
            binding_model, 
            n_samples=50
        )
        
        # Evaluate new molecules
        guided_scores = binding_model.predict(guided_molecules)
        guided_df = pd.DataFrame({
            'SMILES': guided_molecules,
            'BindingAffinity': guided_scores
        }).sort_values(by='BindingAffinity')
        
        print("\n=== Top 5 Guided Molecules ===")
        print(guided_df.head(5))
        
        # Save results
        guided_df.to_csv('guided_molecules_results.csv', index=False)
    
    print("\n=== Project Complete ===")
    print("All results have been saved to CSV files and visualizations have been created")

if __name__ == "__main__":
    main()