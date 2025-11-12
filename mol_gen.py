import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors
import re
import os
import gc

class Config:
    data_path = "BindingDB.csv"
    max_protein_length = 512 
    max_smiles_length = 128
    min_affinity = 1000  # nM

    protein_embed_dim = 320  
    smiles_embed_dim = 768  
    cross_attention_dim = 256  
    num_cross_attention_heads = 4  

    batch_size = 16  
    learning_rate = 5e-5
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_offload = True 

    output_dir = "./saved_models/"
    pretrained_esm_model = "facebook/esm2_t6_8M_UR50D"  # Much smaller ESM model
    pretrained_gpt2_model = "gpt2"

config = Config()

class DTIDataset(Dataset):
    def __init__(self, protein_sequences, smiles, tokenizer_protein, tokenizer_smiles):
        self.protein_sequences = protein_sequences
        self.smiles = smiles
        self.tokenizer_protein = tokenizer_protein
        self.tokenizer_smiles = tokenizer_smiles

    def __len__(self):
        return len(self.protein_sequences)

    def __getitem__(self, idx):
        protein_seq = self.protein_sequences[idx]
        smile = self.smiles[idx]

        protein_tokens = self.tokenizer_protein(
            protein_seq,
            max_length=config.max_protein_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        smile_tokens = self.tokenizer_smiles(
            smile,
            max_length=config.max_smiles_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'protein_input_ids': protein_tokens['input_ids'].squeeze(0),
            'protein_attention_mask': protein_tokens['attention_mask'].squeeze(0),
            'smiles_input_ids': smile_tokens['input_ids'].squeeze(0),
            'smiles_attention_mask': smile_tokens['attention_mask'].squeeze(0)
        }

def load_and_preprocess_data():
    print(f"Loading data from {config.data_path}")
    df = pd.read_csv(config.data_path)
    print(f"Available columns: {df.columns.tolist()}")
    protein_col = 'Seq'
    smiles_col = 'SMILES'

    if protein_col not in df.columns:
        raise ValueError(f"The column '{protein_col}' is not in the dataset. Available columns: {df.columns.tolist()}")

    if smiles_col not in df.columns:
        raise ValueError(f"The column '{smiles_col}' is not in the dataset. Available columns: {df.columns.tolist()}")

    if 'Affinity' in df.columns:
        df = df[df['Affinity'] <= config.min_affinity]
    else:
        print("Warning: 'affinity' column not found. Skipping affinity filtering.")

    # Filter by protein sequence length
    df = df[df[protein_col].str.len() <= config.max_protein_length]

    def is_valid_smiles(s):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return False
            if Descriptors.MolWt(mol) > 600:  # Molecular weight filter
                return False
            if Descriptors.NumHAcceptors(mol) > 10:  # HBA filter
                return False
            if Descriptors.NumHDonors(mol) > 5:  # HBD filter
                return False
            return True
        except:
            return False

    df = df[df[smiles_col].apply(is_valid_smiles)]

    if len(df) > 10000:
        print(f"Dataset is large ({len(df)} entries). Taking a random sample of 10000.")
        df = df.sample(10000, random_state=42)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}")

    return train_df, val_df

class ProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm = AutoModel.from_pretrained(config.pretrained_esm_model)

        # Freeze all layers except the last
        for param in self.esm.parameters():
            param.requires_grad = False

        if hasattr(self.esm, 'encoder') and hasattr(self.esm.encoder, 'layer'):
            for param in self.esm.encoder.layer[-1].parameters():
                param.requires_grad = True

        self.projection = nn.Linear(
            config.protein_embed_dim,
            config.cross_attention_dim
        )

    def forward(self, input_ids, attention_mask):
            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_states = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)

            projected = self.projection(pooled)

            return projected

class CrossAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.smiles_embed_dim,
            kdim=config.cross_attention_dim,
            vdim=config.cross_attention_dim,
            num_heads=config.num_cross_attention_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(config.smiles_embed_dim)

    def forward(self, x, protein_embed, protein_embed_mask=None):

        protein_embed = protein_embed.unsqueeze(1)

        attn_output, _ = self.cross_attn(
            query=x,
            key=protein_embed,
            value=protein_embed,
            key_padding_mask=protein_embed_mask
        )

        return self.norm(x + attn_output)

class DrugDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.pretrained_gpt2_model)

        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer() for _ in range(2)  
        ])
        self.use_layers = [6, 11]  

    def forward(self, input_ids, attention_mask, protein_embed):
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = list(outputs.hidden_states)

        # Applying cross-attention only to selected layers
        for i, layer_idx in enumerate(self.use_layers):
            if i < len(self.cross_attention_layers):
                hidden_states[layer_idx+1] = self.cross_attention_layers[i](
                    hidden_states[layer_idx+1], protein_embed
                )

        # Use last hidden state for prediction
        last_hidden_state = hidden_states[-1]
        logits = self.gpt2.lm_head(last_hidden_state)

        return logits

class ProDrugGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.protein_encoder = ProteinEncoder()
        self.drug_decoder = DrugDecoder()

    def forward(self, protein_input_ids, protein_attention_mask,
                smiles_input_ids, smiles_attention_mask):
        # Use CPU offloading if configured
        if config.cpu_offload and hasattr(torch, 'cuda') and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                protein_embed = self.protein_encoder(
                    protein_input_ids,
                    protein_attention_mask
                )

                logits = self.drug_decoder(
                    smiles_input_ids,
                    smiles_attention_mask,
                    protein_embed
                )
        else:
            protein_embed = self.protein_encoder(
                protein_input_ids,
                protein_attention_mask
            )

            logits = self.drug_decoder(
                smiles_input_ids,
                smiles_attention_mask,
                protein_embed
            )

        return logits

    def generate(self, protein_input_ids, protein_attention_mask,
                 max_length=128, temperature=1.0, top_k=50):
        """Generate SMILES conditioned on protein sequence"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        protein_embed = self.protein_encoder(
            protein_input_ids,
            protein_attention_mask
        )

        device = protein_input_ids.device
        batch_size = protein_input_ids.size(0)

        input_ids = torch.tensor(
            [[config.tokenizer_smiles.bos_token_id]],
            device=device
        ).repeat(batch_size, 1)

        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.drug_decoder(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    protein_embed=protein_embed
                )

            next_token_logits = outputs[:, -1, :] / temperature
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                indices_to_remove = next_token_logits < top_k_logits[:, -1].unsqueeze(-1)
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Stop if EOS token is generated
            if (next_tokens == config.tokenizer_smiles.eos_token_id).any():
                break

        return input_ids

def train_model():
    accumulation_steps = 4  # Effective batch size = config.batch_size * accumulation_steps

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_df, val_df = load_and_preprocess_data()

    tokenizer_protein = AutoTokenizer.from_pretrained(config.pretrained_esm_model)
    tokenizer_smiles = GPT2Tokenizer.from_pretrained(config.pretrained_gpt2_model)

    tokenizer_smiles.add_special_tokens({
        'pad_token': '[PAD]',
        'bos_token': '[BOS]',
        'eos_token': '[EOS]'
    })

    config.tokenizer_smiles = tokenizer_smiles

    train_dataset = DTIDataset(
        train_df['Seq'].values,
        train_df['SMILES'].values,
        tokenizer_protein,
        tokenizer_smiles
    )
    val_dataset = DTIDataset(
        val_df['Seq'].values,
        val_df['SMILES'].values,
        tokenizer_protein,
        tokenizer_smiles
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing to save memory
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = ProDrugGen().to(config.device)
    # Resizing token embeddings for the drug decoder
    model.drug_decoder.gpt2.resize_token_embeddings(len(tokenizer_smiles))

    if hasattr(model.protein_encoder.esm, 'gradient_checkpointing_enable'):
        model.protein_encoder.esm.gradient_checkpointing_enable()
    if hasattr(model.drug_decoder.gpt2, 'gradient_checkpointing_enable'):
        model.drug_decoder.gpt2.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_smiles.pad_token_id)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5
    )

    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            protein_input_ids = batch['protein_input_ids'].to(config.device)
            protein_attention_mask = batch['protein_attention_mask'].to(config.device)
            smiles_input_ids = batch['smiles_input_ids'].to(config.device)
            smiles_attention_mask = batch['smiles_attention_mask'].to(config.device)

            logits = model(
                    protein_input_ids,
                    protein_attention_mask,
                    smiles_input_ids,
                    smiles_attention_mask
                )

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = smiles_input_ids[..., 1:].contiguous()
            loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            loss = loss / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

                if torch.cuda.is_available() and (batch_idx + 1) % (accumulation_steps * 5) == 0:
                    torch.cuda.empty_cache()

            train_loss += loss.item() * accumulation_steps

            # Print batch stats
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()*accumulation_steps:.4f}")

            del protein_input_ids, protein_attention_mask, smiles_input_ids, smiles_attention_mask, logits
            torch.cuda.empty_cache()

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                protein_input_ids = batch['protein_input_ids'].to(config.device)
                protein_attention_mask = batch['protein_attention_mask'].to(config.device)
                smiles_input_ids = batch['smiles_input_ids'].to(config.device)
                smiles_attention_mask = batch['smiles_attention_mask'].to(config.device)

                logits = model(
                    protein_input_ids,
                    protein_attention_mask,
                    smiles_input_ids,
                    smiles_attention_mask
                )

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = smiles_input_ids[..., 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                val_loss += loss.item()

                # Free memory
                del protein_input_ids, protein_attention_mask, smiles_input_ids, smiles_attention_mask, logits
                torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Update learning rate
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(config.output_dir):
                os.makedirs(config.output_dir)
            torch.save(model.state_dict(), f"{config.output_dir}/best_model.pt")
            print("Saved best model")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model

def evaluate_generated_molecules(generated_smiles, protein_sequence=None):
    """Evaluate generated SMILES strings"""
    metrics = {
        'validity': 0,
        'unique': 0,
        'novelty': 0,
        'drug_likeness': 0
    }

    valid_smiles = []
    unique_smiles = set()

    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            metrics['validity'] += 1
            valid_smiles.append(smi)

            canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            if canonical_smi not in unique_smiles:
                metrics['unique'] += 1
                unique_smiles.add(canonical_smi)

            qed = Descriptors.qed(mol)
            if qed > 0.5:  # Threshold for drug-likeness
                metrics['drug_likeness'] += 1

    total = len(generated_smiles)
    metrics['validity'] = metrics['validity'] / total * 100 if total > 0 else 0
    metrics['unique'] = metrics['unique'] / total * 100 if total > 0 else 0
    metrics['drug_likeness'] = metrics['drug_likeness'] / total * 100 if total > 0 else 0

    return metrics

def generate_molecules(model, protein_seq, num_samples=5):
    """Generate molecules for a given protein sequence"""
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer_protein = AutoTokenizer.from_pretrained(config.pretrained_esm_model)
    protein_inputs = tokenizer_protein(
        protein_seq,
        max_length=config.max_protein_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(config.device)

    batch_size = 2
    generated_smiles = []

    for i in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - i)

        batch_protein_inputs = {
            'input_ids': protein_inputs['input_ids'].repeat(current_batch, 1),
            'attention_mask': protein_inputs['attention_mask'].repeat(current_batch, 1)
        }

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                batch_protein_inputs['input_ids'],
                batch_protein_inputs['attention_mask'],
                max_length=config.max_smiles_length,
                temperature=0.8,
                top_k=40
            )

        # Decode
        tokenizer_smiles = GPT2Tokenizer.from_pretrained(config.pretrained_gpt2_model)
        tokenizer_smiles.add_special_tokens({
            'pad_token': '[PAD]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]'
        })

        batch_smiles = tokenizer_smiles.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        batch_smiles = [re.sub(r'\[EOS\].*', '', smi).strip() for smi in batch_smiles]
        generated_smiles.extend(batch_smiles)

        del batch_protein_inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generated_smiles

if __name__ == "__main__":
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Enabling PyTorch memory allocation config to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    print("Starting training...")
    trained_model = train_model()

    # sample protein sequence
    protein_seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAK"

    print("Generating molecules...")
    generated_smiles = generate_molecules(trained_model, protein_seq, num_samples=5)

    output_csv_path = os.path.join(config.output_dir, "generated_molecules.csv")
    df_smiles = pd.DataFrame(generated_smiles, columns=["SMILES"])
    df_smiles.to_csv(output_csv_path, index=False)

    print(f"\nSaved {len(generated_smiles)} generated molecules to {output_csv_path}")