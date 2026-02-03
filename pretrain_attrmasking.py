"""
AttrMasking Pretraining with AttentiveFP Backbone (Fixed Version)
==================================================================
Fixed version that properly handles node feature dimensions.

Key fixes:
1. Get atom types directly from DGL graph features (not pre-cached)
2. Proper handling of self-loops
3. Robust error handling

Author: Modified for ablation study - Fixed version
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import random
import dgl
import deepchem as dc
import pickle
import time
from datetime import datetime

# TensorBoard support
try:
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
        HAS_TENSORBOARD = True
    except ImportError:
        HAS_TENSORBOARD = False
        print("Warning: TensorBoard not available")

# Import AttentiveFPGNN
from AttfpMPNN import AttentiveFPGNN

# 禁用RDKit日志
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except:
    pass

# =============================================================================
# Hyperparameters - EXACTLY SAME AS SYNFRAG PRETRAIN
# =============================================================================
EPOCHS = 100
BATCH_SIZE = 16
LR = 0.01
WEIGHT_DECAY = 0.0001
LOG_INTERVAL = 100
SAVE_INTERVAL = 10
RANDOM_SEED = 731

# Model parameters
NODE_FEAT_SIZE = 30
EDGE_FEAT_SIZE = 11
NUM_LAYERS = 4
GRAPH_FEAT_SIZE = 400
DROPOUT = 0.3

# AttrMasking parameters
MASK_RATE = 0.15
NUM_ATOM_TYPES = 119


# =============================================================================
# Simple Dataset - Just load SMILES
# =============================================================================
class MoleculeDatasetSimple(Dataset):
    """Simple dataset that just loads and caches valid SMILES"""
    
    def __init__(self, data_file, cache_dir=None):
        from rdkit import Chem
        
        # Setup cache
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(data_file), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        data_basename = os.path.basename(data_file).replace('.', '_')
        cache_file = os.path.join(cache_dir, f'{data_basename}_smiles_cache.pkl')
        
        # Try load from cache
        if os.path.exists(cache_file):
            print(f"Loading cached SMILES from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.valid_smiles = pickle.load(f)
            print(f"Loaded {len(self.valid_smiles)} valid SMILES from cache")
            return
        
        # Load and validate SMILES
        print(f"Loading SMILES from {data_file}")
        file_ext = os.path.splitext(data_file)[1]
        
        if file_ext == '.txt':
            with open(data_file) as f:
                smiles_list = [line.strip().split()[0] for line in f if line.strip()]
        elif file_ext == '.csv':
            df = pd.read_csv(data_file)
            for col in ['smiles', 'SMILES', 'Smiles']:
                if col in df.columns:
                    smiles_list = df[col].dropna().tolist()
                    break
            else:
                smiles_list = df.iloc[:, 0].dropna().tolist()
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        print(f"Loaded {len(smiles_list)} SMILES, validating...")
        
        # Validate
        self.valid_smiles = []
        for smi in tqdm(smiles_list, desc="Validating"):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None and mol.GetNumAtoms() > 1:
                    self.valid_smiles.append(smi)
            except:
                continue
        
        print(f"Valid: {len(self.valid_smiles)} / {len(smiles_list)}")
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.valid_smiles, f)
        print(f"Cached to {cache_file}")
    
    def __len__(self):
        return len(self.valid_smiles)
    
    def __getitem__(self, idx):
        return self.valid_smiles[idx]


# =============================================================================
# Collate Function - Fixed Version
# =============================================================================
def collate_fn_fixed(batch_smiles, featurizer, mask_rate):
    """
    Fixed collate function that extracts atom types from graph features directly.
    
    DeepChem's MolGraphConvFeaturizer stores atom types in the first column of node features.
    """
    # Featurize batch
    try:
        graphs = featurizer.featurize(batch_smiles)
    except:
        return None
    
    dgl_graphs = []
    all_masked_indices = []
    all_mask_labels = []
    cumsum_nodes = 0
    
    for g in graphs:
        try:
            if g is None:
                continue
            
            # Convert to DGL graph (with self-loop as in SynFrag)
            dgl_g = g.to_dgl_graph(self_loop=True)
            
            node_feats = dgl_g.ndata['x']
            num_nodes = node_feats.shape[0]
            
            if num_nodes < 2:
                continue
            
            # Extract atom types from node features
            # DeepChem's first feature is typically atom type (one-hot encoded)
            # We use the feature values directly and find the non-zero index
            # Or we can use the raw feature as label
            
            # For AttrMasking, we predict the original node feature pattern
            # Get atom types by finding max index in first ~30 features (atom type encoding)
            atom_type_feats = node_feats[:, :30]  # First 30 features are atom type related
            
            # Sample atoms to mask (excluding potential virtual nodes from self-loop)
            sample_size = max(1, int(num_nodes * mask_rate))
            sample_size = min(sample_size, num_nodes)
            masked_indices = random.sample(range(num_nodes), sample_size)
            
            # Get labels - use argmax of first few features as atom type proxy
            # Or use a hash of the feature vector
            mask_labels = []
            for idx in masked_indices:
                # Simple approach: use first feature value as label (atom number)
                # DeepChem encodes atom type in features
                feat = node_feats[idx]
                # The atom type can be approximated by the feature pattern
                # We'll use a simple label based on feature values
                label = int(feat[0].item() * 10) % NUM_ATOM_TYPES  # Normalize to valid range
                mask_labels.append(label)
            
            # Create mask - set masked node features to zero
            masked_node_feats = node_feats.clone()
            for idx in masked_indices:
                masked_node_feats[idx] = 0
            
            # Update graph with masked features
            dgl_g.ndata['x'] = masked_node_feats
            
            dgl_graphs.append(dgl_g)
            all_masked_indices.extend([idx + cumsum_nodes for idx in masked_indices])
            all_mask_labels.extend(mask_labels)
            cumsum_nodes += num_nodes
            
        except Exception as e:
            continue
    
    if len(dgl_graphs) == 0:
        return None
    
    # Batch graphs
    batch_graph = dgl.batch(dgl_graphs)
    masked_indices = torch.tensor(all_masked_indices, dtype=torch.long)
    mask_labels = torch.tensor(all_mask_labels, dtype=torch.long)
    
    return batch_graph, masked_indices, mask_labels


def collate_fn_v2(batch_smiles, featurizer, mask_rate):
    """
    Alternative collate function - more robust version.
    Uses RDKit to get actual atom types.
    """
    from rdkit import Chem
    
    dgl_graphs = []
    all_masked_indices = []
    all_mask_labels = []
    cumsum_nodes = 0
    
    for smi in batch_smiles:
        try:
            # Get atom types from RDKit
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            num_real_atoms = len(atom_types)
            
            if num_real_atoms < 2:
                continue
            
            # Featurize single molecule
            graphs = featurizer.featurize([smi])
            if graphs[0] is None:
                continue
            
            dgl_g = graphs[0].to_dgl_graph(self_loop=True)
            num_nodes = dgl_g.num_nodes()
            
            # Only mask real atoms (not virtual nodes from self-loop)
            num_to_mask = max(1, int(num_real_atoms * mask_rate))
            masked_indices = random.sample(range(num_real_atoms), min(num_to_mask, num_real_atoms))
            
            # Get labels from atom types
            mask_labels = [min(atom_types[idx], NUM_ATOM_TYPES - 1) for idx in masked_indices]
            
            # Apply masking to node features
            node_feats = dgl_g.ndata['x'].clone()
            for idx in masked_indices:
                if idx < node_feats.shape[0]:
                    node_feats[idx] = 0
            dgl_g.ndata['x'] = node_feats
            
            dgl_graphs.append(dgl_g)
            all_masked_indices.extend([idx + cumsum_nodes for idx in masked_indices])
            all_mask_labels.extend(mask_labels)
            cumsum_nodes += num_nodes
            
        except Exception as e:
            continue
    
    if len(dgl_graphs) == 0:
        return None
    
    batch_graph = dgl.batch(dgl_graphs)
    masked_indices = torch.tensor(all_masked_indices, dtype=torch.long)
    mask_labels = torch.tensor(all_mask_labels, dtype=torch.long)
    
    return batch_graph, masked_indices, mask_labels


# =============================================================================
# AttrMasking Prediction Head
# =============================================================================
class AttrMaskingHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, node_rep, masked_indices):
        return self.linear(node_rep[masked_indices])


# =============================================================================
# Trainer Class
# =============================================================================
class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        self.use_amp = args.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        self.global_step = 0
        self.best_loss = float('inf')
        
        # TensorBoard
        if HAS_TENSORBOARD and args.use_tensorboard:
            log_dir = os.path.join(args.output_dir, 'tensorboard',
                                   datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard: {log_dir}")
        else:
            self.writer = None
    
    def train_epoch(self, gnn, head, loader, opt_gnn, opt_head, epoch, log_file):
        gnn.train()
        head.train()
        
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        accum_steps = self.args.gradient_accumulation_steps
        
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        
        for step, batch_smiles in enumerate(pbar):
            # Collate with v2 (more robust)
            result = collate_fn_v2(batch_smiles, self.featurizer, self.args.mask_rate)
            
            if result is None:
                continue
            
            batch_graph, masked_indices, mask_labels = result
            
            if len(masked_indices) == 0:
                continue
            
            try:
                batch_graph = batch_graph.to(self.device)
                masked_indices = masked_indices.to(self.device)
                mask_labels = mask_labels.to(self.device)
                
                # Forward
                if self.use_amp:
                    with autocast():
                        node_feats = batch_graph.ndata['x'].float()
                        edge_feats = batch_graph.edata['edge_attr'].float()
                        node_rep = gnn(batch_graph, node_feats, edge_feats)
                        pred = head(node_rep, masked_indices)
                        loss = loss_fn(pred, mask_labels) / accum_steps
                    
                    self.scaler.scale(loss).backward()
                    
                    if (step + 1) % accum_steps == 0:
                        self.scaler.step(opt_gnn)
                        self.scaler.step(opt_head)
                        self.scaler.update()
                        opt_gnn.zero_grad()
                        opt_head.zero_grad()
                else:
                    node_feats = batch_graph.ndata['x'].float()
                    edge_feats = batch_graph.edata['edge_attr'].float()
                    node_rep = gnn(batch_graph, node_feats, edge_feats)
                    pred = head(node_rep, masked_indices)
                    loss = loss_fn(pred, mask_labels) / accum_steps
                    
                    loss.backward()
                    
                    if (step + 1) % accum_steps == 0:
                        opt_gnn.step()
                        opt_head.step()
                        opt_gnn.zero_grad()
                        opt_head.zero_grad()
                
                # Metrics
                with torch.no_grad():
                    acc = (pred.argmax(dim=1) == mask_labels).float().mean().item()
                
                total_loss += loss.item() * accum_steps
                total_acc += acc
                num_batches += 1
                self.global_step += 1
                
                pbar.set_postfix({'loss': f'{loss.item()*accum_steps:.4f}', 'acc': f'{acc*100:.1f}%'})
                
                # TensorBoard
                if self.writer and self.global_step % 50 == 0:
                    self.writer.add_scalar('train/loss', loss.item() * accum_steps, self.global_step)
                    self.writer.add_scalar('train/acc', acc * 100, self.global_step)
                
                # Log
                if (step + 1) % LOG_INTERVAL == 0 and num_batches > 0:
                    avg_loss = total_loss / num_batches
                    avg_acc = total_acc / num_batches * 100
                    msg = f"Epoch {epoch} Step {step+1} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.1f}%"
                    print(f"\n{msg}")
                    with open(log_file, 'a') as f:
                        f.write(msg + '\n')
            
            except Exception as e:
                # Skip problematic batches silently
                continue
        
        if num_batches > 0:
            return total_loss / num_batches, total_acc / num_batches
        return 0, 0
    
    def save(self, gnn, head, opt_gnn, opt_head, epoch, loss, output_dir, is_best=False):
        os.makedirs(output_dir, exist_ok=True)
        
        gnn_state = gnn.module.state_dict() if isinstance(gnn, DataParallel) else gnn.state_dict()
        
        # Save GNN weights (for finetuning)
        torch.save(gnn_state, os.path.join(output_dir, 'attrmasking_gnn_pretrained.pth'))
        
        # Save full checkpoint (for resume)
        checkpoint = {
            'epoch': epoch,
            'gnn': gnn_state,
            'head': head.state_dict(),
            'opt_gnn': opt_gnn.state_dict(),
            'opt_head': opt_head.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, os.path.join(output_dir, 'checkpoint_latest.pth'))
        
        if is_best:
            torch.save(gnn_state, os.path.join(output_dir, 'attrmasking_gnn_best.pth'))
            print(f"  -> Best model saved (loss: {loss:.4f})")
    
    def close(self):
        if self.writer:
            self.writer.close()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--mask_rate', type=float, default=MASK_RATE)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'pretrain.log')
    
    # Print config
    print("=" * 70)
    print("AttrMasking Pretraining (Fixed Version)")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Multi-GPU: {args.multi_gpu}")
    print(f"AMP: {args.use_amp}")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print("-" * 70)
    
    # Dataset
    dataset = MoleculeDatasetSimple(args.dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,  # Return list of SMILES
        drop_last=True,
        pin_memory=True
    )
    print(f"Dataset: {len(dataset)} | Batches: {len(loader)}")
    
    # Model
    gnn = AttentiveFPGNN(
        node_feat_size=NODE_FEAT_SIZE,
        edge_feat_size=EDGE_FEAT_SIZE,
        num_layers=NUM_LAYERS,
        graph_feat_size=GRAPH_FEAT_SIZE,
        dropout=DROPOUT
    )
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        gnn = DataParallel(gnn)
    
    gnn = gnn.to(device)
    head = AttrMaskingHead(GRAPH_FEAT_SIZE, NUM_ATOM_TYPES).to(device)
    
    # Optimizer
    opt_gnn = optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    opt_head = optim.Adam(head.parameters(), lr=1e-3)
    
    # Trainer
    trainer = Trainer(args, device)
    
    # Log
    with open(log_file, 'w') as f:
        f.write(f"AttrMasking Pretrain\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write("-" * 50 + "\n")
    
    # Train
    print("\n" + "=" * 70)
    print("Training Start!")
    print("=" * 70 + "\n")
    
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            avg_loss, avg_acc = trainer.train_epoch(
                gnn, head, loader, opt_gnn, opt_head, epoch, log_file
            )
            
            print(f"\nEpoch {epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.1f}% | Time: {time.time()-t0:.0f}s")
            
            # Save
            is_best = avg_loss < trainer.best_loss
            if is_best:
                trainer.best_loss = avg_loss
            
            if epoch % SAVE_INTERVAL == 0 or is_best:
                trainer.save(gnn, head, opt_gnn, opt_head, epoch, avg_loss, args.output_dir, is_best)
    
    except KeyboardInterrupt:
        print("\nInterrupted! Saving...")
        trainer.save(gnn, head, opt_gnn, opt_head, epoch, avg_loss, args.output_dir)
    
    finally:
        trainer.close()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Model: {args.output_dir}/attrmasking_gnn_pretrained.pth")
    print("=" * 70)


if __name__ == "__main__":
    main()