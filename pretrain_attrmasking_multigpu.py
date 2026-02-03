"""
AttrMasking Pretraining with AttentiveFP - Multi-GPU Version (Fixed)
=====================================================================
Fixed issues:
1. DeepChem multiprocessing conflict (JSONDecodeError)
2. DGL API compatibility (copy_edge -> copy_e)
3. Proper distributed training setup

Usage:
    # Single GPU
    python pretrain_attrmasking_multigpu.py --dataset data.txt --device 0
    
    # Multi-GPU (8 GPUs) - Use torchrun
    torchrun --nproc_per_node=8 pretrain_attrmasking_multigpu.py \
        --dataset data.txt --distributed --use_amp --batch_size 32
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import random
import pickle
import time
from datetime import datetime

# TensorBoard - lazy import
HAS_TB = False
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except:
    pass

# RDKit logging - safe import
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except:
    pass

# =============================================================================
# Hyperparameters (Same as SynFrag)
# =============================================================================
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.01
WEIGHT_DECAY = 0.0001
LOG_INTERVAL = 100
SAVE_INTERVAL = 10
RANDOM_SEED = 731

NODE_FEAT_SIZE = 30
EDGE_FEAT_SIZE = 11
NUM_LAYERS = 4
GRAPH_FEAT_SIZE = 400
DROPOUT = 0.3
MASK_RATE = 0.15
NUM_ATOM_TYPES = 119


# =============================================================================
# Lazy imports to avoid multiprocessing conflicts
# =============================================================================
_featurizer = None
_dgl = None

def get_featurizer():
    """Lazy initialization of DeepChem featurizer to avoid multiprocessing conflicts"""
    global _featurizer
    if _featurizer is None:
        import deepchem as dc
        _featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    return _featurizer

def get_dgl():
    """Lazy import of DGL"""
    global _dgl
    if _dgl is None:
        import dgl
        _dgl = dgl
    return _dgl


# =============================================================================
# Dataset - Only loads SMILES strings
# =============================================================================
class MoleculeDataset(Dataset):
    def __init__(self, data_file, cache_dir=None):
        from rdkit import Chem
        
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(data_file), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f'{os.path.basename(data_file).replace(".", "_")}_v3.pkl')
        
        if os.path.exists(cache_file):
            print(f"Loading cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.smiles_list = pickle.load(f)
            print(f"Loaded {len(self.smiles_list)} molecules")
            return
        
        # Load SMILES
        ext = os.path.splitext(data_file)[1]
        if ext == '.txt':
            with open(data_file) as f:
                raw_smiles = [l.strip().split()[0] for l in f if l.strip()]
        else:
            df = pd.read_csv(data_file)
            col = next((c for c in ['smiles', 'SMILES', 'Smiles'] if c in df.columns), df.columns[0])
            raw_smiles = df[col].dropna().tolist()
        
        print(f"Validating {len(raw_smiles)} molecules...")
        self.smiles_list = []
        for smi in tqdm(raw_smiles, desc="Validating"):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol and mol.GetNumAtoms() > 1:
                    self.smiles_list.append(smi)
            except:
                pass
        
        print(f"Valid: {len(self.smiles_list)}")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.smiles_list, f)
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        return self.smiles_list[idx]


# =============================================================================
# Collate Function - Featurizes on-the-fly
# =============================================================================
def collate_fn(batch_smiles, mask_rate=0.15):
    """
    Collate function that featurizes SMILES on-the-fly.
    Uses lazy import to avoid multiprocessing conflicts.
    """
    from rdkit import Chem
    
    dgl = get_dgl()
    featurizer = get_featurizer()
    
    dgl_graphs = []
    masked_indices = []
    mask_labels = []
    node_offset = 0
    
    for smi in batch_smiles:
        try:
            # Get atom types from RDKit
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            n_atoms = len(atom_types)
            
            if n_atoms < 2:
                continue
            
            # Featurize with DeepChem
            graphs = featurizer.featurize([smi])
            if graphs[0] is None:
                continue
            
            g = graphs[0].to_dgl_graph(self_loop=True)
            n_nodes = g.num_nodes()
            
            # Sample atoms to mask (only real atoms, not virtual nodes)
            n_mask = max(1, int(n_atoms * mask_rate))
            mask_idx = random.sample(range(n_atoms), min(n_mask, n_atoms))
            
            # Get labels (atom types)
            labels = [min(atom_types[i], NUM_ATOM_TYPES - 1) for i in mask_idx]
            
            # Zero out masked node features
            feats = g.ndata['x'].clone()
            for i in mask_idx:
                if i < feats.shape[0]:
                    feats[i] = 0
            g.ndata['x'] = feats
            
            dgl_graphs.append(g)
            masked_indices.extend([i + node_offset for i in mask_idx])
            mask_labels.extend(labels)
            node_offset += n_nodes
            
        except Exception as e:
            continue
    
    if not dgl_graphs:
        return None
    
    return (
        dgl.batch(dgl_graphs),
        torch.tensor(masked_indices, dtype=torch.long),
        torch.tensor(mask_labels, dtype=torch.long)
    )


# =============================================================================
# Model Components
# =============================================================================
class MaskHead(nn.Module):
    """Prediction head for masked atom type prediction"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, idx):
        return self.fc(x[idx])


# =============================================================================
# AttentiveFP GNN (Inline to avoid import issues)
# =============================================================================
def create_gnn(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout):
    """Create AttentiveFPGNN model"""
    from AttfpMPNN import AttentiveFPGNN
    return AttentiveFPGNN(
        node_feat_size=node_feat_size,
        edge_feat_size=edge_feat_size,
        num_layers=num_layers,
        graph_feat_size=graph_feat_size,
        dropout=dropout
    )


# =============================================================================
# Training Function
# =============================================================================
def train_epoch(gnn, head, loader, opt, device, scaler, epoch, writer, rank, mask_rate):
    gnn.train()
    head.train()
    
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_acc = 0
    n_batch = 0
    
    # Only show progress bar on main process
    pbar = tqdm(loader, desc=f'Epoch {epoch}', disable=(rank != 0))
    
    for step, batch_smiles in enumerate(pbar):
        # Collate batch (featurization happens here)
        batch = collate_fn(batch_smiles, mask_rate)
        
        if batch is None:
            continue
        
        graph, mask_idx, labels = batch
        
        if len(mask_idx) == 0:
            continue
        
        try:
            graph = graph.to(device)
            mask_idx = mask_idx.to(device)
            labels = labels.to(device)
            
            opt.zero_grad()
            
            if scaler:  # Mixed precision
                with autocast():
                    node_feats = graph.ndata['x'].float()
                    edge_feats = graph.edata['edge_attr'].float()
                    node_rep = gnn(graph, node_feats, edge_feats)
                    pred = head(node_rep, mask_idx)
                    loss = loss_fn(pred, labels)
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                node_feats = graph.ndata['x'].float()
                edge_feats = graph.edata['edge_attr'].float()
                node_rep = gnn(graph, node_feats, edge_feats)
                pred = head(node_rep, mask_idx)
                loss = loss_fn(pred, labels)
                
                loss.backward()
                opt.step()
            
            # Compute accuracy
            with torch.no_grad():
                acc = (pred.argmax(1) == labels).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            n_batch += 1
            
            # Update progress bar
            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc*100:.1f}%'})
                
                # TensorBoard logging
                if writer and step % 50 == 0:
                    global_step = (epoch - 1) * len(loader) + step
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    writer.add_scalar('train/acc', acc * 100, global_step)
        
        except Exception as e:
            if rank == 0 and n_batch < 5:  # Only print first few errors
                print(f"Batch error: {e}")
            continue
    
    return total_loss / max(n_batch, 1), total_acc / max(n_batch, 1)


# =============================================================================
# Distributed Setup
# =============================================================================
def setup_distributed():
    """Setup distributed training environment"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = local_rank
        world_size = torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group('nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Main Function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='AttrMasking Pretraining')
    parser.add_argument('--dataset', type=str, required=True, help='Path to SMILES file')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (single GPU mode)')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--mask_rate', type=float, default=MASK_RATE)
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 for main process)')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--use_tensorboard', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    # Setup distributed training
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    is_main = (rank == 0)
    
    # Set random seeds
    seed = RANDOM_SEED + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    
    # Print configuration (main process only)
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 70)
        print("AttrMasking Pretraining with AttentiveFP")
        print("=" * 70)
        print(f"Dataset: {args.dataset}")
        print(f"Device: {device}")
        print(f"Distributed: {args.distributed} (world_size={world_size})")
        print(f"Mixed Precision (AMP): {args.use_amp}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print("-" * 70)
    
    # Synchronize before loading data
    if args.distributed:
        dist.barrier()
    
    # Load dataset (main process creates cache, others wait)
    if is_main:
        dataset = MoleculeDataset(args.dataset)
        if args.distributed:
            dist.barrier()
    else:
        dist.barrier()
        dataset = MoleculeDataset(args.dataset)
    
    # Create sampler for distributed training
    if args.distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # DataLoader - num_workers=0 to avoid DeepChem multiprocessing issues
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,  # IMPORTANT: Keep 0 to avoid DeepChem conflicts
        collate_fn=lambda x: x,  # Just return list of SMILES
        drop_last=True,
        pin_memory=True
    )
    
    if is_main:
        print(f"Dataset: {len(dataset)} molecules | Batches per GPU: {len(loader)}")
    
    # Initialize DeepChem featurizer before training (one process at a time)
    if args.distributed:
        for i in range(world_size):
            if rank == i:
                _ = get_featurizer()  # Initialize featurizer
                _ = get_dgl()  # Initialize DGL
            dist.barrier()
    else:
        _ = get_featurizer()
        _ = get_dgl()
    
    if is_main:
        print("Featurizer initialized.")
    
    # Create model
    gnn = create_gnn(
        node_feat_size=NODE_FEAT_SIZE,
        edge_feat_size=EDGE_FEAT_SIZE,
        num_layers=NUM_LAYERS,
        graph_feat_size=GRAPH_FEAT_SIZE,
        dropout=DROPOUT
    ).to(device)
    
    head = MaskHead(GRAPH_FEAT_SIZE, NUM_ATOM_TYPES).to(device)
    
    # Wrap with DDP for distributed training
    if args.distributed:
        gnn = DDP(gnn, device_ids=[local_rank], find_unused_parameters=True)
        head = DDP(head, device_ids=[local_rank])
    
    # Optimizer
    opt = optim.Adam(
        list(gnn.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    
    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None
    
    # TensorBoard writer (main process only)
    writer = None
    if is_main and args.use_tensorboard and HAS_TB:
        log_dir = os.path.join(args.output_dir, 'tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S'))
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard: {log_dir}")
    
    # Training log file
    log_file = os.path.join(args.output_dir, 'train.log') if is_main else None
    if log_file:
        with open(log_file, 'w') as f:
            f.write(f"AttrMasking Pretrain\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"World size: {world_size}\n")
            f.write("-" * 50 + "\n")
    
    # Training loop
    if is_main:
        print("\n" + "=" * 70)
        print("Training Start!")
        print("=" * 70 + "\n")
    
    best_loss = float('inf')
    
    try:
        for epoch in range(1, args.epochs + 1):
            # Set epoch for sampler (important for shuffling in distributed mode)
            if args.distributed:
                sampler.set_epoch(epoch)
            
            t0 = time.time()
            
            avg_loss, avg_acc = train_epoch(
                gnn, head, loader, opt, device, scaler, 
                epoch, writer, rank, args.mask_rate
            )
            
            # Update learning rate
            scheduler.step()
            
            # Logging and saving (main process only)
            if is_main:
                elapsed = time.time() - t0
                print(f"\nEpoch {epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.1f}% | Time: {elapsed:.0f}s")
                
                # Write to log file
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.1f}% | Time: {elapsed:.0f}s\n")
                
                # Save checkpoint
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                
                if epoch % SAVE_INTERVAL == 0 or is_best:
                    # Get state dict (handle DDP wrapper)
                    gnn_state = gnn.module.state_dict() if args.distributed else gnn.state_dict()
                    
                    # Save pretrained GNN weights
                    torch.save(gnn_state, os.path.join(args.output_dir, 'attrmasking_gnn_pretrained.pth'))
                    
                    if is_best:
                        torch.save(gnn_state, os.path.join(args.output_dir, 'attrmasking_gnn_best.pth'))
                        print(f"  -> Best model saved! (loss: {avg_loss:.4f})")
                
                # TensorBoard epoch logging
                if writer:
                    writer.add_scalar('epoch/loss', avg_loss, epoch)
                    writer.add_scalar('epoch/acc', avg_acc * 100, epoch)
                    writer.add_scalar('epoch/lr', scheduler.get_last_lr()[0], epoch)
            
            # Synchronize all processes
            if args.distributed:
                dist.barrier()
    
    except KeyboardInterrupt:
        if is_main:
            print("\n\nTraining interrupted! Saving checkpoint...")
            gnn_state = gnn.module.state_dict() if args.distributed else gnn.state_dict()
            torch.save(gnn_state, os.path.join(args.output_dir, 'attrmasking_gnn_interrupted.pth'))
    
    finally:
        # Cleanup
        if is_main:
            print("\n" + "=" * 70)
            print("Training Complete!")
            print(f"Best loss: {best_loss:.4f}")
            print(f"Model saved to: {args.output_dir}/attrmasking_gnn_pretrained.pth")
            print("=" * 70)
            
            if writer:
                writer.close()
        
        if args.distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()