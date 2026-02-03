"""
快速预处理脚本 (修复版)
======================
"""

import argparse
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import gc

def init_worker():
    """每个worker进程初始化一次"""
    global _featurizer, _Chem
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    import deepchem as dc
    _featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    _Chem = Chem

def process_batch(smiles_batch):
    """处理一批SMILES"""
    global _featurizer, _Chem
    results = []
    
    for smi in smiles_batch:
        try:
            mol = _Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() < 2:
                continue
            
            atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            graphs = _featurizer.featurize([smi])
            
            if graphs[0] is None:
                continue
            
            g = graphs[0].to_dgl_graph(self_loop=True)
            
            results.append({
                'node_feats': g.ndata['x'].clone(),
                'edge_feats': g.edata['edge_attr'].clone(),
                'edges': (g.edges()[0].clone(), g.edges()[1].clone()),
                'atom_types': atom_types,
                'num_nodes': g.num_nodes()
            })
        except:
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--workers', type=int, default=8)  # 减少默认workers
    parser.add_argument('--chunk_size', type=int, default=500)
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input.replace('.txt', '_graphs.pkl')
    
    # 增加文件句柄限制
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
        print(f"File limit: {min(65536, hard)}")
    except:
        pass
    
    # 加载
    print(f"Loading SMILES from {args.input}")
    with open(args.input) as f:
        smiles_list = [line.strip().split()[0] for line in f if line.strip()]
    
    total = len(smiles_list)
    print(f"Total: {total} SMILES")
    print(f"Workers: {args.workers}, Chunk size: {args.chunk_size}")
    
    # 分块
    chunks = [smiles_list[i:i+args.chunk_size] for i in range(0, total, args.chunk_size)]
    
    # 处理并分批保存
    all_data = []
    save_interval = 50000  # 每5万条保存一次中间结果
    
    print("\nProcessing...")
    with Pool(args.workers, initializer=init_worker) as pool:
        for i, batch_results in enumerate(tqdm(pool.imap(process_batch, chunks), total=len(chunks))):
            all_data.extend(batch_results)
            
            # 定期保存中间结果
            if len(all_data) >= save_interval and len(all_data) % save_interval < args.chunk_size:
                gc.collect()
    
    print(f"\nTotal processed: {len(all_data)} graphs")
    
    # 最终保存
    print(f"Saving to {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(all_data, f)
    
    size_gb = os.path.getsize(args.output) / 1024**3
    print(f"Done! Size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()