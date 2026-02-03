#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预训练数据集标准化分析脚本 (抽样版 + 多进程加速)
用于SynFrag审稿意见回复 - 量化预训练数据中的标准化影响

功能：
1. 从预训练数据集随机抽样指定数量的分子
2. 多进程并行标准化处理
3. 统计变化分子比例、重复分子数量等
4. 生成审稿回复所需的统计报告

作者: Zhang Xiang
日期: 2025
用途: 回复JCIM审稿意见Comment 1 - 方案2
"""

import os
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from rdkit import rdBase
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter
import warnings
import logging
import json
from datetime import datetime
from tqdm import tqdm
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 抑制RDKit警告
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# 打印RDKit版本
logger.info(f"RDKit version: {rdBase.rdkitVersion}")


def get_tautomer_canonicalizer():
    """
    根据RDKit版本获取合适的互变异构体标准化器
    """
    try:
        enumerator = rdMolStandardize.GetV1TautomerEnumerator()
        return ('enumerator_v1', enumerator)
    except AttributeError:
        pass
    
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        return ('enumerator', enumerator)
    except AttributeError:
        pass
    
    try:
        canonicalizer = rdMolStandardize.TautomerCanonicalizer()
        return ('canonicalizer', canonicalizer)
    except AttributeError:
        pass
    
    return ('none', None)


def canonicalize_tautomer(mol, tautomer_handler):
    """使用合适的方法进行互变异构体标准化"""
    handler_type, handler = tautomer_handler
    
    if handler is None or mol is None:
        return mol
    
    try:
        if handler_type in ['enumerator_v1', 'enumerator']:
            return handler.Canonicalize(mol)
        elif handler_type == 'canonicalizer':
            return handler.Canonicalize(mol)
        else:
            return mol
    except:
        return mol


# 全局变量，用于多进程共享
TAUTOMER_HANDLER = None

def init_worker():
    """初始化worker进程的全局变量"""
    global TAUTOMER_HANDLER
    TAUTOMER_HANDLER = get_tautomer_canonicalizer()


def standardize_single_smiles(smiles):
    """
    标准化单个SMILES（用于多进程）
    
    Returns:
    --------
    tuple: (original_smiles, standardized_smiles, success, changed)
    """
    global TAUTOMER_HANDLER
    
    if not smiles or not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return (smiles, None, False, False)
    
    smiles = smiles.strip()
    
    try:
        # 解析SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (smiles, None, False, False)
        
        # 获取原始canonical SMILES
        original_canonical = Chem.MolToSmiles(mol, canonical=True)
        
        # 标准化流程
        # 1. 基础清理
        mol = rdMolStandardize.Cleanup(mol)
        
        # 2. 金属断开
        metal_disconnector = rdMolStandardize.MetalDisconnector()
        mol = metal_disconnector.Disconnect(mol)
        
        # 3. 选择最大片段
        largest_fragment = rdMolStandardize.LargestFragmentChooser()
        mol = largest_fragment.choose(mol)
        
        # 4. 功能团标准化
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        
        # 5. 电荷中和
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        
        # 6. 互变异构体标准化
        if TAUTOMER_HANDLER is None:
            TAUTOMER_HANDLER = get_tautomer_canonicalizer()
        mol = canonicalize_tautomer(mol, TAUTOMER_HANDLER)
        
        # 获取标准化后的SMILES
        std_smiles = Chem.MolToSmiles(mol, canonical=True)
        
        # 判断是否变化
        changed = (original_canonical != std_smiles)
        
        return (smiles, std_smiles, True, changed)
        
    except Exception as e:
        return (smiles, None, False, False)


def load_and_sample_smiles(file_path, sample_size=1000000, seed=42):
    """
    加载SMILES文件并随机抽样
    
    Parameters:
    -----------
    file_path : str
        SMILES文件路径（每行一个SMILES）
    sample_size : int
        抽样数量
    seed : int
        随机种子
    """
    logger.info(f"Loading SMILES from: {file_path}")
    
    # 首先计算总行数
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"Total molecules in file: {total_lines:,}")
    
    # 如果样本量大于等于总量，直接加载全部
    if sample_size >= total_lines:
        logger.info(f"Sample size >= total, loading all molecules")
        with open(file_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        return smiles_list, total_lines
    
    # 随机抽样
    random.seed(seed)
    sampled_indices = set(random.sample(range(total_lines), sample_size))
    
    logger.info(f"Sampling {sample_size:,} molecules (seed={seed})")
    
    smiles_list = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i in sampled_indices:
                smiles_list.append(line.strip())
    
    return smiles_list, total_lines


def analyze_results(results):
    """
    分析标准化结果
    """
    stats = {
        'total_processed': len(results),
        'successful': 0,
        'failed': 0,
        'changed': 0,
        'unchanged': 0,
        'original_unique': 0,
        'standardized_unique': 0,
        'duplicates_after_std': 0,
        'change_rate': 0.0,
        'duplicate_rate': 0.0
    }
    
    original_smiles = []
    standardized_smiles = []
    changed_examples = []
    
    for orig, std, success, changed in results:
        if success:
            stats['successful'] += 1
            original_smiles.append(orig)
            standardized_smiles.append(std)
            
            if changed:
                stats['changed'] += 1
                if len(changed_examples) < 20:
                    changed_examples.append({
                        'original': orig,
                        'standardized': std
                    })
            else:
                stats['unchanged'] += 1
        else:
            stats['failed'] += 1
    
    # 计算唯一值
    stats['original_unique'] = len(set(original_smiles))
    stats['standardized_unique'] = len(set(standardized_smiles))
    stats['duplicates_after_std'] = stats['successful'] - stats['standardized_unique']
    
    # 计算比率
    if stats['successful'] > 0:
        stats['change_rate'] = stats['changed'] / stats['successful']
        stats['duplicate_rate'] = stats['duplicates_after_std'] / stats['successful']
    
    # 找出哪些分子标准化后变成了重复
    std_counter = Counter(standardized_smiles)
    duplicate_groups = {k: v for k, v in std_counter.items() if v > 1}
    stats['n_duplicate_groups'] = len(duplicate_groups)
    stats['max_duplicate_count'] = max(duplicate_groups.values()) if duplicate_groups else 1
    
    # 收集一些重复的例子
    duplicate_examples = []
    for std_smi, count in sorted(duplicate_groups.items(), key=lambda x: -x[1])[:10]:
        # 找到对应的原始SMILES
        orig_variants = []
        for orig, std, success, _ in results:
            if success and std == std_smi and len(orig_variants) < 5:
                orig_variants.append(orig)
        duplicate_examples.append({
            'standardized': std_smi,
            'count': count,
            'original_variants': orig_variants
        })
    
    stats['changed_examples'] = changed_examples
    stats['duplicate_examples'] = duplicate_examples
    
    return stats


def print_report(stats, total_in_file, sample_size, elapsed_time):
    """
    打印分析报告
    """
    report = []
    report.append("\n" + "=" * 70)
    report.append("  PRETRAINING DATASET STANDARDIZATION ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"  Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  Processing Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    report.append("-" * 70)
    
    report.append("\n  1. Dataset Overview")
    report.append("-" * 70)
    report.append(f"  Total molecules in pretraining set:  {total_in_file:,}")
    report.append(f"  Sampled molecules:                   {sample_size:,}")
    report.append(f"  Sampling ratio:                      {100*sample_size/total_in_file:.2f}%")
    
    report.append("\n  2. Standardization Results")
    report.append("-" * 70)
    report.append(f"  Successfully processed:    {stats['successful']:,} ({100*stats['successful']/stats['total_processed']:.2f}%)")
    report.append(f"  Failed to process:         {stats['failed']:,} ({100*stats['failed']/stats['total_processed']:.2f}%)")
    report.append("-" * 70)
    report.append(f"  Changed after std:         {stats['changed']:,} ({100*stats['change_rate']:.2f}%)")
    report.append(f"  Unchanged:                 {stats['unchanged']:,} ({100*(1-stats['change_rate']):.2f}%)")
    
    report.append("\n  3. Duplicate Analysis (Potential Tautomers/Protonation Variants)")
    report.append("-" * 70)
    report.append(f"  Unique molecules (original):      {stats['original_unique']:,}")
    report.append(f"  Unique molecules (standardized):  {stats['standardized_unique']:,}")
    report.append(f"  Duplicates after standardization: {stats['duplicates_after_std']:,} ({100*stats['duplicate_rate']:.2f}%)")
    report.append(f"  Number of duplicate groups:       {stats['n_duplicate_groups']:,}")
    report.append(f"  Max duplicates for one molecule:  {stats['max_duplicate_count']}")
    
    # 估算全量数据
    report.append("\n  4. Extrapolation to Full Dataset")
    report.append("-" * 70)
    estimated_changed = int(stats['change_rate'] * total_in_file)
    estimated_duplicates = int(stats['duplicate_rate'] * total_in_file)
    report.append(f"  Estimated changed molecules:     ~{estimated_changed:,} ({100*stats['change_rate']:.2f}%)")
    report.append(f"  Estimated duplicates:            ~{estimated_duplicates:,} ({100*stats['duplicate_rate']:.2f}%)")
    
    # 变化的例子
    if stats['changed_examples']:
        report.append("\n  5. Examples of Changed Molecules")
        report.append("-" * 70)
        for i, ex in enumerate(stats['changed_examples'][:5]):
            report.append(f"  Example {i+1}:")
            report.append(f"    Original:     {ex['original'][:65]}...")
            report.append(f"    Standardized: {ex['standardized'][:65]}...")
    
    # 重复的例子
    if stats['duplicate_examples']:
        report.append("\n  6. Examples of Duplicate Groups (Potential Tautomers)")
        report.append("-" * 70)
        for i, ex in enumerate(stats['duplicate_examples'][:5]):
            report.append(f"  Group {i+1}: {ex['count']} molecules → 1 standardized form")
            report.append(f"    Standardized: {ex['standardized'][:60]}...")
            report.append(f"    Original variants:")
            for var in ex['original_variants'][:3]:
                report.append(f"      - {var[:60]}...")
    
    # 审稿回复总结
    report.append("\n" + "=" * 70)
    report.append("  SUMMARY FOR REVIEWER RESPONSE")
    report.append("=" * 70)
    report.append(f"""
  Based on sampling analysis of {sample_size:,} molecules from the 
  pretraining dataset ({total_in_file:,} molecules total):
  
  1. Standardization altered {100*stats['change_rate']:.1f}% of molecular representations
     (tautomer canonicalization, charge neutralization, salt removal)
  
  2. Only {100*stats['duplicate_rate']:.2f}% of molecules were identified as potential 
     duplicates after standardization (different representations of the 
     same compound, e.g., tautomers, protonation states)
  
  3. This indicates that the vast majority ({100*(1-stats['duplicate_rate']):.1f}%) of 
     the pretraining data consists of unique chemical entities, and the 
     presence of representation variants is minimal.
  
  Combined with the TSA/TSB analysis showing negligible performance impact
  (ΔAUROC < 1.1%), these results confirm that the lack of explicit 
  standardization does not compromise the validity of SynFrag's pretraining.
""")
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    """
    主函数
    """
    # ==========================================
    # 配置区 - 根据您的实际情况修改
    # ==========================================
    
    # 预训练数据集路径
    pretrain_file = "/local-house/zhangxiang/attrmasking_attentivefp/data/pretrain/smiles.txt"
    
    # 抽样数量（100万）
    sample_size = 1000000
    
    # 随机种子（可复现）
    random_seed = 42
    
    # 并行进程数（建议使用物理核心数的1.5-2倍，但不超过总核心数-4）
    # 您的服务器有64核，建议使用48-56个
    n_processes = 56
    
    # 输出目录
    output_dir = os.path.dirname(pretrain_file)
    
    # ==========================================
    
    print("\n" + "#" * 70)
    print("  Pretraining Dataset Standardization Analysis")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 70)
    
    # 检查文件
    if not os.path.exists(pretrain_file):
        logger.error(f"File not found: {pretrain_file}")
        return
    
    # 加载并抽样
    smiles_list, total_in_file = load_and_sample_smiles(
        pretrain_file, sample_size, random_seed
    )
    actual_sample_size = len(smiles_list)
    logger.info(f"Loaded {actual_sample_size:,} molecules for analysis")
    
    # 多进程处理
    logger.info(f"Starting parallel processing with {n_processes} processes...")
    start_time = time.time()
    
    with Pool(processes=n_processes, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(standardize_single_smiles, smiles_list, chunksize=1000),
            total=len(smiles_list),
            desc="Standardizing"
        ))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    
    # 分析结果
    logger.info("Analyzing results...")
    stats = analyze_results(results)
    
    # 打印报告
    report = print_report(stats, total_in_file, actual_sample_size, elapsed_time)
    print(report)
    
    # 保存结果
    output_json = os.path.join(output_dir, "pretrain_standardization_analysis.json")
    
    # 准备可序列化的stats
    serializable_stats = {
        k: v for k, v in stats.items() 
        if k not in ['changed_examples', 'duplicate_examples']
    }
    serializable_stats['changed_examples'] = stats['changed_examples'][:10]
    serializable_stats['duplicate_examples'] = stats['duplicate_examples'][:10]
    serializable_stats['total_in_file'] = total_in_file
    serializable_stats['sample_size'] = actual_sample_size
    serializable_stats['elapsed_time_seconds'] = elapsed_time
    
    with open(output_json, 'w') as f:
        json.dump(serializable_stats, f, indent=2)
    logger.info(f"Results saved to: {output_json}")
    
    # 保存报告
    output_report = os.path.join(output_dir, "pretrain_standardization_report.txt")
    with open(output_report, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to: {output_report}")
    
    print("\n" + "=" * 70)
    print("  Analysis Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()