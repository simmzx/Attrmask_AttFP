#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分子标准化脚本 - 用于SynFrag审稿意见回复 (RDKit兼容版)
实现完整的分子标准化流程：
1. 互变异构体标准化 (Tautomer Canonicalization)
2. 电荷中和 (Uncharger)
3. 盐去除/最大片段保留 (LargestFragmentChooser)
4. 金属断开 (MetalDisconnector)
5. 功能团标准化 (Normalizer)

修复: 兼容不同版本的RDKit (2019.x - 2024.x)

作者: Zhang Xiang
日期: 2025
用途: 回复JCIM审稿意见Comment 1
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from rdkit import rdBase
import warnings
import os
from tqdm import tqdm
import logging
import json
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 抑制RDKit警告
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# 检查RDKit版本
RDKIT_VERSION = rdBase.rdkitVersion
logger.info(f"RDKit version: {RDKIT_VERSION}")


def get_tautomer_canonicalizer():
    """
    根据RDKit版本获取合适的互变异构体标准化器
    
    不同版本RDKit的API差异：
    - RDKit < 2020.09: rdMolStandardize.TautomerCanonicalizer()
    - RDKit >= 2020.09: rdMolStandardize.TautomerEnumerator + GetCanonicalTautomer
    - RDKit >= 2022.03: 推荐使用 GetV1TautomerEnumerator()
    """
    # 尝试方法1: 新版本API (2022.03+)
    try:
        enumerator = rdMolStandardize.GetV1TautomerEnumerator()
        logger.info("Using TautomerEnumerator via GetV1TautomerEnumerator() [RDKit 2022.03+]")
        return ('enumerator_v1', enumerator)
    except AttributeError:
        pass
    
    # 尝试方法2: 中间版本API (2020.09 - 2022.03)
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        logger.info("Using TautomerEnumerator() [RDKit 2020.09+]")
        return ('enumerator', enumerator)
    except AttributeError:
        pass
    
    # 尝试方法3: 旧版本API (< 2020.09)
    try:
        canonicalizer = rdMolStandardize.TautomerCanonicalizer()
        logger.info("Using TautomerCanonicalizer() [RDKit < 2020.09]")
        return ('canonicalizer', canonicalizer)
    except AttributeError:
        pass
    
    # 如果都失败，返回None并记录警告
    logger.warning("Could not initialize tautomer standardization - will skip this step")
    return ('none', None)


def canonicalize_tautomer(mol, tautomer_handler):
    """
    使用合适的方法进行互变异构体标准化
    
    Parameters:
    -----------
    mol : rdkit.Chem.Mol
        输入分子
    tautomer_handler : tuple
        (handler_type, handler_object) 从get_tautomer_canonicalizer()获取
        
    Returns:
    --------
    rdkit.Chem.Mol
        标准化后的分子
    """
    handler_type, handler = tautomer_handler
    
    if handler is None or mol is None:
        return mol
    
    try:
        if handler_type == 'enumerator_v1' or handler_type == 'enumerator':
            # 新版本: 使用 Canonicalize 方法
            return handler.Canonicalize(mol)
        elif handler_type == 'canonicalizer':
            # 旧版本: 使用 Canonicalize 方法
            return handler.Canonicalize(mol)
        else:
            return mol
    except Exception as e:
        logger.debug(f"Tautomer canonicalization failed: {e}")
        return mol


class MoleculeStandardizer:
    """
    完整的分子标准化器 (RDKit版本兼容)
    实现审稿人要求的标准化流程
    """
    
    def __init__(self):
        """初始化标准化组件"""
        # 1. 互变异构体标准化器 (版本兼容)
        self.tautomer_handler = get_tautomer_canonicalizer()
        
        # 2. 电荷中和器
        self.uncharger = rdMolStandardize.Uncharger()
        
        # 3. 最大片段选择器（用于盐去除）
        self.largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()
        
        # 4. 金属断开器
        self.metal_disconnector = rdMolStandardize.MetalDisconnector()
        
        # 5. 标准化器（处理功能团标准化）
        self.normalizer = rdMolStandardize.Normalizer()
        
        # 6. 清理函数
        self.cleaner = rdMolStandardize.Cleanup
        
    def standardize_mol(self, mol):
        """
        对单个分子进行完整标准化
        
        Parameters:
        -----------
        mol : rdkit.Chem.Mol
            输入分子对象
            
        Returns:
        --------
        rdkit.Chem.Mol or None
            标准化后的分子对象，失败返回None
        """
        if mol is None:
            return None
            
        try:
            # Step 1: 基础清理
            mol = self.cleaner(mol)
            
            # Step 2: 金属断开
            mol = self.metal_disconnector.Disconnect(mol)
            
            # Step 3: 选择最大片段（盐去除）
            mol = self.largest_fragment_chooser.choose(mol)
            
            # Step 4: 功能团标准化
            mol = self.normalizer.normalize(mol)
            
            # Step 5: 电荷中和
            mol = self.uncharger.uncharge(mol)
            
            # Step 6: 互变异构体标准化 (使用兼容方法)
            mol = canonicalize_tautomer(mol, self.tautomer_handler)
            
            return mol
            
        except Exception as e:
            logger.debug(f"Standardization failed: {e}")
            return None
    
    def standardize_smiles(self, smiles):
        """
        对SMILES字符串进行标准化
        
        Parameters:
        -----------
        smiles : str
            输入SMILES字符串
            
        Returns:
        --------
        tuple: (standardized_smiles, success_flag, change_type)
            change_type: 'unchanged', 'changed', 'invalid', 'parse_error', 'std_error', 'exception'
        """
        if pd.isna(smiles) or not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return None, False, 'invalid'
            
        try:
            # 解析SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, False, 'parse_error'
            
            original_canonical = Chem.MolToSmiles(mol, canonical=True)
            
            # 标准化
            std_mol = self.standardize_mol(mol)
            if std_mol is None:
                return None, False, 'std_error'
            
            # 转换回SMILES
            std_smiles = Chem.MolToSmiles(std_mol, canonical=True)
            
            # 判断变化类型
            if original_canonical == std_smiles:
                change_type = 'unchanged'
            else:
                change_type = 'changed'
            
            return std_smiles, True, change_type
            
        except Exception as e:
            logger.debug(f"Failed to standardize {smiles}: {e}")
            return None, False, 'exception'


def analyze_standardization_changes(df_original, df_output, smiles_column='smiles'):
    """
    详细分析标准化前后的变化
    """
    original_smiles = df_original[smiles_column].tolist()
    standardized_smiles = df_output['smiles'].tolist()
    success_flags = df_output['standardization_success'].tolist()
    
    stats = {
        'total': len(original_smiles),
        'successful': sum(success_flags),
        'failed': len(original_smiles) - sum(success_flags),
        'changed': 0,
        'unchanged': 0,
        'unique_original': 0,
        'unique_standardized': 0,
        'duplicates_after_std': 0,
        'changed_examples': []
    }
    
    valid_original = []
    valid_standardized = []
    
    for i, (orig, std, success) in enumerate(zip(original_smiles, standardized_smiles, success_flags)):
        if success and std is not None:
            valid_original.append(orig)
            valid_standardized.append(std)
            
            # 需要先将原始SMILES规范化再比较
            orig_mol = Chem.MolFromSmiles(orig)
            if orig_mol:
                orig_canonical = Chem.MolToSmiles(orig_mol, canonical=True)
                if orig_canonical != std:
                    stats['changed'] += 1
                    # 记录一些变化的例子（最多10个）
                    if len(stats['changed_examples']) < 10:
                        stats['changed_examples'].append({
                            'index': i,
                            'original': orig,
                            'original_canonical': orig_canonical,
                            'standardized': std
                        })
                else:
                    stats['unchanged'] += 1
            else:
                stats['unchanged'] += 1
    
    # 计算去重统计
    stats['unique_original'] = len(set(valid_original))
    stats['unique_standardized'] = len(set(valid_standardized))
    stats['duplicates_after_std'] = stats['successful'] - stats['unique_standardized']
    
    return stats


def process_dataset(input_path, output_path, smiles_column='smiles'):
    """
    处理单个数据集
    
    输出文件：
    1. {output_path} - 标准化后的干净数据（用于SynFrag预测）
    2. {output_path}_full.csv - 包含所有信息的完整数据
    3. {output_path}_mapping.csv - 原始到标准化的映射表
    """
    logger.info(f"Processing: {input_path}")
    
    # 读取数据
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} molecules")
    
    # 初始化标准化器
    standardizer = MoleculeStandardizer()
    
    # 标准化每个分子
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Standardizing"):
        smiles = row[smiles_column]
        std_smiles, success, change_type = standardizer.standardize_smiles(smiles)
        
        results.append({
            'original_index': idx,
            'original_smiles': smiles,
            'smiles': std_smiles if success else smiles,  # 失败时保留原始
            'standardization_success': success,
            'change_type': change_type
        })
    
    # 创建结果DataFrame
    df_results = pd.DataFrame(results)
    
    # 合并原始数据的其他列
    other_columns = [col for col in df.columns if col != smiles_column]
    for col in other_columns:
        df_results[col] = df[col].values
    
    # 分析变化
    stats = analyze_standardization_changes(df, df_results, smiles_column)
    
    # 保存完整结果
    full_output_path = output_path.replace('.csv', '_full.csv')
    df_results.to_csv(full_output_path, index=False)
    logger.info(f"Saved full results to: {full_output_path}")
    
    # 保存干净版本（只有成功的，只保留必要列）
    df_clean = df_results[df_results['standardization_success'] == True].copy()
    # 保持与原始文件相同的列结构
    clean_columns = ['smiles'] + other_columns
    df_clean = df_clean[clean_columns]
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved clean results to: {output_path}")
    
    # 保存映射表
    mapping_path = output_path.replace('.csv', '_mapping.csv')
    df_mapping = df_results[['original_index', 'original_smiles', 'smiles', 
                             'standardization_success', 'change_type']]
    df_mapping.to_csv(mapping_path, index=False)
    logger.info(f"Saved mapping to: {mapping_path}")
    
    return stats, df_results


def print_stats(stats, dataset_name):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"  Standardization Statistics for {dataset_name}")
    print(f"{'='*60}")
    print(f"  Total molecules:            {stats['total']}")
    print(f"  Successfully standardized:  {stats['successful']} ({100*stats['successful']/stats['total']:.2f}%)")
    print(f"  Failed:                     {stats['failed']} ({100*stats['failed']/stats['total']:.2f}%)")
    print(f"  {'─'*56}")
    print(f"  Changed after standardization:  {stats['changed']} ({100*stats['changed']/stats['successful']:.2f}%)")
    print(f"  Unchanged:                      {stats['unchanged']} ({100*stats['unchanged']/stats['successful']:.2f}%)")
    print(f"  {'─'*56}")
    print(f"  Unique molecules (original):    {stats['unique_original']}")
    print(f"  Unique molecules (standardized):{stats['unique_standardized']}")
    print(f"  Duplicates after standardization: {stats['duplicates_after_std']}")
    
    if stats['changed_examples']:
        print(f"\n  Examples of changed molecules:")
        print(f"  {'─'*56}")
        for i, ex in enumerate(stats['changed_examples'][:5]):
            print(f"  Example {i+1}:")
            print(f"    Original:     {ex['original'][:60]}...")
            print(f"    Standardized: {ex['standardized'][:60]}...")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    # ==========================================
    # 修改此处路径以适应您的环境
    # ==========================================
    base_path = "/local-house/zhangxiang/attrmasking_attentivefp/data/test"
    
    datasets = {
        'TSA': os.path.join(base_path, "TSA.csv"),
        'TSB': os.path.join(base_path, "TSB.csv"),
    }
    
    smiles_column = 'smiles'
    
    # ==========================================
    
    all_stats = {}
    
    for name, input_path in datasets.items():
        print(f"\n{'#'*70}")
        print(f"  Processing {name}")
        print(f"{'#'*70}")
        
        if not os.path.exists(input_path):
            print(f"  Warning: File not found: {input_path}")
            continue
            
        output_path = input_path.replace('.csv', '_standardized.csv')
        
        stats, df = process_dataset(input_path, output_path, smiles_column)
        all_stats[name] = stats
        
        print_stats(stats, name)
    
    # 保存汇总统计
    summary_path = os.path.join(base_path, "standardization_summary.json")
    
    # 转换stats中的numpy类型为Python原生类型
    serializable_stats = {}
    for name, stats in all_stats.items():
        serializable_stats[name] = {
            k: (int(v) if isinstance(v, (np.integer, np.int64)) else 
                float(v) if isinstance(v, (np.floating, np.float64)) else v)
            for k, v in stats.items()
        }
    
    with open(summary_path, 'w') as f:
        json.dump(serializable_stats, f, indent=2, default=str)
    logger.info(f"Saved summary to: {summary_path}")
    
    # 打印用于审稿回复的表格
    print("\n" + "="*70)
    print("  SUMMARY TABLE FOR REVIEWER RESPONSE")
    print("="*70)
    print(f"  {'Dataset':<10} {'Total':<10} {'Success':<10} {'Changed':<10} {'% Changed':<10}")
    print("  " + "-"*50)
    for name, stats in all_stats.items():
        pct_changed = 100 * stats['changed'] / stats['successful'] if stats['successful'] > 0 else 0
        print(f"  {name:<10} {stats['total']:<10} {stats['successful']:<10} {stats['changed']:<10} {pct_changed:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()