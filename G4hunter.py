#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import numpy as np
import torch
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pandas as pd
from functools import partial
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置多进程启动方法为'spawn'
if __name__ == '__main__':
    mp.set_start_method('spawn')

# 检查是否可以使用GPU
def check_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_info = []
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_info.append(f"GPU {i}: {gpu_name}")
        logger.info(f"找到 {num_gpus} 个GPU设备: {', '.join(gpu_info)}")
        return num_gpus
    else:
        logger.info("未找到GPU设备，将使用CPU模式")
        return 0

# G4 translate 函数，将DNA序列转换为G4Hunter编码
def g4_translate(seq):
    """
    将DNA序列转换为G4Hunter编码:
    G的连续长度: 1个G=1, 2个G=2, 3个G=3, >3个G=4
    C的连续长度: 1个C=-1, 2个C=-2, 3个C=-3, >3个C=-4
    其他碱基: 0
    """
    result = []
    i = 0
    
    while i < len(seq):
        if seq[i] == 'G':
            count = 1
            while i + count < len(seq) and seq[i + count] == 'G':
                count += 1
            
            value = min(count, 4) if count <= 3 else 4
            result.extend([value] * count)
            i += count
            
        elif seq[i] == 'C':
            count = 1
            while i + count < len(seq) and seq[i + count] == 'C':
                count += 1
            
            value = -min(count, 4) if count <= 3 else -4
            result.extend([value] * count)
            i += count
            
        else:
            result.append(0)
            i += 1
            
    return np.array(result, dtype=np.float32)

# G4 运行平均函数
def g4_runmean(g4_values, window_size=25):
    """计算G4Hunter编码的运行平均值"""
    if len(g4_values) < window_size:
        return np.array([])
    
    # 创建卷积核用于计算滑动窗口平均值
    kernel = np.ones(window_size) / window_size
    
    # 使用np.convolve计算运行平均值
    run_means = np.convolve(g4_values, kernel, mode='valid')
    
    return run_means

# GPU版本的G4Hunter计算
def g4_hunt_gpu(seq_data, window_size=25, threshold=1.5, device_id=0):
    """GPU加速的G4Hunt计算"""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    results = []
    seq_id, seq = seq_data
    
    # 转换序列为G4Hunter编码
    g4_values = g4_translate(str(seq).upper())
    
    if len(g4_values) < window_size:
        return []
    
    # 创建卷积核用于计算滑动窗口平均值
    kernel = torch.ones(window_size, device=device) / window_size
    
    # 转换为张量并移动到GPU
    g4_tensor = torch.tensor(g4_values, device=device)
    
    # 用卷积计算运行平均值
    g4_tensor_padded = torch.nn.functional.pad(g4_tensor.unsqueeze(0).unsqueeze(0), 
                                            (window_size-1, 0), mode='constant')
    run_means = torch.nn.functional.conv1d(g4_tensor_padded, kernel.unsqueeze(0).unsqueeze(0)).squeeze()
    
    # 找到超过阈值的区域 (正向和负向)
    positive_regions = (run_means >= threshold).nonzero().flatten().cpu().numpy()
    negative_regions = (run_means <= -threshold).nonzero().flatten().cpu().numpy()
    
    # 处理正向G4区域
    for start_idx in positive_regions:
        end_idx = start_idx + window_size - 1
        if end_idx >= len(seq):
            continue
            
        region_seq = str(seq[start_idx:end_idx+1])
        g4_score = float(g4_values[start_idx:end_idx+1].mean())
        
        # 优化区域边界
        refined_start, refined_end = refine_boundaries(seq, start_idx, end_idx, 'G')
        if refined_start >= refined_end:
            continue
            
        refined_seq = str(seq[refined_start:refined_end+1])
        refined_score = float(g4_values[refined_start:refined_end+1].mean())
        
        if refined_score == 0.0:
            continue
            
        results.append({
            'chromosome': seq_id,
            'start': refined_start + 1,
            'end': refined_end + 1,
            'strand': '+',
            'score': refined_score,
            'sequence': refined_seq,
            'window': window_size,
            'threshold': threshold
        })
    
    # 处理负向G4区域
    for start_idx in negative_regions:
        end_idx = start_idx + window_size - 1
        if end_idx >= len(seq):
            continue
            
        region_seq = str(seq[start_idx:end_idx+1])
        g4_score = float(g4_values[start_idx:end_idx+1].mean())
        
        # 优化区域边界
        refined_start, refined_end = refine_boundaries(seq, start_idx, end_idx, 'C')
        if refined_start >= refined_end:
            continue
            
        refined_seq = str(seq[refined_start:refined_end+1])
        refined_score = float(g4_values[refined_start:refined_end+1].mean())
        
        if refined_score == 0.0:
            continue
            
        results.append({
            'chromosome': seq_id,
            'start': refined_start + 1,
            'end': refined_end + 1,
            'strand': '-',
            'score': refined_score,
            'sequence': refined_seq,
            'window': window_size,
            'threshold': threshold
        })
    
    return results

# CPU版本的G4Hunter计算
def g4_hunt_cpu(seq_data, window_size=25, threshold=1.5):
    """CPU版本的G4Hunt计算"""
    results = []
    seq_id, seq = seq_data
    
    # 转换序列为G4Hunter编码
    g4_values = g4_translate(str(seq).upper())
    
    if len(g4_values) < window_size:
        return []
    
    # 计算运行平均值
    run_means = g4_runmean(g4_values, window_size)
    
    # 找到超过阈值的区域
    positive_regions = np.where(run_means >= threshold)[0]
    negative_regions = np.where(run_means <= -threshold)[0]
    
    # 处理正向G4区域
    for start_idx in positive_regions:
        end_idx = start_idx + window_size - 1
        region_seq = str(seq[start_idx:end_idx+1])
        g4_score = float(g4_values[start_idx:end_idx+1].mean())
        
        # 优化区域边界 - 找到一个完整的G重复序列
        refined_start, refined_end = refine_boundaries(seq, start_idx, end_idx, 'G')
        refined_seq = str(seq[refined_start:refined_end+1])
        refined_score = float(g4_values[refined_start:refined_end+1].mean())
        
        results.append({
            'chromosome': seq_id,
            'start': refined_start + 1,  # 转换为1-based索引
            'end': refined_end + 1,  # 转换为1-based索引
            'strand': '+',
            'score': refined_score,
            'sequence': refined_seq,
            'window': window_size,
            'threshold': threshold
        })
    
    # 处理负向G4区域
    for start_idx in negative_regions:
        end_idx = start_idx + window_size - 1
        region_seq = str(seq[start_idx:end_idx+1])
        g4_score = float(g4_values[start_idx:end_idx+1].mean())
        
        # 优化区域边界 - 找到一个完整的C重复序列
        refined_start, refined_end = refine_boundaries(seq, start_idx, end_idx, 'C')
        refined_seq = str(seq[refined_start:refined_end+1])
        refined_score = float(g4_values[refined_start:refined_end+1].mean())
        
        results.append({
            'chromosome': seq_id,
            'start': refined_start + 1,  # 转换为1-based索引
            'end': refined_end + 1,  # 转换为1-based索引
            'strand': '-',
            'score': refined_score,
            'sequence': refined_seq,
            'window': window_size,
            'threshold': threshold
        })
    
    return results

# 优化区域边界
def refine_boundaries(seq, start, end, letter):
    """找到序列中完整的G或C重复序列"""
    seq_str = str(seq)
    
    # 添加边界检查
    if start < 0 or end >= len(seq_str) or start > end:
        return start, end
    
    # 确保开始位置有效
    try:
        # 调整开始位置
        if start > 0:
            if seq_str[start] == letter:
                while start > 0 and seq_str[start-1] == letter:
                    start -= 1
            else:
                while start <= end and seq_str[start] != letter:
                    start += 1
                    if start > end:
                        return start, end
        
        # 调整结束位置
        if end < len(seq_str) - 1:
            if seq_str[end] == letter:
                while end < len(seq_str) - 1 and seq_str[end+1] == letter:
                    end += 1
            else:
                while end >= start and seq_str[end] != letter:
                    end -= 1
                    if end < start:
                        return start, end
    except IndexError:
        logger.warning(f"索引越界: start={start}, end={end}, seq_length={len(seq_str)}")
        return start, end
    
    return start, end

def calculate_score(g4_values, start, end):
    """安全地计算G4分数"""
    if start > end or start < 0 or end >= len(g4_values):
        return 0.0
    
    slice_values = g4_values[start:end+1]
    if len(slice_values) == 0:
        return 0.0
    
    return float(slice_values.mean())

# 修改 split_into_batches 函数来处理大文件
def split_into_batches(sequences, num_batches):
    """将序列均匀分成num_batches个批次"""
    batch_size = max(1, len(sequences) // num_batches)
    
    # 返回序列的索引范围，而不是序列本身
    ranges = []
    for i in range(0, len(sequences), batch_size):
        end = min(i + batch_size, len(sequences))
        ranges.append((i, end))
    return ranges

# 修改GPU处理函数以使用序列索引范围
def process_batch_gpu(seq_range, sequences, window_size, threshold, device_id):
    results = []
    start_idx, end_idx = seq_range
    
    for i in range(start_idx, end_idx):
        results.extend(g4_hunt_gpu(sequences[i], window_size, threshold, device_id))
    return results

# 修改CPU处理函数以使用序列索引范围
def process_batch_cpu(seq_range, sequences, window_size, threshold):
    results = []
    start_idx, end_idx = seq_range
    
    for i in range(start_idx, end_idx):
        results.extend(g4_hunt_cpu(sequences[i], window_size, threshold))
    return results

# 合并来自不同处理的结果
def merge_results(results_list):
    all_results = []
    for results in results_list:
        all_results.extend(results)
    return all_results

# 主分析函数
def analyze_genome(input_file, output_dir, window_size=25, threshold=1.5):
    """分析基因组并找到所有G4四链体形成序列"""
    start_time = time.time()
    logger.info(f"开始分析基因组文件: {input_file}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取基因组序列
    sequences = []
    logger.info("读取基因组序列...")
    for record in SeqIO.parse(input_file, "fasta"):
        sequences.append((record.id, record.seq))
    
    logger.info(f"读取了 {len(sequences)} 条序列")
    
    # 检查GPU和CPU数量
    num_gpus = check_gpu()
    num_cpus = mp.cpu_count()
    logger.info(f"可用CPU核心数: {num_cpus}")
    
    all_results = []
    
    # 使用分块处理大型基因组文件
    for i, seq_data in enumerate(sequences):
        logger.info(f"正在处理序列 {i+1}/{len(sequences)}: {seq_data[0]}")
        seq_id, seq = seq_data
        
        if len(seq) < window_size:
            logger.info(f"序列 {seq_id} 长度小于窗口大小，跳过")
            continue
        
        # 分割大序列进行并行处理
        chunk_size = 1000000  # 每块1MB
        chunks = [(j, min(j + chunk_size, len(seq))) for j in range(0, len(seq), chunk_size)]
        
        chunk_results = []
        
        if num_gpus > 0:
            # GPU模式
            logger.info(f"使用 {num_gpus} 个GPU处理序列 {seq_id}, 共 {len(chunks)} 个数据块")
            
            # 使用上下文管理器确保进程池正确关闭
            with ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp.get_context('spawn')) as executor:
                futures = []
                
                for c_idx, (start, end) in enumerate(chunks):
                    gpu_id = c_idx % num_gpus
                    subseq = seq[start:end]
                    subseq_data = (f"{seq_id}:{start}-{end}", subseq)
                    
                    future = executor.submit(g4_hunt_gpu, subseq_data, window_size, threshold, gpu_id)
                    futures.append((future, start))
                
                # 收集结果
                for i, (future, start) in enumerate(futures):
                    try:
                        results = future.result()
                        
                        # 调整坐标到原始序列
                        for result in results:
                            result['start'] += start
                            result['end'] += start
                            result['chromosome'] = seq_id
                        
                        chunk_results.extend(results)
                        
                        if (i + 1) % 10 == 0 or i + 1 == len(futures):
                            logger.info(f"处理进度: {i + 1}/{len(futures)} 块 ({((i + 1) / len(futures) * 100):.1f}%)")
                    
                    except Exception as e:
                        logger.error(f"处理块时发生错误: {e}")
                        continue
        
        else:
            # CPU模式
            logger.info(f"使用 {num_cpus} 个CPU核心处理序列 {seq_id}, 共 {len(chunks)} 个数据块")
            
            with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                futures = []
                
                for start, end in chunks:
                    subseq = seq[start:end]
                    subseq_data = (f"{seq_id}:{start}-{end}", subseq)
                    future = executor.submit(g4_hunt_cpu, subseq_data, window_size, threshold)
                    futures.append((future, start))
                
                for i, (future, start) in enumerate(futures):
                    results = future.result()
                    
                    for result in results:
                        result['start'] += start
                        result['end'] += start
                        result['chromosome'] = seq_id
                    
                    chunk_results.extend(results)
                    
                    if (i + 1) % 10 == 0 or i + 1 == len(futures):
                        logger.info(f"处理进度: {i + 1}/{len(futures)} 块 ({((i + 1) / len(futures) * 100):.1f}%)")
        
        all_results.extend(chunk_results)
        logger.info(f"序列 {seq_id} 处理完成，找到 {len(chunk_results)} 个潜在的G4四链体形成序列")
    
    # 保存结果
    logger.info(f"所有序列处理完成，共找到 {len(all_results)} 个潜在的G4四链体形成序列")
    
    # 创建结果DataFrame
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # 保存为BED格式
        output_bed = os.path.join(output_dir, f"G4Hunter_w{window_size}_s{threshold}.bed")
        with open(output_bed, 'w') as f:
            for _, row in df_results.iterrows():
                # BED格式: chromosome start end name score strand
                f.write(f"{row['chromosome']}\t{row['start']-1}\t{row['end']}\t"
                       f"G4_{row['chromosome']}_{row['start']}_{row['end']}\t"
                       f"{row['score']}\t{row['strand']}\n")
        
        # 保存完整结果为CSV
        output_csv = os.path.join(output_dir, f"G4Hunter_w{window_size}_s{threshold}.csv")
        df_results.to_csv(output_csv, index=False)
        
        logger.info(f"结果已保存为BED格式: {output_bed}")
        logger.info(f"完整结果已保存为CSV格式: {output_csv}")
    else:
        logger.warning("未找到符合条件的G4四链体形成序列")
    
    elapsed_time = time.time() - start_time
    logger.info(f"分析完成，耗时: {elapsed_time:.2f} 秒")

def main():
    parser = argparse.ArgumentParser(description='G4Hunter: 预测基因组中的G4四链体形成序列')
    parser.add_argument('-i', '--input', required=True, help='输入FASTA格式的基因组文件')
    parser.add_argument('-o', '--output', required=True, help='输出目录')
    parser.add_argument('-w', '--window', type=int, default=25, help='滑动窗口大小 (默认: 25)')
    parser.add_argument('-s', '--score', type=float, default=1.5, help='G4Hunter分数阈值 (默认: 1.5)')
    
    args = parser.parse_args()
    
    analyze_genome(args.input, args.output, args.window, args.score)

if __name__ == "__main__":
    main()
