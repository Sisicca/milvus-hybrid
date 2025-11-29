"""
性能基准测试脚本

测试不同检索模式和参数设置的性能
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_retriever import HybridRetriever


def measure_time(func, *args, **kwargs) -> Tuple[float, any]:
    """
    测量函数执行时间
    
    Returns:
        (执行时间(秒), 函数返回值)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def benchmark_search_modes(retriever: HybridRetriever, queries: List[str]):
    """基准测试不同检索模式"""
    print("\n" + "=" * 80)
    print(" 检索模式性能对比")
    print("=" * 80)
    
    modes = [
        ("稀疏检索 (BM25)", lambda q: retriever.sparse_search(q, limit=5)),
        ("稠密检索 (语义)", lambda q: retriever.dense_search(q, limit=5)),
        ("混合检索 (RRF)", lambda q: retriever.hybrid_search(q, limit=5, use_rrf=True)),
        ("混合检索 (加权)", lambda q: retriever.hybrid_search(q, limit=5, use_rrf=False)),
    ]
    
    for mode_name, search_func in modes:
        total_time = 0
        print(f"\n【{mode_name}】")
        print("-" * 80)
        
        for i, query in enumerate(queries, 1):
            elapsed, results = measure_time(search_func, query)
            total_time += elapsed
            print(f"  查询 {i}: {elapsed:.4f}秒 ({len(results)} 结果)")
        
        avg_time = total_time / len(queries)
        print(f"\n  总时间: {total_time:.4f}秒")
        print(f"  平均时间: {avg_time:.4f}秒")
        print(f"  QPS: {1/avg_time:.2f}")


def benchmark_limit_sizes(retriever: HybridRetriever, query: str):
    """基准测试不同的结果数量限制"""
    print("\n" + "=" * 80)
    print(" 结果数量限制性能对比")
    print("=" * 80)
    print(f"\n查询: {query}\n")
    
    limits = [1, 5, 10, 20, 50]
    
    for limit in limits:
        elapsed, results = measure_time(
            retriever.hybrid_search,
            query,
            limit=limit
        )
        print(f"  limit={limit:2d}: {elapsed:.4f}秒 ({len(results)} 结果)")


def benchmark_ranker_params(retriever: HybridRetriever, query: str):
    """基准测试不同的排序参数"""
    print("\n" + "=" * 80)
    print(" 排序策略参数对比")
    print("=" * 80)
    print(f"\n查询: {query}\n")
    
    # 测试RRF的不同k值
    print("【RRF不同k值】")
    for k in [30, 60, 100]:
        elapsed, results = measure_time(
            retriever.hybrid_search,
            query,
            limit=5,
            use_rrf=True,
            rrf_k=k
        )
        print(f"  RRF (k={k:3d}): {elapsed:.4f}秒")
    
    # 测试加权的不同权重
    print("\n【加权不同权重】")
    weight_configs = [
        (0.5, 0.5, "均衡"),
        (0.7, 0.3, "偏关键词"),
        (0.3, 0.7, "偏语义"),
        (1.0, 0.0, "仅稀疏"),
        (0.0, 1.0, "仅稠密"),
    ]
    
    for sparse_w, dense_w, desc in weight_configs:
        elapsed, results = measure_time(
            retriever.hybrid_search,
            query,
            limit=5,
            use_rrf=False,
            sparse_weight=sparse_w,
            dense_weight=dense_w
        )
        print(f"  {desc} ({sparse_w:.1f}/{dense_w:.1f}): {elapsed:.4f}秒")


def benchmark_with_filters(retriever: HybridRetriever, query: str):
    """基准测试带过滤条件的检索"""
    print("\n" + "=" * 80)
    print(" 过滤条件性能对比")
    print("=" * 80)
    print(f"\n查询: {query}\n")
    
    # 无过滤
    elapsed, results = measure_time(
        retriever.hybrid_search,
        query,
        limit=5
    )
    print(f"  无过滤: {elapsed:.4f}秒 ({len(results)} 结果)")
    
    # 单学科过滤
    elapsed, results = measure_time(
        retriever.hybrid_search,
        query,
        limit=5,
        filter_expr='subject == "Anatomy"'
    )
    print(f"  单学科过滤: {elapsed:.4f}秒 ({len(results)} 结果)")
    
    # 多学科过滤
    elapsed, results = measure_time(
        retriever.hybrid_search,
        query,
        limit=5,
        filter_expr='subject in ["Anatomy", "Physiology", "InternalMed"]'
    )
    print(f"  多学科过滤: {elapsed:.4f}秒 ({len(results)} 结果)")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" Milvus混合检索系统 - 性能基准测试")
    print("=" * 80)
    
    # 初始化检索器
    print("\n正在初始化混合检索器...")
    try:
        retriever = HybridRetriever(
            uri="milvus_db_hub/med_corpus.db",
            collection_name="med_corpus",
            model_path="model-hub/Qwen3-Embedding-0.6B"
        )
        print("✓ 初始化成功！")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return
    
    try:
        # 测试查询
        test_queries = [
            "What is diabetes mellitus?",
            "Describe the cardiovascular system",
            "What are beta blockers?",
            "Explain wound healing",
            "Symptoms of pneumonia",
        ]
        
        # 基准测试1: 不同检索模式
        benchmark_search_modes(retriever, test_queries)
        
        # 基准测试2: 不同结果数量
        benchmark_limit_sizes(retriever, test_queries[0])
        
        # 基准测试3: 不同排序参数
        benchmark_ranker_params(retriever, test_queries[1])
        
        # 基准测试4: 过滤条件
        benchmark_with_filters(retriever, test_queries[2])
        
        print("\n" + "=" * 80)
        print(" 基准测试完成")
        print("=" * 80)
        print("\n注意: 实际性能可能受系统负载、硬件配置等因素影响")
        
    finally:
        # 关闭连接
        retriever.close()
        print("\n✓ 已关闭数据库连接")


if __name__ == "__main__":
    main()

