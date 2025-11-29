"""
综合测试脚本

演示混合检索器的所有功能，包括：
1. 三种检索模式的比较
2. 不同融合策略的对比
3. 学科过滤检索
4. 自定义过滤表达式
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_retriever import HybridRetriever


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def compare_search_modes(retriever: HybridRetriever, query: str):
    """比较三种检索模式"""
    print_section(f"查询: {query}")
    
    # 1. 稀疏检索
    print("\n【1. 稀疏检索 (BM25)】")
    print("-" * 80)
    sparse_results = retriever.sparse_search(query, limit=3)
    print(retriever.format_results(sparse_results))
    
    # 2. 稠密检索
    print("\n【2. 稠密检索 (语义向量)】")
    print("-" * 80)
    dense_results = retriever.dense_search(query, limit=3)
    print(retriever.format_results(dense_results))
    
    # 3. 混合检索
    print("\n【3. 混合检索 (RRF融合)】")
    print("-" * 80)
    hybrid_results = retriever.hybrid_search(query, limit=3)
    print(retriever.format_results(hybrid_results))


def compare_ranking_strategies(retriever: HybridRetriever, query: str):
    """比较不同的排序策略"""
    print_section(f"排序策略比较: {query}")
    
    # 1. RRF排序
    print("\n【1. RRF排序 (k=60)】")
    print("-" * 80)
    rrf_results = retriever.hybrid_search(
        query, 
        limit=3,
        use_rrf=True,
        rrf_k=60
    )
    print(retriever.format_results(rrf_results))
    
    # 2. 加权排序 - 偏向稀疏
    print("\n【2. 加权排序 - 偏向关键词匹配 (稀疏0.7, 稠密0.3)】")
    print("-" * 80)
    sparse_heavy_results = retriever.hybrid_search(
        query,
        limit=3,
        use_rrf=False,
        sparse_weight=0.7,
        dense_weight=0.3
    )
    print(retriever.format_results(sparse_heavy_results))
    
    # 3. 加权排序 - 偏向稠密
    print("\n【3. 加权排序 - 偏向语义理解 (稀疏0.3, 稠密0.7)】")
    print("-" * 80)
    dense_heavy_results = retriever.hybrid_search(
        query,
        limit=3,
        use_rrf=False,
        sparse_weight=0.3,
        dense_weight=0.7
    )
    print(retriever.format_results(dense_heavy_results))


def test_subject_filtering(retriever: HybridRetriever):
    """测试学科过滤"""
    print_section("学科过滤检索")
    
    test_cases = [
        ("heart anatomy", "Anatomy", "解剖学"),
        ("beta blockers", "Pharmacology", "药理学"),
        ("diabetes pathophysiology", "InternalMed", "内科学"),
    ]
    
    for i, (query, subject, subject_cn) in enumerate(test_cases, 1):
        print(f"\n【测试 {i}】在 {subject_cn} ({subject}) 中检索: '{query}'")
        print("-" * 80)
        results = retriever.search_by_subject(
            query=query,
            subject=subject,
            limit=2,
            search_type="hybrid"
        )
        print(retriever.format_results(results))


def test_custom_filters(retriever: HybridRetriever):
    """测试自定义过滤表达式"""
    print_section("自定义过滤表达式")
    
    # 多学科过滤
    query = "treatment guidelines"
    print(f"\n查询: '{query}'")
    print("过滤条件: 仅在药理学和内科学中检索")
    print("-" * 80)
    
    results = retriever.hybrid_search(
        query=query,
        limit=3,
        filter_expr='subject in ["Pharmacology", "InternalMed"]'
    )
    print(retriever.format_results(results))


def test_medical_queries(retriever: HybridRetriever):
    """测试常见医学查询"""
    print_section("常见医学查询测试")
    
    queries = [
        "What are the symptoms of myocardial infarction?",
        "Mechanism of action of ACE inhibitors",
        "Stages of wound healing",
        "Diagnosis of pneumonia",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n【查询 {i}】{query}")
        print("-" * 80)
        results = retriever.hybrid_search(query, limit=2)
        # 只显示前100个字符的内容
        for j, result in enumerate(results, 1):
            score = result.get('distance', 0)
            subject = result.get('entity', {}).get('subject', 'Unknown')
            content = result.get('entity', {}).get('content', '')[:100] + "..."
            print(f"{j}. Score: {score:.4f} | Subject: {subject}")
            print(f"   Content: {content}\n")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" Milvus混合检索系统 - 综合测试")
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
        # 测试1: 比较不同检索模式
        compare_search_modes(
            retriever,
            "What is positron emission tomography?"
        )
        
        # 测试2: 比较排序策略
        compare_ranking_strategies(
            retriever,
            "diabetes mellitus treatment"
        )
        
        # 测试3: 学科过滤
        test_subject_filtering(retriever)
        
        # 测试4: 自定义过滤
        test_custom_filters(retriever)
        
        # 测试5: 医学查询
        test_medical_queries(retriever)
        
        print_section("测试完成")
        print("\n所有测试已完成！\n")
        
    finally:
        # 关闭连接
        retriever.close()
        print("✓ 已关闭数据库连接")


if __name__ == "__main__":
    main()

