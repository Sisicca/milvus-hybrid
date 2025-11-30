"""
混合检索器使用示例

演示如何使用HybridRetriever进行各种类型的检索
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_retriever import HybridRetriever


def main():
    """主函数"""
    
    # 初始化检索器
    print("正在初始化混合检索器...")
    retriever = HybridRetriever(
        uri="milvus_db_hub/med_corpus.db",
        collection_name="med_corpus",
        model_path="./model-hub/Qwen3-Embedding-0.6B"
    )
    print("初始化完成！\n")
    
    # 定义测试查询
    queries = [
        "What is positron emission tomography?",
        "Describe the anatomy of the heart",
        "What are the symptoms of diabetes?",
        "Explain the mechanism of action of beta blockers"
    ]
    
    # 对每个查询执行检索
    for query in queries:
        print("=" * 80)
        print(f"查询: {query}")
        print("=" * 80)
        
        # 执行混合检索
        results = retriever.hybrid_search(query, limit=3)
        
        # 显示结果
        print(retriever.format_results(results))
        print("\n")
    
    # 演示学科过滤
    print("=" * 80)
    print("学科过滤示例：在药理学中检索")
    print("=" * 80)
    subject_results = retriever.search_by_subject(
        query="beta blockers",
        subject="Pharmacology",
        limit=3,
        search_type="hybrid"
    )
    print(retriever.format_results(subject_results))
    
    # 关闭连接
    retriever.close()
    print("\n检索完成！")


if __name__ == "__main__":
    main()

